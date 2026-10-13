"""Shared-memory rings between the ACE sidecar and the trainer (the per-step hot path).

The Arrow pipe (schema.py) keeps carrying commands and whole-map events; what happens every
trainer step - live localizations, continuous-mapper relocalizations with their seed points,
and the trainer's per-view feedback - moves through fixed-size records in /dev/shm instead:

    sidecar --DELTA ring (pose rows)---------------> trainer
    sidecar --SEED ring (inlier points per pose)---> trainer
    sidecar <--FEEDBACK ring (per-view loss/trust)-- trainer

Each ring is single-producer / single-consumer (csrc/streamops.c ring_*): no locks, no
serialization, and a full ring refuses records instead of growing (backpressure the producer
sees). The record layouts below are the contract with the CUDA side (hypersplat/stream/cuda)
and with the Arrow record/replay schemas. Python 3.8-compatible: the sidecar imports it.
"""

import ctypes
import mmap
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import native

# Record kinds (DELTA_DTYPE.kind)
KIND_POSE = 0     # live localization of a new keyframe
KIND_RELOC = 1    # continuous-mapper relocalization
KIND_ANCHOR = 2   # relocalization that anchored the frame (it is not relocalized again)
KIND_DROP = 3     # frame left the mapper (evicted / forgotten)

DELTA_DTYPE = np.dtype([
    ("seq", "<u8"),           # producer's running record number
    ("step", "<i8"),          # trainer step the sidecar last ticked for (-1: not ticking)
    ("kid", "<i8"),           # keyframe id (from the kf_<kid>.jpg frame name)
    ("mapper_tick", "<i8"),   # sidecar mapper units so far
    ("map_version", "<i4"),
    ("kind", "u1"),
    ("anchored", "u1"),       # sidecar anchor state after this row
    ("_pad0", "u1", (2,)),
    ("conf", "<f4"),          # RANSAC inlier count (ACE confidence)
    ("inlier_frac", "<f4"),   # mapper head fit (inlier EMA) when the row was made; NaN if none
    ("move", "<f4"),          # camera centre move vs the previous adopted pose / path extent; NaN
    ("focal", "<f4"),         # px, of the frame files
    ("w2c", "<f8", (16,)),    # row-major world-to-camera, ACE map frame
    ("n_seeds", "<i4"),       # points pushed to the SEED ring for this row
    ("_pad1", "u1", (4,)),
])
assert DELTA_DTYPE.itemsize == 192, DELTA_DTYPE.itemsize

SEED_MAX_POINTS = 1536
SEED_DTYPE = np.dtype([
    ("seq", "<u8"),           # DELTA seq this belongs to
    ("kid", "<i8"),
    ("map_version", "<i4"),
    ("n", "<i4"),
    ("xyz", "<f4", (SEED_MAX_POINTS, 3)),
    ("pix", "<f4", (SEED_MAX_POINTS, 2)),   # normalized [0, 1] image positions
])
assert SEED_DTYPE.itemsize % 8 == 0, SEED_DTYPE.itemsize

FEEDBACK_DTYPE = np.dtype([
    ("kid", "<i8"),
    ("step", "<i8"),
    ("loss_ema", "<f4"),
    ("psnr_ema", "<f4"),
    ("pose_trust", "<f4"),    # 0..1; NaN when the trainer does not refine poses
    ("tier", "u1"),
    ("_pad0", "u1", (3,)),
    ("exposure", "<f4", (6,)),
    ("pose_delta", "<f8", (16,)),  # refined c2w relative to the input c2w (identity if none)
])
assert FEEDBACK_DTYPE.itemsize % 8 == 0, FEEDBACK_DTYPE.itemsize


def frame_kid(path: str) -> int:
    """Keyframe id of a stream frame file (kf_000123.jpg -> 123); -1 if it is not one."""
    stem = Path(path).stem
    if stem.startswith("kf_"):
        try:
            return int(stem[3:])
        except ValueError:
            pass
    return -1


class ShmRing:
    """One single-producer / single-consumer ring of ``dtype`` records in a shared file."""

    def __init__(self, path: str, dtype: np.dtype, capacity: Optional[int] = None):
        """capacity given: create (or reset) the ring file; None: open an existing one."""
        L = native.lib()
        if L is None:
            raise RuntimeError("shared-memory rings need the native streamops library")
        self.path, self.dtype, self._L = str(path), np.dtype(dtype), L
        rec = self.dtype.itemsize
        if capacity is not None:
            size = int(L.ring_bytes(rec, int(capacity)))
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.ftruncate(fd, size)
                self._mm = mmap.mmap(fd, size)
            finally:
                os.close(fd)
        else:
            fd = os.open(self.path, os.O_RDWR)
            try:
                self._mm = mmap.mmap(fd, os.fstat(fd).st_size)
            finally:
                os.close(fd)
        self._buf = (ctypes.c_char * len(self._mm)).from_buffer(self._mm)
        self._base = ctypes.addressof(self._buf)
        if capacity is not None:
            if L.ring_init(self._base, len(self._mm), rec, int(capacity)) != 0:
                self.close()
                raise ValueError("ring capacity must be a power of two")
        self.capacity = int(L.ring_check(self._base, rec))
        if self.capacity < 0:
            self.close()
            raise ValueError("%s is not a ring of %d-byte records" % (self.path, rec))
        self._stats = (ctypes.c_int64 * 3)()

    def push(self, records: np.ndarray) -> int:
        """Publish records (a 1-D array of this ring's dtype); returns how many fit."""
        recs = np.ascontiguousarray(records, dtype=self.dtype)
        if not len(recs):
            return 0
        return int(self._L.ring_push(self._base, recs.ctypes.data, len(recs)))

    def pop(self, max_n: int = 1 << 30) -> np.ndarray:
        """Consume up to max_n records (a copy, in publish order)."""
        waiting = self.stats()[0]
        n = min(int(max_n), waiting)
        out = np.empty(n, dtype=self.dtype)
        if n:
            got = int(self._L.ring_pop(self._base, out.ctypes.data, n))
            out = out[:got]
        return out

    def stats(self) -> Tuple[int, int, int]:
        """(records waiting, published in total, refused because the ring was full)."""
        self._L.ring_stats(self._base, self._stats)
        return int(self._stats[0]), int(self._stats[1]), int(self._stats[2])

    @property
    def free(self) -> int:
        return self.capacity - self.stats()[0]

    def close(self):
        if getattr(self, "_mm", None) is None:
            return
        self._buf = None
        try:
            self._mm.close()
        except BufferError:
            pass
        self._mm = None

    def unlink(self):
        self.close()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
