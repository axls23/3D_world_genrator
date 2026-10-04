"""Bounded in-memory keyframe store; the only place stream data lives.

Keyframes are held as encoded JPEG in Arrow buffers. ``views()`` builds an immutable Arrow
table (VIEW_SCHEMA) for the trainer whose jpeg column *references* those buffers, so a
snapshot costs no image copies and the store's footprint is exactly what it reports.

Budget: at most ``max_keyframes`` real keyframes and ``budget_mb`` of encoded data (train
JPEGs + the ACE-resolution copies on tmpfs + Difix pseudo views). Over budget, it evicts in
this order, never touching the newest ``protect_recent`` keyframes:

  1. frames still unlocalized ``pending_ttl`` seconds after a map exists (the camera moved on),
  2. frames ACE could not register (confidence below the threshold),
  3. the most redundant registered frame: the one closest to its nearest neighbour in
     pose space (centre distance in units of the median keyframe spacing + angle / 15 deg),
     lower confidence first on ties. Val frames compete only with val frames.

Pseudo views are not evicted one by one: each Difix round replaces the previous round's set.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from . import native
from .schema import VIEW_SCHEMA

ANGLE_UNIT_DEG = 15.0


@dataclass
class Keyframe:
    kid: int
    name: str
    t: float
    jpeg: pa.Buffer
    width: int
    height: int
    split: str
    sharpness: float = 0.0
    created: float = field(default_factory=time.time)
    ace_path: Optional[str] = None
    ace_bytes: int = 0
    w2c: Optional[np.ndarray] = None   # ACE map frame, world-to-camera
    conf: float = 0.0
    map_version: int = 0
    seeded: bool = False
    pinned: bool = False               # bootstrap candidates: never evicted before the map exists

    @property
    def nbytes(self) -> int:
        return self.jpeg.size + self.ace_bytes


@dataclass
class PseudoView:
    name: str
    jpeg: pa.Buffer
    c2w: np.ndarray                    # trainer (normalized) frame
    K: np.ndarray
    width: int
    height: int
    weight: float
    conf: float
    src_kid: int


def _binary_chunk(buf: pa.Buffer) -> pa.Array:
    """One-element binary array over an existing buffer (no copy)."""
    offsets = pa.py_buffer(np.array([0, buf.size], dtype=np.int32).tobytes())
    return pa.Array.from_buffers(pa.binary(), 1, [None, offsets, buf])


class KeyframeStore:
    def __init__(self, budget_mb: float = 256, max_keyframes: int = 400, test_every: int = 8,
                 protect_recent: int = 8, pending_ttl: float = 30.0, conf_threshold: float = 500.0):
        self.budget_bytes = int(budget_mb * 1024 * 1024)
        self.max_keyframes = max_keyframes
        self.test_every = test_every
        self.protect_recent = protect_recent
        self.pending_ttl = pending_ttl
        self.conf_threshold = conf_threshold
        self._frames: Dict[int, Keyframe] = {}
        self._pseudo: List[PseudoView] = []
        self._next_kid = 0
        self._lock = threading.RLock()
        self.version = 0          # bumps on any change the trainer should see
        self.evicted = 0
        self.map_ready_at: Optional[float] = None  # pending_ttl only runs once a map exists

    # ---------------------------------------------------------------- mutation
    def add(self, jpeg: bytes, width: int, height: int, t: float, sharpness: float = 0.0) -> Keyframe:
        with self._lock:
            kid = self._next_kid
            self._next_kid += 1
            split = "val" if self.test_every > 0 and kid > 0 and kid % self.test_every == 0 else "train"
            kf = Keyframe(kid, f"kf_{kid:06d}", t, pa.py_buffer(jpeg), width, height, split, sharpness)
            self._frames[kid] = kf
            return kf

    def set_ace_file(self, kid: int, path: str, nbytes: int):
        with self._lock:
            if kid in self._frames:
                self._frames[kid].ace_path, self._frames[kid].ace_bytes = path, nbytes

    def update_pose(self, kid: int, w2c: Optional[np.ndarray], conf: float, map_version: int) -> bool:
        """Returns True if the frame is registered after the update."""
        with self._lock:
            kf = self._frames.get(kid)
            if kf is None:
                return False
            kf.w2c, kf.conf, kf.map_version = (None if w2c is None else np.asarray(w2c, np.float64)), float(conf), map_version
            self.version += 1
            return self.is_registered(kf)

    def set_pseudo(self, views: List[PseudoView]):
        with self._lock:
            self._pseudo = list(views)
            self.version += 1

    def mark_seeded(self, kid: int):
        with self._lock:
            if kid in self._frames:
                self._frames[kid].seeded = True

    # ---------------------------------------------------------------- queries
    def is_registered(self, kf: Keyframe) -> bool:
        return kf.w2c is not None and kf.conf >= self.conf_threshold

    def get(self, kid: int) -> Optional[Keyframe]:
        with self._lock:
            return self._frames.get(kid)

    def by_ace_path(self) -> Dict[str, Keyframe]:
        with self._lock:
            return {kf.ace_path: kf for kf in self._frames.values() if kf.ace_path}

    def frames(self) -> List[Keyframe]:
        with self._lock:
            return sorted(self._frames.values(), key=lambda k: k.kid)

    def registered(self, split: Optional[str] = None) -> List[Keyframe]:
        return [kf for kf in self.frames() if self.is_registered(kf) and (split is None or kf.split == split)]

    def pending(self) -> List[Keyframe]:
        return [kf for kf in self.frames() if kf.w2c is None]

    def unregistered(self) -> List[Keyframe]:
        return [kf for kf in self.frames() if kf.w2c is not None and not self.is_registered(kf)]

    def pseudo(self) -> List[PseudoView]:
        with self._lock:
            return list(self._pseudo)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(kf.nbytes for kf in self._frames.values()) + sum(p.jpeg.size for p in self._pseudo)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    # ---------------------------------------------------------------- eviction
    def evict_if_needed(self) -> List[Keyframe]:
        evicted = []
        with self._lock:
            while len(self._frames) > self.max_keyframes or self.nbytes > self.budget_bytes:
                victim = self._pick_victim()
                if victim is None:
                    break
                del self._frames[victim.kid]
                evicted.append(victim)
            if evicted:
                self.evicted += len(evicted)
                self.version += 1
        return evicted

    def _pick_victim(self) -> Optional[Keyframe]:
        frames = sorted(self._frames.values(), key=lambda k: k.kid)
        candidates = frames[:-self.protect_recent] if self.protect_recent else frames
        candidates = [k for k in candidates if not k.pinned]
        if not candidates:
            return None
        now = time.time()
        stale = [k for k in candidates if k.w2c is None and self.map_ready_at is not None
                 and now - max(k.created, self.map_ready_at) > self.pending_ttl]
        if stale:
            return stale[0]
        failed = [k for k in candidates if k.w2c is not None and not self.is_registered(k)]
        if failed:
            return min(failed, key=lambda k: (k.conf, k.kid))
        registered = [k for k in frames if self.is_registered(k)]
        cand_ids = {k.kid for k in candidates}
        best, best_key = None, None
        for split in ("train", "val"):
            pool = [k for k in registered if k.split == split]
            if len(pool) < 2:
                continue
            c2ws = np.linalg.inv(np.stack([k.w2c for k in pool]))
            spacing = float(np.median(np.linalg.norm(np.diff(c2ws[:, :3, 3], axis=0), axis=1))) or 1.0
            query = np.array([k.kid in cand_ids for k in pool], dtype=np.uint8)
            nn = native.pose_nn_distance(c2ws, query, spacing, ANGLE_UNIT_DEG)  # C, O(N^2)
            for k, d, q in zip(pool, nn, query):
                if q and (best_key is None or (d, k.conf) < best_key):
                    best, best_key = k, (d, k.conf)
        if best is not None:
            return best
        # Nothing registered to compare: drop the oldest unprotected frame
        return candidates[0]

    # ---------------------------------------------------------------- trainer snapshot
    def views(self, to_trainer: Callable[[np.ndarray], np.ndarray],
              intrinsics: Callable[[int, int], np.ndarray],
              weight_of: Callable[[Keyframe], float]) -> Tuple[int, pa.Table]:
        """(version, VIEW_SCHEMA table) of registered keyframes + pseudo views.

        to_trainer maps an ACE-frame camera-to-world to the trainer's normalized frame;
        intrinsics(width, height) gives K for a frame size; weight_of a keyframe's loss weight.
        """
        with self._lock:
            version = self.version
            reg = self.registered()
            rows, chunks = [], []
            for kf in reg:
                rows.append({"kid": kf.kid, "name": kf.name, "kind": "real", "split": kf.split,
                             "c2w": to_trainer(np.linalg.inv(kf.w2c)).reshape(-1).tolist(),
                             "K": intrinsics(kf.width, kf.height).reshape(-1).tolist(),
                             "width": kf.width, "height": kf.height, "weight": weight_of(kf),
                             "conf": kf.conf, "src_kid": kf.kid})
                chunks.append(_binary_chunk(kf.jpeg))
            for i, p in enumerate(self._pseudo):
                rows.append({"kid": -1 - i, "name": p.name, "kind": "pseudo", "split": "train",
                             "c2w": np.asarray(p.c2w).reshape(-1).tolist(),
                             "K": np.asarray(p.K).reshape(-1).tolist(), "width": p.width,
                             "height": p.height, "weight": p.weight, "conf": p.conf,
                             "src_kid": p.src_kid})
                chunks.append(_binary_chunk(p.jpeg))
        meta_schema = pa.schema([f for f in VIEW_SCHEMA if f.name != "jpeg"])
        meta = pa.Table.from_pylist(rows, schema=meta_schema)
        jpeg = pa.chunked_array(chunks, type=pa.binary()) if chunks else pa.chunked_array([], type=pa.binary())
        return version, meta.append_column(VIEW_SCHEMA.field("jpeg"), jpeg)
