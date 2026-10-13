"""Record / replay of the ACE event stream a session consumed, keyed by trainer step.

Continuous ACE mapping runs asynchronously next to training, so two runs on the same cached map
still see different pose streams (drone, 2k steps: 21.2-22.2 dB run to run). A recording pins
that stream: every event the session applied (map, pose, reloc, error) is written with the
trainer step it was applied at, and a replay feeds the same events back at the same steps
from a ReplaySidecar instead of the ACE process. Trainer-side A/Bs then differ only in what
they change (and the replay needs no ACE GPU time).

    record   StreamConfig.record = path    (session._on_event appends to an Arrow IPC file)
    replay   StreamConfig.replay = path    (ReplaySidecar stands in for AceSidecar)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from .schema import EVENT_SCHEMA, event_batch

logger = logging.getLogger(__name__)

RECORD_SCHEMA = pa.schema(list(EVENT_SCHEMA) + [
    pa.field("step", pa.int64()),       # trainer step the event was applied at (-1: before training)
    pa.field("seq", pa.int64()),        # event number (rows of one event share it)
])


def _encodable(row: Dict) -> Dict:
    """A decoded event row (numpy arrays) back to EVENT_SCHEMA values."""
    out = dict(row)
    w2c = out.get("w2c")
    out["w2c"] = None if w2c is None else np.asarray(w2c, np.float64).reshape(-1).tolist()
    for k in ("points", "pix"):
        v = out.get(k)
        out[k] = [] if v is None else np.asarray(v, np.float32).reshape(-1).tolist()
    return out


class EventRecorder:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sink = pa.OSFile(str(self.path), "wb")
        self._writer = pa.ipc.new_stream(self._sink, RECORD_SCHEMA)
        self._lock = threading.Lock()
        self._seq = 0
        self.events = 0

    def add(self, kind: str, rows: List[Dict], step: int):
        if not rows:
            return
        batch = event_batch([_encodable(dict(r, kind=kind)) for r in rows])
        with self._lock:
            if self._writer is None:
                return
            n = batch.num_rows
            cols = batch.columns + [pa.array([int(step)] * n, pa.int64()), pa.array([self._seq] * n, pa.int64())]
            self._writer.write_batch(pa.RecordBatch.from_arrays(cols, schema=RECORD_SCHEMA))
            self._seq += 1
            self.events += 1

    def close(self):
        with self._lock:
            if self._writer is not None:
                self._writer.close()
                self._sink.close()
                self._writer = None


def load_recording(path: str) -> List[Tuple[int, str, List[Dict]]]:
    """[(step, kind, rows)] in recorded order; rows decoded like the live sidecar's."""
    from .sidecar import decode_event_rows

    with pa.OSFile(str(path), "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    events: List[Tuple[int, str, List[Dict]]] = []
    if table.num_rows == 0:
        return events
    seqs = table.column("seq").to_numpy()
    starts = np.flatnonzero(np.r_[True, seqs[1:] != seqs[:-1]])
    ends = np.r_[starts[1:], len(seqs)]
    ev = table.select([f.name for f in EVENT_SCHEMA])
    steps = table.column("step").to_numpy()
    for a, b in zip(starts, ends):
        rows = decode_event_rows(ev.slice(a, b - a).to_batches()[0] if b > a else None)
        events.append((int(steps[a]), rows[0]["kind"], rows))
    return events


class ReplaySidecar:
    """Stands in for AceSidecar: emits a recording's events instead of running ACE.

    A map event is released once the session asks for its map (bootstrap / load / refit) and
    pose rows once the session has sent their frame for localization; every event also waits
    for the trainer to reach its recorded step. Mapping commands are no-ops."""

    def __init__(self, path: str, on_event: Callable[[str, List[Dict]], None],
                 step_fn: Callable[[], int], frame_timeout: float = 60.0,
                 frames_dir: Optional[str] = None):
        self.events = load_recording(path)
        if frames_dir is not None:
            # Frame files live in the recording session's work dir (per process): same names here
            for _, _, rows in self.events:
                for row in rows:
                    if row.get("frame"):
                        row["frame"] = str(Path(frames_dir) / Path(row["frame"]).name)
        self.on_event = on_event
        self.step_fn = step_fn
        self.frame_timeout = frame_timeout
        self.ready = threading.Event()
        self.ready.set()
        self._cv = threading.Condition()
        self._maps_requested = 0
        self._requested: set = set()
        self._closed = False
        self.emitted = self.skipped = 0
        logger.info(f"[Stream] Replaying {len(self.events)} ACE events from {path} (no ACE process)")
        self._thread = threading.Thread(target=self._run, name="ace-replay", daemon=True)
        self._thread.start()

    @property
    def alive(self) -> bool:
        return not self._closed

    def _wait(self, pred, timeout=None) -> bool:
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            while not pred() and not self._closed:
                left = None if deadline is None else deadline - time.time()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(0.05 if left is None else min(0.05, left))
        return not self._closed

    def _run(self):
        maps_seen = 0
        for step, kind, rows in self.events:
            if self._closed:
                return
            if kind == "map":
                maps_seen += 1
                need = maps_seen
                if not self._wait(lambda: self._maps_requested >= need):
                    return
            elif kind == "pose":
                frame = rows[0]["frame"]
                if not self._wait(lambda: frame in self._requested, self.frame_timeout):
                    if self._closed:
                        return
                    logger.warning(f"[Stream] replay: {frame} was never sent for localization; skipped")
                    self.skipped += 1
                    continue
            # Trainer step (the trainer polls; the session sets trainer_step every step)
            if not self._wait(lambda: self.step_fn() >= step):
                return
            try:
                self.on_event(kind, rows)
                self.emitted += 1
            except Exception:
                logger.exception("replayed event handler failed")
        logger.info(f"[Stream] replay finished: {self.emitted} events, {self.skipped} skipped")

    def _notify(self, fn):
        with self._cv:
            fn()
            self._cv.notify_all()

    # AceSidecar interface ------------------------------------------------------------------
    def bootstrap(self, frames, args, focal=-1.0, cache_dir=""):
        self._notify(lambda: setattr(self, "_maps_requested", self._maps_requested + 1))

    def load(self, head, pose_file, frames, image_resolution=480):
        self._notify(lambda: setattr(self, "_maps_requested", self._maps_requested + 1))

    def seed(self, frame, depth, **params):
        self._notify(lambda: setattr(self, "_maps_requested", self._maps_requested + 1))

    def refit(self, frames, iterations, conf_threshold, args):
        self._notify(lambda: setattr(self, "_maps_requested", self._maps_requested + 1))

    def localize(self, frame: str):
        self._notify(lambda: self._requested.add(frame))

    def mapper(self, frames, **params):
        pass

    def tick(self, n: int = 1):
        pass

    def forget(self, frames):
        pass

    def rings(self, **paths):
        pass

    def profile(self, units, out_dir):
        pass

    def close(self, timeout: float = 8.0):
        self._notify(lambda: setattr(self, "_closed", True))
        self._thread.join(timeout=2)
