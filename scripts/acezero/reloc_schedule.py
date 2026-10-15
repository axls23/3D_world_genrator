"""Relocalization scheduling for the continuous ACE mapper (stream_sidecar.py).

Stdlib only and Python 3.8 syntax: it runs in the ACE-Zero env and is unit-tested from the
trainer's env.

* Frontier first: among unregistered frames, the next relocalization goes to the one whose
  keyframe index is nearest a registered frame. ACE-Zero grows a map from frames that overlap
  what it already covers, so the map spreads outward instead of cycling through every failure.
* Backoff: a frame that failed is retried as soon as the map grows (a frame joined it), or else
  after ``backoff_ticks * 2**fails`` mapper ticks (multiplier capped at ``backoff_cap``). Ticks
  keep coming while the mapper idles (the trainer grants one per 3DGS step), so a frame never
  waits forever.
* Idle: the mapper has no work when nothing was added or changed (``dirty``), the map fits its
  frames (inlier EMA >= ``idle_inliers``), every buffered frame is anchored and no failed frame
  is due for a retry.
"""

import bisect
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


class RelocScheduler:
    def __init__(self, backoff_ticks: int = 8, backoff_cap: int = 64, idle_inliers: float = 0.85):
        self.backoff_ticks = int(backoff_ticks)
        self.backoff_cap = int(backoff_cap)
        self.idle_inliers = float(idle_inliers)
        self.growth = 0                 # frames that joined the map so far
        self.ticks = 0                  # mapper ticks consumed (active + idle): the backoff clock
        self.idle_ticks = 0
        self.dirty = True               # something changed that the mapper should train on
        self.fails = {}                 # type: Dict[str, Tuple[int, int, int]]  # frame -> (fails, growth, tick)
        self._pos = 0

    def configure(self, cfg: Dict) -> None:
        self.backoff_ticks = int(cfg.get("retry_backoff_ticks", self.backoff_ticks))
        self.backoff_cap = int(cfg.get("retry_backoff_cap", self.backoff_cap))
        self.idle_inliers = float(cfg.get("idle_inliers", self.idle_inliers))

    @staticmethod
    def frame_index(frame: str) -> int:
        """Keyframe index from the file name (kf_000624.jpg -> 624); -1 if there is none."""
        digits = re.findall(r"\d+", Path(frame).stem)
        return int(digits[-1]) if digits else -1

    # ------------------------------------------------------------------ events
    def tick(self, idle: bool) -> None:
        self.ticks += 1
        if idle:
            self.idle_ticks += 1

    def note_failure(self, frame: str) -> None:
        fails = self.fails.get(frame, (0, 0, 0))[0] + 1
        self.fails[frame] = (fails, self.growth, self.ticks)

    def note_joined(self, frame: str) -> None:
        self.growth += 1
        self.fails.pop(frame, None)
        self.dirty = True

    def forget(self, frame: str) -> None:
        self.fails.pop(frame, None)
        self.dirty = True

    @property
    def waiting(self) -> int:
        return len(self.fails)

    # ------------------------------------------------------------------ decisions
    def retry_ok(self, frame: str) -> bool:
        st = self.fails.get(frame)
        if st is None:
            return True
        fails, growth_at, tick_at = st
        wait = self.backoff_ticks * min(2 ** fails, self.backoff_cap)
        return self.growth > growth_at or self.ticks >= tick_at + wait

    def pick(self, known: Iterable[str], registered: Set[str], anchored: Set[str]) -> Optional[str]:
        """Next frame to relocalize, or None. Alternates between refining registered but
        unanchored frames and expanding the frontier (nearest keyframe index first)."""
        known = list(known)
        refine = [f for f in known if f in registered and f not in anchored]
        frontier = [f for f in known if f not in registered and self.retry_ok(f)]
        if frontier:
            reg_idx = sorted(self.frame_index(f) for f in registered)

            def gap(f):
                i = self.frame_index(f)
                if not reg_idx or i < 0:
                    return (1 << 30, i)
                j = bisect.bisect_left(reg_idx, i)
                near = [abs(reg_idx[k] - i) for k in (j - 1, j) if 0 <= k < len(reg_idx)]
                return (min(near), i)
            frontier.sort(key=gap)
        self._pos += 1
        if refine and (not frontier or self._pos % 2):
            return refine[(self._pos // 2) % len(refine)]
        return frontier[0] if frontier else None

    def has_work(self, adds: List, inlier_ema: Optional[float], buffered: Iterable[str],
                 anchored: Set[str], known: Iterable[str], registered: Set[str]) -> bool:
        if adds or self.dirty:
            return True
        if (inlier_ema or 0.0) < self.idle_inliers:
            return True
        if any(f not in anchored for f in buffered):
            return True
        return any(f not in registered and self.retry_ok(f) for f in known)
