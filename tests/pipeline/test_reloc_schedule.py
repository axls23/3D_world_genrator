"""ACE sidecar relocalization scheduling (scripts/acezero/reloc_schedule.py); stdlib only, CPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "acezero"))
from reloc_schedule import RelocScheduler  # noqa: E402

NAMES = [f"/x/kf_{i:06d}.jpg" for i in range(20)]
REG = set(NAMES[:5])


def _sched():
    s = RelocScheduler(backoff_ticks=2, backoff_cap=4)
    for n in NAMES[5:]:
        s.note_failure(n)
    s.dirty = False
    return s


def test_failures_wait_for_growth_then_frontier_first():
    s = _sched()
    assert s.pick(NAMES, REG, anchored=set(REG)) is None  # nothing eligible yet
    s.note_joined(NAMES[4])
    assert s.pick(NAMES, REG, anchored=set(REG)) == NAMES[5]  # nearest keyframe index first


def test_backoff_expires_by_ticks_while_idle():
    s = _sched()
    anchored = set(REG)
    assert not s.has_work([], 0.9, list(REG), anchored, NAMES, REG)  # idle: nothing due
    for _ in range(2 * 2):  # backoff_ticks * 2**1 idle ticks, no map growth
        s.tick(idle=True)
    assert s.retry_ok(NAMES[5])
    assert s.has_work([], 0.9, list(REG), anchored, NAMES, REG)  # wakes up for the retry


def test_backoff_doubles_and_is_capped():
    s = RelocScheduler(backoff_ticks=1, backoff_cap=4)
    for _ in range(5):
        s.note_failure("kf_000001.jpg")
    for _ in range(3):
        s.tick(idle=True)
    assert not s.retry_ok("kf_000001.jpg")
    s.tick(idle=True)
    assert s.retry_ok("kf_000001.jpg")  # 1 * min(2**5, 4) = 4 ticks


def test_unfitted_or_unanchored_or_dirty_means_work():
    s = _sched()
    assert s.has_work([], 0.5, list(REG), set(REG), NAMES, REG)          # map not fitted
    assert s.has_work([], 0.9, list(REG), set(NAMES[:4]), NAMES, REG)    # a buffered frame unanchored
    s.dirty = True
    assert s.has_work([], 0.9, list(REG), set(REG), NAMES, REG)


def test_forget_clears_failure():
    s = _sched()
    s.forget(NAMES[6])
    assert s.retry_ok(NAMES[6]) and s.dirty
