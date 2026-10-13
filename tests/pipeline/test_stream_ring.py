"""Shared-memory rings between the ACE sidecar and the trainer (hypersplat/stream/ring.py)."""
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from hypersplat.stream import native
from hypersplat.stream.ring import (DELTA_DTYPE, FEEDBACK_DTYPE, SEED_DTYPE, ShmRing,
                                    frame_kid)

pytestmark = pytest.mark.skipif(native.lib() is None, reason="no C compiler for streamops")

ROOT = Path(__file__).resolve().parents[2]
ACE_PY = ROOT.parent / ".conda_envs" / "ace0" / "bin" / "python"


def _deltas(seqs):
    r = np.zeros(len(seqs), DELTA_DTYPE)
    r["seq"] = seqs
    r["kid"] = np.asarray(seqs) % 7
    r["w2c"] = np.arange(16, dtype=np.float64)[None] + np.asarray(seqs, np.float64)[:, None]
    return r


def test_fifo_order_and_wraparound(tmp_path):
    ring = ShmRing(tmp_path / "r", DELTA_DTYPE, capacity=8)
    seq = 0
    got = []
    for burst in [3, 5, 2, 7, 1, 8, 4]:     # crosses the end of the buffer many times
        assert ring.push(_deltas(range(seq, seq + burst))) == burst
        seq += burst
        got.extend(ring.pop(burst)["seq"].tolist())
    assert got == list(range(seq))
    assert ring.stats() == (0, seq, 0)
    ring.unlink()


def test_full_ring_refuses_and_counts(tmp_path):
    ring = ShmRing(tmp_path / "r", DELTA_DTYPE, capacity=4)
    assert ring.push(_deltas(range(6))) == 4      # backpressure: only what fits
    assert ring.stats() == (4, 4, 2)
    assert ring.free == 0
    out = ring.pop(10)
    assert out["seq"].tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(out["w2c"], _deltas(range(4))["w2c"])
    assert ring.push(_deltas([9])) == 1
    ring.unlink()


def test_open_existing_and_reject_mismatch(tmp_path):
    a = ShmRing(tmp_path / "r", FEEDBACK_DTYPE, capacity=16)
    b = ShmRing(tmp_path / "r", FEEDBACK_DTYPE)            # second mapping of the same file
    assert b.capacity == 16
    rec = np.zeros(2, FEEDBACK_DTYPE)
    rec["kid"] = [5, 6]
    rec["loss_ema"] = [0.1, 0.2]
    a.push(rec)
    out = b.pop()
    assert out["kid"].tolist() == [5, 6]
    np.testing.assert_allclose(out["loss_ema"], [0.1, 0.2])
    with pytest.raises(ValueError):
        ShmRing(tmp_path / "r", DELTA_DTYPE)               # wrong record size
    with pytest.raises(ValueError):
        ShmRing(tmp_path / "bad", DELTA_DTYPE, capacity=6)  # not a power of two
    a.unlink()
    b.close()


def test_seed_record_round_trip(tmp_path):
    ring = ShmRing(tmp_path / "s", SEED_DTYPE, capacity=4)
    rec = np.zeros(1, SEED_DTYPE)
    rec["kid"], rec["n"] = 12, 3
    rec["xyz"][0, :3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ring.push(rec)
    out = ring.pop()[0]
    assert out["kid"] == 12 and out["n"] == 3
    np.testing.assert_array_equal(out["xyz"][:3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ring.unlink()


def test_frame_kid():
    assert frame_kid("/dev/shm/x/frames/kf_000123.jpg") == 123
    assert frame_kid("kf_7.png") == 7
    assert frame_kid("/a/b/frame_12.jpg") == -1
    assert frame_kid("kf_abc.jpg") == -1


@pytest.mark.skipif(not ACE_PY.exists(), reason="ace0 env not installed")
def test_cross_interpreter_producer(tmp_path):
    """The ACE env (py3.8) produces, this interpreter consumes, concurrently."""
    n, cap = 20000, 64
    ring = ShmRing(tmp_path / "x", DELTA_DTYPE, capacity=cap)
    producer = textwrap.dedent(f"""
        import sys, time
        import numpy as np
        sys.path.insert(0, {str(ROOT)!r})
        from hypersplat.stream.ring import DELTA_DTYPE, ShmRing
        ring = ShmRing({str(tmp_path / 'x')!r}, DELTA_DTYPE)
        seq = 0
        while seq < {n}:
            k = min(17, {n} - seq)
            r = np.zeros(k, DELTA_DTYPE)
            r["seq"] = np.arange(seq, seq + k)
            r["w2c"][:, 0] = r["seq"]
            seq += ring.push(r)          # backpressure: retry what did not fit
        print("done", ring.stats()[2])
    """)
    proc = subprocess.Popen([str(ACE_PY), "-c", producer], stdout=subprocess.PIPE, text=True)
    got = []
    while len(got) < n:
        out = ring.pop(cap)
        assert np.all(out["w2c"][:, 0] == out["seq"])   # never a torn record
        got.extend(out["seq"].tolist())
        if proc.poll() is not None and proc.returncode != 0:
            break
    stdout, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, stdout
    assert got == list(range(n))
    assert int(stdout.split()[1]) > 0            # the small ring did push back
    ring.unlink()
