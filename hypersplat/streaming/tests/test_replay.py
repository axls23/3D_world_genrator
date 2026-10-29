"""recording.py export and replay.py (rows 3, 7, 12, 21): CPU only, FakeSender, tmp_path data."""

import io
import math
import os
import signal
import threading
import time
import uuid

import pytest

from hypersplat.streaming import contract as c
from hypersplat.streaming import recording, replay
from hypersplat.streaming.producer import FakeSender


def _pose(seq, x=0.0, conf=100.0):
    return c.CameraPose("rec", seq, 1.0, "kf_%03d.jpg" % seq, 1.0, 0.0, 0.0, 0.0,
                        x, 0.0, 0.0, 500.0, conf)


def _snap(seq, step):
    return c.TrainingSnapshot("rec", seq, 1.0, step, 0.1, 1000, 2.0)


def _events(n_poses=30, every=10):
    """n_poses poses on a straight line with a snapshot after every ``every`` poses."""
    out = []
    for i in range(n_poses):
        out.append(_pose(len(out), x=float(i)))
        if (i + 1) % every == 0:
            out.append(_snap(len(out), i))
    return out


def _sent(sender):
    sender.flush()
    return sender.events()


# --------------------------------------------------------------------------- recording export

def _w2c(x):
    return [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


@pytest.fixture
def arrow_recording(tmp_path):
    pytest.importorskip("pyarrow")
    from hypersplat.stream.record import EventRecorder

    path = tmp_path / "stream_events.arrows"
    rec = EventRecorder(str(path))
    rec.add("map", [dict(frame="/x/kf_000.jpg", w2c=_w2c(0.0), conf=900.0, focal=600.0, version=1),
                    dict(frame="/x/kf_001.jpg", w2c=_w2c(0.1), conf=800.0, focal=600.0, version=1)], -1)
    rec.add("pose", [dict(frame="/x/kf_002.jpg", w2c=None, conf=0.0, version=1)], 5)   # unregistered
    rec.add("pose", [dict(frame="/x/kf_003.jpg", w2c=_w2c(0.3), conf=700.0, version=1)], 10)  # no focal
    rec.add("reloc", [dict(frame="/x/kf_000.jpg", w2c=_w2c(0.0), conf=950.0, focal=600.0, version=1)], 25)
    rec.close()
    return path


@pytest.fixture
def tb_dir(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    d = tmp_path / "tb"
    w = EventFileWriter(str(d))
    for step, wall in ((0, 1000.0), (10, 1010.0), (20, 1020.0)):
        vals = {"train/loss": 0.5 - step / 100, "train/num_GS": 1000 + step, "train/mem": 1.5,
                "train/l1loss": 0.1, "stream/keyframes": 3}
        summ = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=t, simple_value=v)
                                          for t, v in vals.items()])
        w.add_event(event_pb2.Event(wall_time=wall, step=step, summary=summ))
    w.close()
    return d


def test_export_events_and_tb(arrow_recording, tb_dir, tmp_path):
    out = tmp_path / "recordings" / "run.jsonl"
    assert recording.main(["export", "--events", str(arrow_recording), "--tb", str(tb_dir),
                           "--out", str(out)]) == 0
    events = replay.load(str(out))
    kinds = [(type(e).__name__, getattr(e, "step", None), getattr(e, "frame", None)) for e in events]
    assert kinds == [
        ("CameraPose", None, "kf_000.jpg"), ("CameraPose", None, "kf_001.jpg"),  # map, step -1
        ("TrainingSnapshot", 0, None),
        ("CameraPose", None, "kf_003.jpg"),                                       # step 10 pose ...
        ("TrainingSnapshot", 10, None),                                           # ... before step 10
        ("TrainingSnapshot", 20, None),
        ("CameraPose", None, "kf_000.jpg"),                                       # reloc, step 25
    ]
    assert [e.seq for e in events] == list(range(7))
    assert {e.job_id for e in events} == {"run"}
    poses = [e for e in events if isinstance(e, c.CameraPose)]
    assert [p.source for p in poses] == ["map", "map", "pose", "reloc"]
    assert poses[2].focal == 600.0 and poses[2].tx == pytest.approx(0.3)  # focal from the map
    assert poses[2].ts == pytest.approx(1010.0)   # step 10 -> TB wall time
    snap = events[2]
    assert snap.loss == pytest.approx(0.5) and snap.num_gaussians == 1000 and snap.l1 == pytest.approx(0.1)
    # Each line is what contract.encode() would put on Kafka
    with open(out, "rb") as f:
        first = f.readline().rstrip(b"\n")
    assert first == c.encode(events[0])[2]


def test_export_counts_unregistered_and_tb_optional(arrow_recording):
    events, stats = recording.build("j", events=str(arrow_recording))
    assert stats == {"poses": 4, "training": 0, "unregistered": 1, "invalid": 0}
    assert all(isinstance(e, c.CameraPose) for e in events)


def test_export_pose_file(tmp_path):
    pf = tmp_path / "poses_final.txt"
    pf.write_text("frames/a.jpg 1 0 0 0 0 0 0 600 10\n"
                  "frames/b.jpg 1 0 0 0 0.1 0 0 600 12\n"
                  "frames/c.jpg 1 0 0 0 nan 0 0 600 12\n")
    out = tmp_path / "batch.jsonl"
    assert recording.main(["export", "--pose-file", str(pf), "--out", str(out)]) == 0
    events = replay.load(str(out))
    assert [e.frame for e in events] == ["a.jpg", "b.jpg"]
    assert all(e.source == "file" for e in events)


# --------------------------------------------------------------------------- replay

def test_replay_rewrites_job_seq_ts():
    s = FakeSender()
    s.next_seq("other")  # another job's counter must not leak in
    events = _events()
    t0 = time.time()
    assert replay.run(events, s, "replay-x", rate=1000, status=replay.Status("x", 1, 1, io.StringIO())) == 33
    got = _sent(s)
    assert [e.seq for e in got] == list(range(33))
    assert {e.job_id for e in got} == {"replay-x"}
    assert all(t0 <= e.ts <= time.time() for e in got)
    assert [type(e) for e in got] == [type(e) for e in events]
    assert {e.source for e in got if isinstance(e, c.CameraPose)} == {"replay"}
    assert all(e.injected is None for e in got)
    s.close()


def test_replay_keep_source_and_loop():
    s = FakeSender()
    events = [replace_source(e) for e in _events(4, every=100)]
    n = replay.run(events, s, "j", rate=1000, loop=True, keep_source=True, max_events=10,
                   status=replay.Status("j", 4, 1, io.StringIO()))
    got = _sent(s)
    assert n == 10 and [e.seq for e in got] == list(range(10))
    assert {e.source for e in got} == {"reloc"}
    s.close()


def replace_source(e):
    from dataclasses import replace
    return replace(e, source="reloc")


def test_replay_pace():
    s = FakeSender()
    t0 = time.monotonic()
    replay.run(_events(10, every=100), s, "j", rate=50, status=replay.Status("j", 1, 1, io.StringIO()))
    dt = time.monotonic() - t0
    assert 9 / 50 * 0.8 < dt < 9 / 50 + 0.3   # first event at t=0, then every 20 ms
    s.close()


def test_pause_blocks_sending_and_resume_does_not_burst():
    s = FakeSender()
    ctrl = replay.Control()
    out = io.StringIO()
    status = replay.Status("j", 50, 20, out)
    th = threading.Thread(target=replay.run, args=(_events(50, every=100), s, "j"),
                          kwargs=dict(rate=20, ctrl=ctrl, status=status))
    th.start()
    time.sleep(0.3)
    ctrl.toggle()
    time.sleep(0.15)
    n_paused = status.sent
    time.sleep(0.5)
    assert status.sent == n_paused and "PAUSED" in out.getvalue()
    ctrl.toggle()
    time.sleep(0.25)
    assert n_paused < status.sent <= n_paused + 7   # ~5 at 20 Hz, not a catch-up burst
    ctrl.stop()
    th.join(2)
    assert not th.is_alive()
    s.close()


def test_sigusr1_toggles_pause():
    ctrl = replay.Control()
    old = signal.signal(signal.SIGUSR1, ctrl.toggle)
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.05)
        assert ctrl.paused
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.05)
        assert not ctrl.paused
    finally:
        signal.signal(signal.SIGUSR1, old)


# --------------------------------------------------------------------------- injection

def test_inject_shuffle_labels_exactly_k_jumps():
    events = _events(40)
    out = replay.inject(events, "shuffle", at=10, length=5)
    poses = [e for e in out if isinstance(e, c.CameraPose)]
    labeled = [i for i, p in enumerate(poses) if p.injected == "shuffle"]
    assert labeled == [10, 11, 12, 13, 14]
    # pose k sits at x=k: even offsets jump K ahead, odd offsets K behind
    assert [poses[i].tx for i in labeled] == [15.0, 6.0, 17.0, 8.0, 19.0]
    assert [p.frame for p in poses] == ["kf_%03d.jpg" % e.seq for e in events if isinstance(e, c.CameraPose)]
    unchanged = [i for i in range(len(out)) if out[i] is events[i]]
    assert len(unchanged) == len(events) - 5   # training untouched, other poses untouched
    assert events == _events(40)               # input not mutated


def test_inject_blur():
    events = _events(40)
    out = replay.inject(events, "blur", at=20, length=4, seed=1)
    poses = [e for e in out if isinstance(e, c.CameraPose)]
    labeled = [i for i, p in enumerate(poses) if p.injected == "blur"]
    assert labeled == [20, 21, 22, 23]
    for i in labeled:
        p = poses[i]
        assert p.confidence == pytest.approx(10.0)
        ctr = c.camera_center(p)  # jitter moves the centre by a fraction of the 1.0 step
        assert 0 < math.dist(ctr, (-float(i), 0.0, 0.0)) < 1.0
        assert math.isclose(p.qw ** 2 + p.qx ** 2 + p.qy ** 2 + p.qz ** 2, 1.0)
        assert c.turn_deg(p, events[0]) < 6.0
    assert out == replay.inject(events, "blur", at=20, length=4, seed=1)  # deterministic


@pytest.mark.parametrize("kind,at,length", [("shuffle", 38, 5), ("blur", -1, 2), ("blur", 0, 0),
                                            ("shuffle", 0, 3), ("shuffle", 10, 1),
                                            ("fog", 0, 1)])
def test_inject_rejects_bad_ranges(kind, at, length):
    with pytest.raises(ValueError):
        replay.inject(_events(40), kind, at, length)


def test_replay_counts_injected():
    s = FakeSender()
    status = replay.Status("j", 1, 1, io.StringIO())
    replay.run(replay.inject(_events(20), "blur", 2, 3), s, "j", rate=1000, status=status)
    got = _sent(s)
    assert status.counts["injected"] == 3 == sum(e.injected == "blur" for e in got)
    s.close()


def test_cli_requires_at_and_job_id_for_reset(tmp_path):
    with pytest.raises(SystemExit):
        replay.main(["--recording", "x.jsonl", "--inject", "shuffle"])
    with pytest.raises(SystemExit):
        replay.main(["--recording", "x.jsonl", "--reset"])


# --------------------------------------------------------------------------- database reset

@pytest.fixture
def db():
    psycopg = pytest.importorskip("psycopg")
    try:
        conn = psycopg.connect(c.DB_DSN, connect_timeout=3, autocommit=True)
    except Exception as e:
        pytest.skip("PostgreSQL unreachable at %s: %s" % (c.DB_DSN, e))
    yield conn
    conn.close()


def test_reset_deletes_job_rows(db):
    db.execute(c.DDL)
    job, other = "test-reset-" + uuid.uuid4().hex[:8], "test-keep-" + uuid.uuid4().hex[:8]
    for j in (job, other):
        db.execute("INSERT INTO jobs (job_id) VALUES (%s)", (j,))
        db.execute("INSERT INTO pose_history (job_id, seq, ts, frame) VALUES (%s, 0, now(), 'a.jpg')", (j,))
        db.execute("INSERT INTO training_history (job_id, seq, ts, step, loss) VALUES (%s, 0, now(), 0, 0.1)", (j,))
    try:
        assert replay.reset_job(job) == {"pose_history": 1, "training_history": 1, "jobs": 1}
        assert db.execute("SELECT count(*) FROM jobs WHERE job_id = %s", (job,)).fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM pose_history WHERE job_id = %s", (other,)).fetchone()[0] == 1
        assert replay.reset_job(job) == {"pose_history": 0, "training_history": 0, "jobs": 0}
    finally:
        replay.reset_job(job)
        replay.reset_job(other)
