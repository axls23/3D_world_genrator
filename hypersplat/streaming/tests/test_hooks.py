"""Unit 2: live runtime hooks (session + SummaryWriter), sitecustomize autoload, file tail."""
import os
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import numpy as np
import pytest

from hypersplat.streaming import contract as c
from hypersplat.streaming import hooks, sources
from hypersplat.streaming.producer import FakeSender, Sender

REPO = Path(__file__).resolve().parents[3]
AUTOHOOK = REPO / "hypersplat" / "streaming" / "autohook"
RECORDING = Path("/mnt/Shared/3D_WORLD_GENRATION/3D_world_genrator/experiments/cuda_baseline/"
                 "record1/results/stream_3dgs/stream_events.arrows")


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("HYPERSPLAT_JOB_ID", "u2-test")
    sender = FakeSender()
    hooks.uninstall()
    hooks.install(sender_factory=lambda: sender)
    yield sender
    hooks.uninstall()
    sender.close(1.0)


def _stub_session():
    """Enough of a StreamSession for the original _on_event to run without ACE."""
    calls = []
    s = types.SimpleNamespace(_recorder=None, device_deltas_on=False, failed=None,
                              _stop=types.SimpleNamespace(is_set=lambda: True), calls=calls,
                              _cond=threading.Condition(),
                              gpu_lease=types.SimpleNamespace(release=lambda holder: None))
    s._on_pose = lambda row, from_map=False: calls.append("pose")
    s._on_map = lambda rows: calls.append("map")
    s._on_reloc = lambda rows: calls.append("reloc")
    s._on_error = lambda meta: calls.append("error")
    return s


def _row(kind="pose", frame="f0.png", w2c="eye", focal=500.0, conf=100.0):
    if isinstance(w2c, str):
        w2c = np.eye(4)
        w2c[:3, 3] = (0.1, 0.2, 0.3)
    return dict(kind=kind, version=1, frame=frame, w2c=w2c, conf=conf, focal=focal,
                points=np.zeros((0, 3)), pix=np.zeros((0, 2)), meta={})


# ------------------------------------------------------------------------------ session hook
def test_session_hook_wraps_and_sends(fake):
    from hypersplat.stream.session import StreamSession

    assert getattr(StreamSession._on_event, hooks._WRAPPED, False)
    s = _stub_session()
    StreamSession._on_event(s, "map", [_row("map", "a.png"), _row("map", "b.png", focal=520.0)])
    StreamSession._on_event(s, "pose", [_row("pose", "c.png", focal=None)])   # map focal reused
    StreamSession._on_event(s, "reloc", [_row("reloc", "d.png", w2c=None)])   # unregistered
    StreamSession._on_event(s, "pose", [_row("pose", "e.png", w2c=np.full((4, 4), np.nan))])
    StreamSession._on_event(s, "error", [{"kind": "error", "meta": {"msg": "x"}}])
    StreamSession._on_event(s, "exit", [{"kind": "exit", "meta": {"returncode": 0}}])
    fake.flush()
    assert s.calls == ["map", "pose", "reloc", "pose", "error"]  # original ran every time
    evs = fake.events()
    assert [(e.frame, e.source, e.seq) for e in evs] == [("a.png", "map", 0), ("b.png", "map", 1),
                                                        ("c.png", "pose", 2)]
    assert evs[2].focal == 520.0 and evs[0].tx == pytest.approx(0.1)
    assert hooks.unregistered("u2-test") == 1 and fake.dropped("u2-test") == 1  # NaN w2c


def test_session_hook_never_raises_into_caller(fake, monkeypatch):
    from hypersplat.stream.session import StreamSession

    def boom():
        raise RuntimeError("sender exploded")

    monkeypatch.setattr(hooks, "_sender_factory", boom)
    StreamSession._on_event(_stub_session(), "pose", [_row()])  # no exception
    # ...but the session's own errors still propagate unchanged
    bad = _stub_session()
    bad._on_pose = lambda row, from_map=False: 1 / 0
    with pytest.raises(ZeroDivisionError):
        StreamSession._on_event(bad, "pose", [_row()])


@pytest.mark.skipif(not RECORDING.exists(), reason="record1 recording not on this machine")
def test_recording_rows_become_poses(fake):
    from hypersplat.stream.record import load_recording
    from hypersplat.stream.session import StreamSession

    rec = load_recording(str(RECORDING))
    want = {}
    s = _stub_session()
    for _, kind, rows in rec:
        StreamSession._on_event(s, kind, rows)
        for r in rows:
            if kind in hooks.POSE_KINDS and r.get("w2c") is not None:
                want[kind] = want.get(kind, 0) + 1
    fake.flush(10.0)
    got = {}
    for e in fake.events():
        got[e.source] = got.get(e.source, 0) + 1
    assert got == want and sum(got.values()) > 0
    assert fake.dropped("u2-test") == 0


# ----------------------------------------------------------------------------- training hook
def test_summary_writer_hook_groups_steps(fake, tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    w = SummaryWriter(log_dir=str(tmp_path / "tb"))
    for step in range(0, 500, 10):
        w.add_scalar("train/loss", 0.5 - step * 1e-4, step)
        w.add_scalar("train/l1loss", 0.3, step)
        w.add_scalar("train/num_GS", 1000 + step, step)
        w.add_scalar("train/mem", 2.5, step)
        w.add_scalar("train/depthloss", 9.0, step)  # not a snapshot field
        if step % 100 == 0:
            w.flush()                                # trainer flushes after its scalars
            w.add_scalar("train/mem", 3.0, step)    # late tag for an emitted step: ignored
        time.sleep(0.001)
    w.add_scalar("train/mem", 2.0, 600)             # no train/loss: dropped at close
    w.close()
    fake.flush()
    snaps = fake.events()
    assert [s.step for s in snaps] == list(range(0, 500, 10))
    assert [s.seq for s in snaps] == list(range(50))
    assert snaps[0].steps_per_s is None and all(s.steps_per_s > 0 for s in snaps[1:])
    assert snaps[3].num_gaussians == 1030 and snaps[0].gpu_mem_gb == 2.5 and snaps[0].l1 == 0.3
    assert fake.dropped("u2-test") == 1
    assert (tmp_path / "tb").exists() and any((tmp_path / "tb").iterdir())  # TB still written


def test_job_id_from_result_dir(monkeypatch):
    monkeypatch.delenv("HYPERSPLAT_JOB_ID", raising=False)
    monkeypatch.setattr(hooks, "_job", None)
    monkeypatch.setattr(sys, "argv", ["simple_trainer.py", "--result_dir", "/x/out/stream_3dgs"])
    assert hooks.job_id() == "stream_3dgs-" + hooks._START
    monkeypatch.setattr(hooks, "_job", None)
    monkeypatch.setattr(sys, "argv", ["x"])
    assert hooks.job_id("/y/run7/tb") == "run7-" + hooks._START
    monkeypatch.setattr(hooks, "_job", None)


# ---------------------------------------------------------------- non-blocking, dead broker
def test_dead_broker_hooks_stay_under_1ms(monkeypatch, tmp_path):
    from hypersplat.stream.session import StreamSession
    from torch.utils.tensorboard import SummaryWriter

    monkeypatch.setenv("HYPERSPLAT_JOB_ID", "u2-dead")
    # small queue: the sender thread sits in kafka-python's bootstrap retries meanwhile, so the
    # drops counted right away are the queue-full ones
    sender = Sender("localhost:1", queue_max=10)
    hooks.uninstall()
    # time the hook alone: wrap a no-op add_scalar instead of TensorBoard's own (whose event-file
    # writes jitter by ~1 ms and are not the hook's cost)
    monkeypatch.setattr(SummaryWriter, "add_scalar", lambda self, *a, **k: None)
    hooks.install(sender_factory=lambda: sender)
    try:
        s, w = _stub_session(), SummaryWriter(log_dir=str(tmp_path / "tb"))
        worst = 0.0
        for i in range(200):
            t0 = time.perf_counter()
            StreamSession._on_event(s, "pose", [_row(frame="f%d.png" % i)])
            worst = max(worst, time.perf_counter() - t0)
        for step in range(50):  # each step's first tag emits the previous step's snapshot
            for tag, v in (("train/loss", 0.1), ("train/num_GS", 1000), ("train/mem", 2.0)):
                t0 = time.perf_counter()
                w.add_scalar(tag, v, step)
                worst = max(worst, time.perf_counter() - t0)
        assert worst < 1e-3, worst
        w.close()
        assert sender.dropped("u2-dead") >= 250 - 11 and sender.sent == 0  # 200 poses + 50 snaps
    finally:
        hooks.uninstall()
        sender.close(0.5)


# ------------------------------------------------------------------------ autoload, argv
def _run_py(code, **env):
    full = dict(os.environ, PYTHONPATH="%s:%s" % (AUTOHOOK, REPO), **env)
    return subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True,
                          timeout=120)


CHECK = ("import hypersplat.stream.session as s; "
         "print(getattr(s.StreamSession._on_event, '_hypersplat_streaming_hook', False))")


def test_sitecustomize_patches_after_import():
    on = _run_py(CHECK, HYPERSPLAT_STREAMING="1")
    assert on.returncode == 0 and on.stdout.strip() == "True", on.stderr
    off = _run_py(CHECK, HYPERSPLAT_STREAMING="0")
    assert off.returncode == 0 and off.stdout.strip() == "False", off.stderr


def test_sitecustomize_chains_and_never_breaks_startup(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("import os\nos.environ['CHAINED'] = '1'\n")
    env = dict(os.environ, HYPERSPLAT_STREAMING="1",
               PYTHONPATH="%s:%s:%s" % (AUTOHOOK, other, REPO))
    code = "import os, sys; print(os.environ.get('CHAINED'), sys.meta_path[0].__class__.__name__)"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.stdout.split() == ["1", "_PostImportFinder"], r.stderr
    (other / "sitecustomize.py").write_text("raise RuntimeError('broken neighbour')\n")
    r = subprocess.run([sys.executable, "-c", "print('alive')"], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0 and r.stdout.strip() == "alive" and "skipped" in r.stderr


def test_extend_pipeline_argv(monkeypatch):
    monkeypatch.setenv(hooks.TRAINER_ARGS_ENV, "tb_every=10 foo=1")
    argv = ["scripts/pipeline/automated_intelligent_pipeline.py", "usb", "--stream",
            "--trainer-arg", "foo=2"]
    hooks.extend_pipeline_argv(argv)
    assert argv[-2:] == ["--trainer-arg", "tb_every=10"] and argv.count("--trainer-arg") == 2
    other = ["examples/simple_trainer.py", "default"]
    hooks.extend_pipeline_argv(other)
    assert other == ["examples/simple_trainer.py", "default"]


def test_live_sh_syntax():
    assert subprocess.run(["bash", "-n", str(REPO / "hypersplat/streaming/live.sh")]).returncode == 0


def test_py38_syntax():
    import ast

    for f in ("hooks.py", "sources.py", "autohook/sitecustomize.py"):
        ast.parse((REPO / "hypersplat/streaming" / f).read_text(), feature_version=(3, 8))


# ------------------------------------------------------------------------------ file tail
LINE = "frames/%s 1 0 0 0 0.1 0.2 0.3 500 120\n"


def test_tail_append_partial_truncate_rewrite(tmp_path):
    p = tmp_path / "poses_final.txt"
    t = sources.PoseFileTail(str(p))
    assert t.poll() == []                                    # not there yet
    p.write_text(LINE % "a.png" + (LINE % "b.png")[:12])     # second line half written
    assert [l.split()[0] for l in t.poll()] == ["frames/a.png"]
    with open(p, "a") as f:
        f.write((LINE % "b.png")[12:])
    assert [l.split()[0] for l in t.poll()] == ["frames/b.png"]
    p.write_text(LINE % "a.png")                             # truncated, same first line
    assert t.poll() == [] and t.rewrites == 1                # already sent: no duplicate
    p.write_text(LINE % "a.png" + (LINE % "c.png").replace("0.3", "0.4"))
    assert [l.split()[0] for l in t.poll()] == ["frames/c.png"]
    tmp = tmp_path / "new.txt"                               # atomic replace with a moved pose
    tmp.write_text((LINE % "a.png").replace("0.1", "0.9"))
    os.replace(tmp, p)
    assert [l.split()[0] for l in t.poll()] == ["frames/a.png"]


def test_tail_sends_file_poses(tmp_path):
    p = tmp_path / "poses_final.txt"
    p.write_text(LINE % "a.png" + "garbage line\n" + LINE % "b.png" + (LINE % "c.png").strip())
    s = FakeSender()
    assert sources.tail(str(p), "u2-file", s, once=True) == 3  # --once takes the last line too
    s.flush()
    evs = s.events()
    assert [e.frame for e in evs] == ["a.png", "b.png", "c.png"]
    assert {e.source for e in evs} == {"file"} and s.dropped("u2-file") == 1
    s.close(1.0)


# ------------------------------------------------------------------------- real Kafka (opt)
def _kafka_up():
    host, port = c.KAFKA_BOOTSTRAP.split(",")[0].rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _kafka_up(), reason="Kafka unreachable")
def test_hooks_reach_kafka(monkeypatch):
    kafka = pytest.importorskip("kafka")
    from hypersplat.stream.session import StreamSession

    job = "u2-kafka-" + uuid.uuid4().hex[:8]
    monkeypatch.setenv("HYPERSPLAT_JOB_ID", job)
    sender = Sender()
    hooks.uninstall()
    hooks.install(sender_factory=lambda: sender)
    try:
        StreamSession._on_event(_stub_session(), "map", [_row("map", "m%d.png" % i) for i in range(5)])
        sender.flush(10.0)
    finally:
        hooks.uninstall()
        sender.close(2.0)
    consumer = kafka.KafkaConsumer(c.TOPIC_POSES.name, bootstrap_servers=c.KAFKA_BOOTSTRAP,
                                   auto_offset_reset="earliest", consumer_timeout_ms=5000,
                                   group_id=None)
    # only this job's messages (by key): the shared topic may hold other jobs' or dead-lettered garbage
    got = [c.decode(m.value) for m in consumer if m.key == job.encode()]
    consumer.close()
    assert [(e.frame, e.source) for e in got] == [("m%d.png" % i, "map") for i in range(5)]


def test_two_threads_publishing_poses_never_invert_sequence_order(fake):
    """StreamSession._on_event is called by the sidecar reader thread and by the ring thread. The seq is
    handed out and the event queued by the hook, so the pair must be atomic: an event queued after a
    higher seq would be skipped by the processor as late and silently lost from the history."""
    import sys

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)               # make the interleaving that breaks a non-atomic pair likely
    try:
        def worker(tag):
            for i in range(300):
                hooks.publish_rows("pose", [_row(frame="%s%d.png" % (tag, i))])

        threads = [threading.Thread(target=worker, args=(t,)) for t in "abcd"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)
    fake.flush(5.0)
    seqs = [e.seq for e in fake.events()]
    assert len(seqs) == 1200
    assert seqs == sorted(seqs) and seqs == list(range(1200))
