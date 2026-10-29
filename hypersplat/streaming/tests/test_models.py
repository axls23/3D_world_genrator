"""Offline model training and the contamination experiment (rows 10-12), CPU only, on small
synthetic runs; models go to tmp_path via HYPERSPLAT_MODEL_DIR."""

import json
import math
import random
from dataclasses import replace

import pytest

pytest.importorskip("sklearn")
joblib = pytest.importorskip("joblib")

from hypersplat.streaming import contract as c  # noqa: E402
from hypersplat.streaming import experiment as ex  # noqa: E402
from hypersplat.streaming import train_models as tm  # noqa: E402


@pytest.fixture(autouse=True)
def model_dir(tmp_path, monkeypatch):
    d = tmp_path / "models"
    monkeypatch.setenv("HYPERSPLAT_MODEL_DIR", str(d))
    monkeypatch.setattr(c, "MODEL_DIR", str(d))  # contract read the env at import
    return d


def ace_line(i, rng, yaw_step=0.01):
    """A camera flying along +x while slowly yawing, in poses_final.txt format."""
    half = yaw_step * i / 2
    qw, qy = math.cos(half), math.sin(half)
    return "images/frame_%05d.jpg %r 0.0 %r 0.0 %r 0.0 0.0 500.0 %d\n" % (
        i, qw, qy, -0.1 * i + rng.gauss(0, 0.002), int(5000 + rng.gauss(0, 60)))


def pose_seq(n=60, seed=0, job="j"):
    rng = random.Random(seed)
    return [c.pose_from_ace_line(ace_line(i, rng), job, i, ts=float(i)) for i in range(n)]


def snap_seq(n=60, seed=0, job="t"):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        loss = 0.05 + 0.2 * math.exp(-i / 10) + abs(rng.gauss(0, 0.004))
        out.append(c.TrainingSnapshot(job, i, float(i), step=100 * i, loss=loss,
                                      num_gaussians=50000 + 500 * min(i, 20),
                                      gpu_mem_gb=1.0 + 0.01 * min(i, 20)))
    return out


def write_tb(path, snaps):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    w = EventFileWriter(str(path))
    for s in snaps:
        vals = {"train/loss": s.loss, "train/num_GS": s.num_gaussians, "train/mem": s.gpu_mem_gb,
                "train/other": 1.0}
        for tag, v in vals.items():
            w.add_event(Event(wall_time=1e9 + s.step, step=s.step, summary=Summary(
                value=[Summary.Value(tag=tag, simple_value=v)])))
    w.close()


def make_runs_dir(root, n_tb=3, n_ace=3):
    for k in range(n_tb):
        d = root / ("cap%d" % k) / "results" / "tb"
        d.mkdir(parents=True)
        write_tb(d, snap_seq(40, seed=k))
    for k in range(n_ace):
        d = root / ("cap%d" % k) / "ace"
        d.mkdir(parents=True, exist_ok=True)
        rng = random.Random(k)
        lines = [ace_line(i, rng) for i in range(50)]
        random.Random(k).shuffle(lines)  # ACE writes in processing order, not frame order
        (d / "poses_final.txt").write_text("".join(lines) + "garbage line\n")
    return root


# ---------------------------------------------------------------------------------------------

def test_windows_match_processor():
    seq = pose_seq(50)
    used, rows = tm.feature_rows(tm.POSE, seq)
    assert used[0] == c.MIN_WINDOW_TO_SCORE - 1 and used[-1] == 49
    assert len(rows) == 50 - c.MIN_WINDOW_TO_SCORE + 1
    assert all(len(r) == len(c.POSE_FEATURES) for r in rows)
    assert len(tm.window_at(seq, 49)) == min(c.WINDOW, 50)
    f = c.pose_features(tm.window_at(seq, 20))
    assert rows[used.index(20)] == [f[n] for n in c.POSE_FEATURES]
    assert tm.feature_rows(tm.TRAINING, snap_seq(3))[1] == []  # too short to score


def test_load_ace_sorts_frames_and_skips_bad_lines(tmp_path):
    root = make_runs_dir(tmp_path, n_tb=0, n_ace=1)
    poses = tm.load_ace_poses(str(root / "cap0" / "ace" / "poses_final.txt"), "job")
    assert [p.frame for p in poses] == ["frame_%05d.jpg" % i for i in range(50)]
    assert [p.seq for p in poses] == list(range(50))


def test_split_sweeps_on_frame_restart():
    seq = pose_seq(10)
    stream = seq[:6] + seq[2:10] + seq[0:3]
    sweeps = tm.split_sweeps(stream)
    assert [len(s) for s in sweeps] == [6, 8, 3]


def test_load_tb_reads_contract_tags(tmp_path):
    write_tb(tmp_path / "tb", snap_seq(12))
    (path,) = list((tmp_path / "tb").iterdir())
    snaps = tm.load_tb(str(path), "job")
    assert [s.step for s in snaps] == [100 * i for i in range(12)]
    assert snaps[3].num_gaussians == 51500 and snaps[0].gpu_mem_gb == pytest.approx(1.0)


def test_discover_and_split(tmp_path):
    root = make_runs_dir(tmp_path)
    (tmp_path / ".venv_x" / "tb").mkdir(parents=True)
    (tmp_path / ".venv_x" / "tb" / "events.out.tfevents.1").write_bytes(b"")
    found = tm.discover([str(root)])
    assert len(found["tb"]) == 3 and len(found["ace"]) == 3 and not found["recording"]
    runs = tm.collect_runs([str(root)])
    assert {r.id.split("/")[0] for r in runs} == {"cap0", "cap1", "cap2"}
    train, test = tm.split_runs(runs, ["cap1"])
    assert {r.id.split("/")[0] for r in test} == {"cap1"} and len(test) == 2
    # no match: each kind still gets a test run; everything matched: train on all of them
    assert sum(r.kind == tm.POSE for r in tm.split_runs(runs, ["nope"])[1]) == 1
    assert tm.split_runs(runs, ["cap"])[1] == []


def test_main_writes_models_and_meta(tmp_path, model_dir):
    root = make_runs_dir(tmp_path / "runs")
    assert tm.main(["--runs", str(root), "--holdout", "cap2", "--contamination", "0.1"]) == 0
    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta["contamination"] == 0.1 and meta["holdout"] == ["cap2"]
    for name, feats, kind in ((c.POSE_MODEL, c.POSE_FEATURES, tm.POSE),
                              (c.TRAINING_MODEL, c.TRAINING_FEATURES, tm.TRAINING)):
        m = meta["models"][name]
        assert m["features"] == list(feats)
        assert m["n_train"] > 0 and m["n_test"] > 0
        assert all(r.startswith(("cap0", "cap1")) for r in m["train_runs"])
        assert all(r.startswith("cap2") for r in m["test_runs"])
        assert set(m["feature_mean"]) == set(feats) and m["sklearn_version"]
        model = joblib.load(c.model_path(name))
        assert model.contamination == 0.1 and model.n_estimators == c.IFOREST_PARAMS["n_estimators"]
        seq = pose_seq(20) if kind == tm.POSE else snap_seq(20)
        _, rows = tm.feature_rows(kind, seq)
        assert model.decision_function(rows).shape == (len(rows),)


def test_injections():
    rng = random.Random(0)
    seq = pose_seq(40)
    out, pos = ex.inject_shuffle(seq, 10, rng)
    assert pos == [10, 20] and out[10].tx == seq[20].tx and out[20].qw == seq[10].qw
    assert out[10].frame == seq[10].frame and out[10].injected == "shuffle"
    assert sum(p.injected is not None for p in out) == 2 and seq[10].injected is None
    out, pos = ex.inject_blur(seq, 10, rng)
    assert pos == list(range(10, 10 + ex.BLUR_LEN))
    for i in pos:
        assert out[i].confidence == pytest.approx(seq[i].confidence * ex.BLUR_CONF_SCALE)
        assert math.isclose(sum(getattr(out[i], k) ** 2 for k in ("qw", "qx", "qy", "qz")), 1)
        assert out[i].injected == "blur"

    snaps = snap_seq(40)
    out, pos = ex.inject_spike(snaps, 10, rng)
    assert pos == [10] and out[10].loss == pytest.approx(snaps[10].loss * ex.SPIKE_SCALE)
    out, pos = ex.inject_rise(snaps, 10, rng)
    assert all(out[i + 1].loss > out[i].loss for i in pos[:-1])
    out, pos = ex.inject_explosion(snaps, 10, rng)
    assert out[pos[-1]].num_gaussians > 5 * snaps[9].num_gaussians
    assert ex.placements(40, 11)[0] == c.MIN_WINDOW_TO_SCORE
    assert max(ex.placements(40, 11)) + 11 <= 40 and ex.placements(8, 11) == []


def synthetic_runs():
    runs = [tm.Run("cap%d/ace" % k, tm.POSE, pose_seq(60, seed=k, job="p%d" % k)) for k in range(4)]
    runs += [tm.Run("cap%d/tb" % k, tm.TRAINING, snap_seq(60, seed=k, job="t%d" % k))
             for k in range(4)]
    return runs


def test_injected_windows_score_lower():
    runs = synthetic_runs()
    train, test = tm.split_runs(runs, ["cap3"])
    for kind in (tm.POSE, tm.TRAINING):
        model = tm.fit(tm.matrix(kind, train), c.CONTAMINATION)
        clean, bad, labels = ex.eval_set(kind, test)
        assert len(bad) == len(labels) > 0
        assert model.decision_function(bad).mean() < model.decision_function(clean).mean()


def test_experiment_table(tmp_path):
    results = ex.run_experiment(synthetic_runs(), ["cap3"])
    assert set(results) == {tm.POSE, tm.TRAINING}
    for res in results.values():
        assert [r["contamination"] for r in res["rows"]] == list(c.CONTAMINATION_GRID)
        for r in res["rows"]:
            assert r["caught"] + r["missed"] == r["injected"] > 0
            assert sum(t for _, t in r["by_label"].values()) == r["injected"]
            assert 0 <= r["recall"] <= 1
        # a looser threshold never catches less
        caught = [r["caught"] for r in res["rows"]]
        assert caught == sorted(caught)
    assert set(results[tm.POSE]["rows"][0]["by_label"]) == set(c.INJECTIONS)
    md = ex.to_markdown(results)
    assert md.count("| contamination | injected | caught | missed | false alarms |") == 2
    out = tmp_path / "res" / "contamination.md"
    assert ex.main(["--runs", str(make_runs_dir(tmp_path / "runs")), "--holdout", "cap2",
                    "--out", str(out)]) == 0
    assert "pose_iforest" in out.read_text()


def test_fallback_holdout_takes_whole_source():
    runs = [tm.Run("a/poses_final.txt", tm.POSE, pose_seq(10))]
    runs += [tm.Run("rec/stream_events.arrows#sweep%02d" % k, tm.POSE, pose_seq(10)) for k in range(3)]
    train, test = tm.split_runs(runs, ["nope"])
    assert [r.id for r in train] == ["a/poses_final.txt"] and len(test) == 3


def test_blur_jitter_is_about_one_step():
    seq = [replace(p, tz=p.tz + 50.0) for p in pose_seq(40)]  # map origin far from the cameras
    out, pos = ex.inject_blur(seq, 10, random.Random(1))
    step = ex._median_step(seq)
    for i in pos:
        moved = math.dist(c.camera_center(out[i]), c.camera_center(seq[i]))
        assert moved < 5 * step


def test_stale_model_removed(tmp_path, model_dir):
    root = make_runs_dir(tmp_path / "runs")
    assert tm.main(["--runs", str(root), "--holdout", "cap2"]) == 0
    assert (model_dir / (c.TRAINING_MODEL + ".joblib")).exists()
    for k in range(3):  # second run: poses only
        for f in (root / ("cap%d" % k) / "results" / "tb").iterdir():
            f.unlink()
    assert tm.main(["--runs", str(root), "--holdout", "cap2"]) == 0
    assert not (model_dir / (c.TRAINING_MODEL + ".joblib")).exists()
    assert list(json.loads((model_dir / "meta.json").read_text())["models"]) == [c.POSE_MODEL]
