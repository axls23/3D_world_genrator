import math

import pytest

from hypersplat.streaming import contract as c

ACE_LINE = "frames/000012.jpg 0.9998 0.01 0.015 0.005 0.12 -0.03 1.4 612.5 1830"


def test_ace_line_roundtrip():
    p = c.pose_from_ace_line(ACE_LINE, "job1", 0, ts=1.0)
    assert p.frame == "000012.jpg" and p.focal == 612.5
    assert math.isclose(p.qw ** 2 + p.qx ** 2 + p.qy ** 2 + p.qz ** 2, 1.0)
    topic, key, value = c.encode(p)
    assert topic == c.TOPIC_POSES.name and key == b"job1"
    assert len(value) < c.MAX_MESSAGE_BYTES
    assert c.decode(value) == p


@pytest.mark.parametrize("line", [
    "a.jpg 1 0 0 0 0 0 0 600",            # short
    "a.jpg 1 0 0 0 nan 0 0 600 10",        # NaN position
    "a.jpg 3 0 0 0 0 0 0 600 10",          # corrupt quaternion
    "a.jpg 1 0 0 0 0 0 0 -1 10",           # bad focal
])
def test_bad_pose_lines_rejected(line):
    with pytest.raises(c.InvalidEvent):
        c.pose_from_ace_line(line, "job1", 0)


def test_snapshot_from_tensorboard_tags():
    s = c.snapshot_from_scalars(
        100, {"train/loss": 0.05, "train/num_GS": 250000.0, "train/mem": 3.2,
              "train/grad_norm": float("inf"), "train/unrelated": 1.0}, "job1", 10)
    assert s.num_gaussians == 250000 and s.grad_norm is None
    assert c.decode(c.encode(s)[2]) == s
    with pytest.raises(c.InvalidEvent):
        c.snapshot_from_scalars(100, {"train/loss": float("nan"), "train/mem": 1.0}, "job1", 11)


def test_decode_garbage():
    for raw in (b"not json", b'{"kind":"nope"}', b'{"kind":"pose","job_id":"j"}'):
        with pytest.raises(c.InvalidEvent):
            c.decode(raw)


def test_features():
    poses = [c.validate(c.CameraPose, dict(job_id="j", seq=i, ts=i, frame="%d.jpg" % i,
                                           qw=1, qx=0, qy=0, qz=0, tx=-i, ty=0, tz=0,
                                           focal=600, confidence=100))
             for i in range(3)]
    f = c.pose_features(poses)
    assert set(c.POSE_FEATURES) <= set(f)
    assert {"step_dist", "confidence"} <= set(f)    # kept for the KPIs and the database
    assert math.isclose(f["step_dist"], 1.0) and f["turn_deg"] == 0.0
    snaps = [c.TrainingSnapshot("j", i, i, i * 10, 1.0 / (i + 1), 1000 + i, 2.0) for i in range(30)]
    t = c.training_features(snaps)
    assert set(t) == set(c.TRAINING_FEATURES) and t["loss_trend"] < 0


def test_layers_in_start_order_and_ports_free():
    assert [l.order for l in c.LAYERS] == sorted(l.order for l in c.LAYERS)
    assert not {l.port for l in c.LAYERS} & set(c.RESERVED_PORTS)


def _w2c(q, t):
    R = c._rotmat(q)
    return [R[0] + [t[0]], R[1] + [t[1]], R[2] + [t[2]], [0, 0, 0, 1]]


def test_event_row_adapter_matches_quaternion():
    import numpy as np

    q = (0.9, 0.1, -0.3, 0.2)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    row = {"kind": "reloc", "frame": "/tmp/ace/000007.png", "w2c": np.array(_w2c(q, (1, 2, 3))),
           "conf": np.float32(512.0), "focal": 0.0, "version": 3}
    p = c.pose_from_event_row(row, "job1", 4, focal=600.0)
    assert p.source == "reloc" and p.frame == "000007.png" and p.focal == 600.0
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip((p.qw, p.qx, p.qy, p.qz), q))
    assert (p.tx, p.ty, p.tz) == (1, 2, 3) and p.confidence == 512.0
    flat = dict(row, w2c=[v for r in _w2c(q, (1, 2, 3)) for v in r], focal=700.0)
    assert c.pose_from_event_row(flat, "job1", 5).focal == 700.0


def test_event_row_unregistered_and_bad():
    with pytest.raises(c.Unregistered):
        c.pose_from_event_row({"kind": "pose", "frame": "a.png", "w2c": None, "conf": 0.0}, "j", 0, focal=600)
    with pytest.raises(c.InvalidEvent):  # no focal anywhere
        c.pose_from_event_row({"kind": "pose", "frame": "a.png", "w2c": _w2c((1, 0, 0, 0), (0, 0, 0)),
                               "conf": 10.0, "focal": 0.0}, "j", 0)
    with pytest.raises(c.InvalidEvent):
        c.validate(c.CameraPose, dict(job_id="j", seq=0, ts=0, frame="a", qw=1, qx=0, qy=0, qz=0,
                                      tx=0, ty=0, tz=0, focal=600, confidence=1, source="bogus"))


def test_topic_prefix(monkeypatch):
    import importlib

    monkeypatch.setenv("HYPERSPLAT_TOPIC_PREFIX", "u9.")
    try:
        m = importlib.reload(c)
        assert all(t.name.startswith("u9.hypersplat.") for t in m.TOPICS)
    finally:
        monkeypatch.delenv("HYPERSPLAT_TOPIC_PREFIX")
        importlib.reload(c)


def _walk(scale, jump_at=None, n=12):
    """Camera moving along x by `scale` per frame (identity rotation: C = -t); one optional jump."""
    out = []
    for i in range(n):
        x = i * scale + (50 * scale if jump_at is not None and i >= jump_at else 0.0)
        out.append(c.CameraPose("j", i, 1.0 + i, "f%d.jpg" % i, 1.0, 0.0, 0.0, 0.0, -x, 0.0, 0.0, 600.0, 0.9))
    return out


def test_pose_features_do_not_depend_on_scene_scale():
    small = c.pose_features(_walk(0.01))
    large = c.pose_features(_walk(100.0))
    assert set(c.POSE_FEATURES) <= set(small)
    for name in c.POSE_FEATURES:
        assert small[name] == pytest.approx(large[name], rel=1e-6), name
    assert small["step_ratio"] == pytest.approx(1.0)


def test_a_jump_stands_out_at_any_scale():
    for scale in (0.01, 100.0):
        assert c.pose_features(_walk(scale, jump_at=11))["step_ratio"] > 20


def _snaps(losses):
    return [c.TrainingSnapshot("j", i, float(i), i * 10, v, 1000, 2.0) for i, v in enumerate(losses)]


def test_rising_loss_flags_a_sustained_climb():
    flagged, rise = c.rising_loss(_snaps([1.0, 0.8, 0.6, 0.5, 0.5, 0.55, 0.62, 0.7, 0.8, 0.9, 1.0, 1.1]))
    assert flagged and rise > 0.5


@pytest.mark.parametrize("losses", [
    [1.0, 0.8, 0.65, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38, 0.37, 0.36, 0.35],        # healthy descent
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],                           # plateau
    [0.5, 0.49, 0.48, 0.47, 0.9, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42],      # one bump (a Difix round)
    [0.5, 0.52, 0.49, 0.51, 0.5, 0.52, 0.49, 0.51, 0.5, 0.51],                    # noise around a level
])
def test_rising_loss_ignores_descent_plateau_bump_and_noise(losses):
    assert c.rising_loss(_snaps(losses))[0] is False


def test_rising_loss_needs_enough_history_and_is_scale_free():
    assert c.rising_loss(_snaps([0.1, 0.2, 0.3]))[0] is False
    base = [0.5, 0.5, 0.52, 0.56, 0.6, 0.66, 0.72, 0.8, 0.9, 1.0]
    assert c.rising_loss(_snaps(base))[0] is c.rising_loss(_snaps([v * 1000 for v in base]))[0] is True
