"""Tests for the dynamic-parameter foundation: profile, resolve, strategy discovery."""

import json
import struct
from types import SimpleNamespace

from hypersplat.pipeline.params import Decision, SceneProfile, resolve
from hypersplat.pipeline.params import signals
from hypersplat.pipeline.params import strategies
from hypersplat.pipeline.params.strategies import Param, apply_stage


def test_profile_roundtrip(tmp_path):
    p = SceneProfile.load(tmp_path / "scene_profile.json")
    p.set("gpu.free_mb", 5000, "test")
    p.append("difix.rounds", {"angle": 6})
    p.save()
    q = SceneProfile.load(tmp_path / "scene_profile.json")
    assert q.get("gpu.free_mb") == 5000
    assert q.get("difix.rounds") == [{"angle": 6}]
    assert q.get("missing.key", 7) == 7
    assert q.get("_sources")["gpu.free_mb"] == "test"


def test_profile_save_merges_concurrent_writers(tmp_path):
    f = tmp_path / "scene_profile.json"
    manager_view = SceneProfile.load(f)
    manager_view.set("gpu.free_mb", 5000)
    manager_view.save()
    subprocess_view = SceneProfile.load(f)               # e.g. generate_points_wsl.py
    subprocess_view.set("decisions.INIT_POINT_SUBSAMPLE", {"value": 0.03})
    subprocess_view.set("points.count", 309050)
    subprocess_view.save()
    manager_view.set("decisions.CAP_MAX", {"value": 300000})  # stale in-memory copy
    manager_view.save()
    merged = SceneProfile.load(f)
    assert merged.get("decisions.INIT_POINT_SUBSAMPLE") == {"value": 0.03}
    assert merged.get("decisions.CAP_MAX") == {"value": 300000}
    assert merged.get("points.count") == 309050
    assert manager_view.get("points.count") == 309050    # in-memory copy refreshed


def test_profile_discard_drops_previous_run(tmp_path):
    f = tmp_path / "scene_profile.json"
    old = SceneProfile.load(f)
    old.set("ace.focal_median", 593.0)
    old.set("train.val", [{"step": 6999, "psnr": 12.7}])
    old.set("decisions.CAP_MAX", {"value": 1})
    old.save()
    rerun = SceneProfile.load(f)
    rerun.discard("train", "difix", "decisions")
    rerun.set("gpu.free_mb", 5000)
    rerun.save()
    fresh = SceneProfile.load(f)
    assert fresh.get("ace.focal_median") == 593.0      # scene signal kept
    assert fresh.get("train.val") is None               # previous run's results gone
    assert fresh.get("decisions") is None


def test_profile_load_corrupt_file(tmp_path):
    f = tmp_path / "scene_profile.json"
    f.write_text("{not json")
    assert SceneProfile.load(f).data == {}


def test_resolve_precedence(drone_profile):
    derive = lambda: Decision(42, "rule", "why")
    assert resolve("X", 7, derive, 1, drone_profile) == 7          # user wins
    assert resolve("X", None, derive, 1, drone_profile) == 42      # then derived
    assert drone_profile.get("decisions.X")["strategy"] == "rule"
    assert resolve("X", None, lambda: None, 1, drone_profile) == 1  # then fallback


def test_resolve_survives_broken_strategy():
    def boom():
        raise RuntimeError("bad")
    assert resolve("X", None, boom, 3) == 3


def test_apply_stage_respects_user_set(monkeypatch, make_config, drone_profile):
    fake = [Param("CAP_MAX", "pre_train", lambda prof, cfg: Decision(123, "t", "r")),
            Param("FPS", "pre_train", lambda prof, cfg: Decision(99.0, "t", "r")),
            Param("DATA_FACTOR", "loop", lambda prof, cfg: Decision(8, "t", "r"))]
    monkeypatch.setattr(strategies, "discover", lambda: fake)
    cfg = make_config(USER_SET={"FPS"}, FPS=5.0)
    apply_stage("pre_train", drone_profile, cfg)
    assert cfg.CAP_MAX == 123          # derived
    assert cfg.FPS == 5.0              # user-set value untouched
    assert cfg.DATA_FACTOR == 2        # other stage not applied
    assert drone_profile.get("decisions.CAP_MAX")["value"] == 123


def test_discover_real_package_is_valid():
    for p in strategies.discover():
        assert p.STAGE in strategies.STAGES
        assert callable(p.derive)


def test_read_ace_poses_and_points(tmp_path):
    (tmp_path / "acezero_output").mkdir()
    (tmp_path / "acezero_output" / "poses_final.txt").write_text(
        "/x/images/frame_00001.jpg 1 0 0 0 0 0 0 593.0 5000\n"
        "/x/images/frame_00002.jpg 1 0 0 0 0 0 0 595.0 2200\n")
    ace = signals.read_ace_poses(tmp_path)
    assert ace["focal_median"] == 594.0
    assert ace["conf"] == {"frame_00001.jpg": 5000.0, "frame_00002.jpg": 2200.0}
    pts = tmp_path / "points3D.bin"
    pts.write_bytes(struct.pack("<Q", 184610) + b"\0" * 64)
    assert signals.count_points3d(pts) == 184610


def test_read_val_stats_sorts_numerically(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    for step, psnr in [(2999, 20.0), (10099, 25.0), (6999, 23.0)]:
        (stats / f"val_step{step:04d}.json").write_text(json.dumps({"psnr": psnr}))
    assert [r["step"] for r in signals.read_val_stats(tmp_path)] == [2999, 6999, 10099]


def test_fixtures_have_schema(drone_profile, whatsapp_profile):
    for prof in (drone_profile, whatsapp_profile):
        for key in ("video.width", "video.avg_motion", "frames.count", "ace.focal_median",
                    "ace.conf", "points.count", "gpu.free_mb", "gpu.total_mb"):
            assert prof.get(key) is not None, key
    assert whatsapp_profile.get("points.count") == 1153899
    assert round(drone_profile.get("ace.focal_median")) == 317
