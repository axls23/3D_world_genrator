"""Initial point cloud budget (strategies/points.py), replayed on the real drone/WhatsApp runs."""

import os
import time

import pytest

from hypersplat.pipeline.params import SceneProfile
from hypersplat.pipeline.params.strategies import points


def _frames(profile):
    return (profile.get("frames.width"), profile.get("frames.height"), profile.get("frames.count"))


def test_whatsapp_budget_avoids_the_oom_cloud(whatsapp_profile):
    w, h, n = _frames(whatsapp_profile)
    observed = whatsapp_profile.get("_observed.oom.init_points")
    d = points.derive_subsample(whatsapp_profile, w, h, n)
    assert d.strategy == "point-budget"
    expected_points = d.value * w * h * n
    assert expected_points < observed / 3          # was 1,153,899 at 2%
    assert 200000 <= expected_points <= 400000     # ~300k on a 6 GB card
    assert d.value < points.DEFAULT_SUBSAMPLE / 3


def test_drone_budget_is_reasonable(drone_profile):
    w, h, n = _frames(drone_profile)
    d = points.derive_subsample(drone_profile, w, h, n)
    expected_points = d.value * w * h * n
    assert 250000 <= expected_points <= 350000
    assert d.value <= points.MAX_SUBSAMPLE


def test_fraction_math():
    profile = SceneProfile(data={"gpu": {"free_mb": 5500, "total_mb": 6000}})
    target = points.target_points(profile).value
    assert target == 300000
    d = points.derive_subsample(profile, 1000, 500, 20)   # 10M pixels
    assert d.value == pytest.approx(target / 10e6)


def test_target_clamped():
    tiny = SceneProfile(data={"gpu": {"free_mb": 100}})
    huge = SceneProfile(data={"gpu": {"free_mb": 80000}})
    total_only = SceneProfile(data={"gpu": {"total_mb": 5500}})
    assert points.target_points(tiny).value == points.MIN_POINTS
    assert points.target_points(huge).value == points.MAX_POINTS
    assert points.target_points(total_only).value == 300000
    # a zero free-memory reading must not fall back to total memory (the OOM case)
    full = SceneProfile(data={"gpu": {"free_mb": 0, "total_mb": 24000}})
    assert points.target_points(full).value == points.MIN_POINTS


def test_density_cap_for_few_small_frames(drone_profile):
    d = points.derive_subsample(drone_profile, 320, 180, 5)
    assert d.value == points.MAX_SUBSAMPLE
    assert "density-capped" in d.reason


def test_explicit_subsample_wins(whatsapp_profile):
    assert points.resolve_subsample(0.1, whatsapp_profile, 480, 864, 139) == 0.1
    assert whatsapp_profile.get("decisions.INIT_POINT_SUBSAMPLE.strategy") == "user"


def test_no_profile_falls_back_to_old_fraction():
    assert points.resolve_subsample(None, None, 480, 864, 139) == points.DEFAULT_SUBSAMPLE
    assert points.resolve_subsample(None, SceneProfile(data={}), 480, 864, 139) == 0.02
    # unreadable images (0x0) also fall back
    assert points.resolve_subsample(None, SceneProfile(data={"gpu": {"free_mb": 5000}}),
                                    0, 0, 139) == 0.02


def test_decision_recorded_and_logged(whatsapp_profile, caplog):
    with caplog.at_level("INFO", logger="hypersplat.params"):
        value = points.resolve_subsample(None, whatsapp_profile, 480, 864, 139)
    assert whatsapp_profile.get("decisions.INIT_POINT_SUBSAMPLE.value") == value
    assert "[param] INIT_POINT_SUBSAMPLE=" in caplog.text
    assert "point-budget" in caplog.text


def test_pick_poses_file_prefers_fresh_filtered(tmp_path):
    raw = tmp_path / "poses_final.txt"
    filt = tmp_path / "poses_final_filtered.txt"
    raw.write_text("raw\n")
    assert points.pick_poses_file(raw) == raw
    filt.write_text("filtered\n")
    assert points.pick_poses_file(raw) == filt
    # stale filtered file from a previous ACE-Zero run is ignored
    past = time.time() - 100
    os.utime(filt, (past, past))
    assert points.pick_poses_file(raw) == raw


def test_not_a_config_strategy():
    """Helpers only: apply_stage must not try to set a config attribute from this module."""
    from hypersplat.pipeline.params.strategies import discover
    assert all(p.NAME != points.PARAM_NAME for p in discover())
