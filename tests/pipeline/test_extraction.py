"""Frame extraction strategies: FPS (target frames x parallax) and extraction long side."""

import logging

from hypersplat.pipeline.params import SceneProfile
from hypersplat.pipeline.params.strategies import apply_stage, extraction


def _frames(decision, profile):
    return decision.value * profile.get("video.duration")


def test_drone_gets_dense_sampling_not_40_frames(drone_profile, make_config):
    d = extraction.derive(drone_profile, make_config())
    assert d.strategy == "target-frames-parallax"
    # Old fps=2 gave 40 frames (needle artifacts); now >= 100 but tractable for ACE-Zero
    assert 100 <= _frames(d, drone_profile) <= extraction.MAX_FRAMES
    assert d.value <= drone_profile.get("video.fps")
    assert "avg_motion=19.2" in d.reason


def test_whatsapp_frame_count_reasonable(whatsapp_profile, make_config):
    d = extraction.derive(whatsapp_profile, make_config())
    # Observed 139 frames at 15 fps was good
    assert 100 <= _frames(d, whatsapp_profile) <= 200
    assert d.value <= whatsapp_profile.get("video.fps")


def test_fast_motion_raises_fps(drone_profile, make_config):
    drone_profile.set("video.avg_motion", 8.0)
    slow = extraction.derive(drone_profile, make_config()).value
    drone_profile.set("video.avg_motion", 40.0)
    fast = extraction.derive(drone_profile, make_config()).value
    assert fast > slow


def test_missing_motion_uses_base_target(drone_profile, make_config):
    drone_profile.data["video"].pop("avg_motion")
    d = extraction.derive(drone_profile, make_config())
    assert abs(_frames(d, drone_profile) - extraction.TARGET_FRAMES) < 1


def test_short_clip_clamped_to_source_fps(make_config):
    p = SceneProfile(data={"video": {"duration": 3.0, "fps": 30.0, "avg_motion": 20.0}})
    assert extraction.derive(p, make_config()).value == 30.0


def test_long_clip_hits_min_fps_then_frame_cap(make_config):
    p = SceneProfile(data={"video": {"duration": 400.0, "fps": 30.0, "avg_motion": 20.0}})
    assert extraction.derive(p, make_config()).value == extraction.MIN_FPS  # 200 frames
    p.set("video.duration", 1200.0)  # 20 min: 0.5 fps would be 600 frames
    d = extraction.derive(p, make_config())
    assert _frames(d, p) <= extraction.MAX_FRAMES + 1
    assert "capped" in d.reason


def test_no_signal_keeps_fallback(make_config):
    assert extraction.derive(SceneProfile(), make_config()) is None
    assert extraction.derive(None, make_config()) is None
    cfg = make_config()
    apply_stage("pre_ace", SceneProfile(), cfg)
    assert cfg.FPS == 10.0


def test_apply_stage_sets_fps_unless_user_set(drone_profile, make_config):
    cfg = make_config()
    apply_stage("pre_ace", drone_profile, cfg)
    assert cfg.FPS != 10.0
    assert drone_profile.get("decisions.FPS")["strategy"] == "target-frames-parallax"
    cfg = make_config(FPS=2.0, USER_SET={"FPS"})
    apply_stage("pre_ace", drone_profile, cfg)
    assert cfg.FPS == 2.0


def test_long_side_budgets_by_vram(drone_profile):
    assert extraction.derive_long_side(drone_profile).value == 960  # 5.8 GB, 1920x1080 src
    drone_profile.set("gpu.total_mb", 12000)
    assert extraction.derive_long_side(drone_profile).value == 1280
    drone_profile.set("gpu.total_mb", 3000)
    assert extraction.derive_long_side(drone_profile).value == 640


def test_long_side_never_upscales(whatsapp_profile):
    whatsapp_profile.set("gpu.total_mb", 24000)
    assert extraction.derive_long_side(whatsapp_profile).value == 864  # 480x864 source
    p = SceneProfile(data={"gpu": {"total_mb": 12000}, "video": {"width": 641, "height": 360}})
    assert extraction.derive_long_side(p).value == 640  # even


def test_max_long_side_fallback_and_logging(caplog, drone_profile):
    with caplog.at_level(logging.INFO, logger="hypersplat.params"):
        assert extraction.max_long_side(None) == extraction.LONG_SIDE_FALLBACK
        assert extraction.max_long_side(SceneProfile()) == extraction.LONG_SIDE_FALLBACK
        assert extraction.max_long_side(drone_profile) == 960
    assert "[param] EXTRACT_LONG_SIDE=960 (vram-pixel-budget" in caplog.text


def test_scale_filter_and_fitted_size():
    f = extraction.scale_filter(960)
    assert "min(960,iw)" in f and "min(960,ih)" in f and "-2" in f
    assert extraction.fitted_size(1920, 1080, 960) == (960, 540)
    assert extraction.fitted_size(480, 864, 960) == (480, 864)   # no upscale
    assert extraction.fitted_size(1080, 1920, 960) == (540, 960)  # portrait
