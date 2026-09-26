"""Tests for the VRAM-aware training budget strategies (CAP_MAX, DATA_FACTOR, step counts)."""

from hypersplat.pipeline.params.strategies import apply_stage
from hypersplat.pipeline.params.strategies import training_budget as tb


def _apply(profile, config):
    """Run only this module's strategies (other units' modules may exist on sibling branches)."""
    user_set = getattr(config, "USER_SET", set())
    for p in tb.PARAMS:
        if p.NAME in user_set:
            continue
        d = p.derive(profile, config)
        if d is not None:
            setattr(config, p.NAME, d.value)
    return config


def _peak(profile, cfg):
    return tb.estimate_peak_mb(cfg.CAP_MAX, profile.get("frames.width"),
                               profile.get("frames.height"), cfg.DATA_FACTOR)


def test_model_reproduces_whatsapp_oom(whatsapp_profile):
    # 1.15M Gaussians at 480x864, f=1 needed ~5.2 GB and OOM'd with 5140 MB free.
    obs = whatsapp_profile.get("_observed.oom")
    est = tb.estimate_peak_mb(obs["init_points"], *obs["resolution"], obs["data_factor"])
    assert est > whatsapp_profile.get("gpu.free_mb")


def test_whatsapp_fits_with_margin(whatsapp_profile, make_config):
    cfg = _apply(whatsapp_profile, make_config())
    free = whatsapp_profile.get("gpu.free_mb")
    assert cfg.DATA_FACTOR == 1
    assert cfg.CAP_MAX < whatsapp_profile.get("points.count")
    assert _peak(whatsapp_profile, cfg) <= free * (1 - 0.15)


def test_drone_cap_and_resolution(drone_profile, make_config):
    cfg = _apply(drone_profile, make_config())
    assert 150_000 <= cfg.CAP_MAX <= 500_000
    assert cfg.DATA_FACTOR == 1


def test_tiny_gpu_raises_data_factor(drone_profile, whatsapp_profile, make_config):
    for profile in (drone_profile, whatsapp_profile):
        profile.set("gpu.free_mb", 1500)
        cfg = _apply(profile, make_config())
        assert cfg.DATA_FACTOR in (2, 4)
        assert cfg.CAP_MAX >= tb.ABS_MIN_CAP
        assert _peak(profile, cfg) <= 1500


def test_user_set_values_untouched(whatsapp_profile, make_config):
    cfg = make_config(CAP_MAX=1_000_000, DATA_FACTOR=1, TRAINING_MAX_STEPS=300,
                      USER_SET={"CAP_MAX", "DATA_FACTOR", "TRAINING_MAX_STEPS"})
    apply_stage("pre_train", whatsapp_profile, cfg)
    assert (cfg.CAP_MAX, cfg.DATA_FACTOR, cfg.TRAINING_MAX_STEPS) == (1_000_000, 1, 300)
    # Checkpoints and warmup follow the user's max_steps.
    assert cfg.TRAINING_EVAL_STEPS == [150, 300]
    assert cfg.TRAINING_SAVE_STEPS == [150, 300]
    assert 30 <= cfg.POSE_OPT_WARMUP <= 60


def test_user_cap_picks_resolution_it_fits_at(whatsapp_profile, make_config):
    whatsapp_profile.set("gpu.free_mb", 2500)
    cfg = _apply(whatsapp_profile, make_config(CAP_MAX=400_000, USER_SET={"CAP_MAX"}))
    assert cfg.CAP_MAX == 400_000
    assert cfg.DATA_FACTOR > 1
    assert _peak(whatsapp_profile, cfg) <= 2500 * (1 - tb.SAFETY_MARGIN)


def test_steps_consistent(drone_profile, whatsapp_profile, make_config):
    for profile in (drone_profile, whatsapp_profile):
        cfg = _apply(profile, make_config())
        ms = cfg.TRAINING_MAX_STEPS
        assert 3000 <= ms <= 30000
        assert cfg.TRAINING_EVAL_STEPS == cfg.TRAINING_SAVE_STEPS
        assert cfg.TRAINING_EVAL_STEPS[-1] == ms
        assert abs(cfg.TRAINING_EVAL_STEPS[0] - ms / 2) <= 100
        assert abs(cfg.POSE_OPT_WARMUP - 0.15 * ms) <= 10
    # More training images -> more steps.
    d = _apply(drone_profile, make_config()).TRAINING_MAX_STEPS
    w = _apply(whatsapp_profile, make_config()).TRAINING_MAX_STEPS
    assert w > d


def test_missing_signals_keep_fallbacks(make_config):
    from hypersplat.pipeline.params import SceneProfile
    cfg = make_config()
    apply_stage("pre_train", SceneProfile(data={}), cfg)
    assert cfg.CAP_MAX is None and cfg.DATA_FACTOR == 2
    assert cfg.TRAINING_MAX_STEPS == 7000 and cfg.TRAINING_EVAL_STEPS == [3000, 7000]
    assert cfg.POSE_OPT_WARMUP == 1000


def test_reasons_name_numbers(whatsapp_profile, make_config):
    cfg = make_config()
    apply_stage("pre_train", whatsapp_profile, cfg)
    reason = whatsapp_profile.get("decisions.CAP_MAX")["reason"]
    assert "free 5140 MB" in reason and "f=1" in reason
    assert "train imgs" in whatsapp_profile.get("decisions.TRAINING_MAX_STEPS")["reason"]
