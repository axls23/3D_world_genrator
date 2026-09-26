"""ACE-Zero schedule: smooth, envelope-bounded functions of the measured scene."""

import logging

import pytest

from hypersplat.pipeline.params import SceneProfile
from hypersplat.pipeline.params.strategies import ace_schedule as S


def _schedule(profile, mode="balanced", cpu=24, **kw):
    args = dict(
        avg_motion=profile.get("video.avg_motion"), num_frames=profile.get("frames.count"),
        frame_width=profile.get("frames.width"), frame_height=profile.get("frames.height"),
        quality_mode=mode, cpu_count=cpu, free_mb=profile.get("gpu.free_mb"))
    args.update(kw)
    return {k: d.value for k, d in S.compute_schedule(**args).items()}


@pytest.mark.parametrize("mode", ["fast", "balanced", "quality"])
@pytest.mark.parametrize("fixture", ["drone_profile", "whatsapp_profile"])
def test_real_profiles_stay_in_old_envelope(request, fixture, mode):
    v = _schedule(request.getfixturevalue(fixture), mode)
    for key, (lo, hi) in S.ENVELOPE.items():
        if key == "registration_confidence":  # scaled by (resolution/480)^2
            lo *= (v["image_resolution"] / 480.0) ** 2
        if key == "try_seeds" and mode == "fast":
            assert v[key] == 1
            continue
        assert lo <= v[key] <= hi, (key, v[key])
    assert v["cooldown_threshold"] == S.PRESETS[mode]["cooldown_threshold"]
    assert v["registration_threshold"] == S.PRESETS[mode]["registration_threshold"]
    assert 1 <= v["seed_parallel_workers"] <= 8
    assert v["num_head_blocks"] in (1, 2)


def test_real_profile_values(drone_profile, whatsapp_profile):
    drone, wa = _schedule(drone_profile), _schedule(whatsapp_profile)
    # Drone frames are 640x360: ACE must not upscale them to 480
    assert drone["image_resolution"] == 360
    assert wa["image_resolution"] == 480
    # Shakier handheld clip -> tighter clamp, stricter registration, more seeds
    assert wa["repro_loss_soft_clamp"] < drone["repro_loss_soft_clamp"]
    assert wa["registration_confidence"] > drone["registration_confidence"]
    assert S.registration_confidence(19.2, 480).value > drone["registration_confidence"]
    assert wa["try_seeds"] >= drone["try_seeds"]
    # 139 vs 40 frames -> longer mapping
    assert wa["hybrid_train_iterations"] > drone["hybrid_train_iterations"]
    # Both fit in 6GB with the buffer on the GPU (what ran before)
    assert drone["training_buffer_cpu"] is False and wa["training_buffer_cpu"] is False


def test_monotonic_in_frames():
    prev = None
    for n in (10, 30, 60, 100, 150, 250, 400, 1000):
        v = {k: d.value for k, d in S.compute_schedule(20.0, n, 640, 360, "balanced", 24, None).items()}
        if prev:
            for key in ("hybrid_train_iterations", "seed_iterations", "refit_iterations",
                        "hybrid_pose_wait", "cooldown_iterations", "iterations_max",
                        "try_seeds", "num_data_workers"):
                assert v[key] >= prev[key], (key, n)
            assert v["learning_rate_max"] <= prev["learning_rate_max"]
        prev = v


def test_monotonic_in_motion_and_continuous():
    prev = None
    for m in [x * 0.5 for x in range(0, 101)]:
        v = {k: d.value for k, d in S.compute_schedule(m, 100, 640, 360, "balanced", 24, None).items()}
        if prev:
            assert v["repro_loss_soft_clamp"] <= prev["repro_loss_soft_clamp"]
            assert v["registration_confidence"] >= prev["registration_confidence"]
            assert v["try_seeds"] >= prev["try_seeds"]
            assert v["aug_rotation"] >= prev["aug_rotation"]
            # No step-function jumps: 0.5 motion units change the clamp by at most 1
            assert prev["repro_loss_soft_clamp"] - v["repro_loss_soft_clamp"] <= 1
            assert v["registration_confidence"] - prev["registration_confidence"] <= 30
        prev = v
    # Old bucket endpoints are still reached
    lo = {k: d.value for k, d in S.compute_schedule(5, 100, 640, 480).items()}
    hi = {k: d.value for k, d in S.compute_schedule(40, 100, 640, 480).items()}
    assert (lo["repro_loss_soft_clamp"], lo["registration_confidence"]) == (50, 500)
    assert (hi["repro_loss_soft_clamp"], hi["registration_confidence"]) == (30, 1500)


def test_quality_mode_is_a_preset_scale():
    fast, bal, qual = (
        {k: d.value for k, d in S.compute_schedule(20, 139, 480, 864, m, 24, 5000).items()}
        for m in ("fast", "balanced", "quality"))
    assert fast["hybrid_train_iterations"] < bal["hybrid_train_iterations"] < qual["hybrid_train_iterations"]
    assert fast["try_seeds"] == 1 and fast["seed_parallel_workers"] == 1
    assert fast["training_buffer_cpu"] is True


def test_image_resolution_never_upscales():
    assert S.image_resolution(640, 360).value == 360
    assert S.image_resolution(1920, 1080).value == 480
    assert S.image_resolution(0, 0).value == 480


def test_resources_from_cpu_and_vram():
    assert S.num_data_workers(139, 4).value == 2
    assert S.num_data_workers(139, 24).value == 6
    assert S.seed_parallel_workers(10, 6, None, "balanced").value == 2
    assert S.seed_parallel_workers(10, 24, 1500, "balanced").value == 2
    assert S.seed_parallel_workers(10, 24, None, "balanced").value == 8
    # Large scene on a small GPU: buffer to CPU, single head
    assert S.training_buffer_cpu(1000, 5000, "balanced").value is True
    assert S.num_head_blocks(1000, 3000, False).value == 1
    assert S.num_head_blocks(1000, 24000, False).value == 2
    assert S.num_head_blocks(100, 24000, False).value == 1


def test_arg_builder_passes_soft_clamp_and_resolution(whatsapp_profile):
    values = _schedule(whatsapp_profile)
    values.update(refinement="mlp", refinement_ortho="gram-schmidt",
                  pose_refinement_wait=5000, pose_refinement_lr=0.001)
    for hybrid in (True, False):
        args = S.build_ace_args(values, hybrid)
        opts = dict(zip(args[::2], args[1::2]))
        assert opts["--repro_loss_soft_clamp"] == str(values["repro_loss_soft_clamp"])
        assert opts["--image_resolution"] == str(values["image_resolution"])
        if hybrid:
            assert opts["--registration_confidence"] == str(values["registration_confidence"])
        assert opts["--training_buffer_cpu"] == "False"
        if not hybrid:
            assert opts["--try_seeds"] == str(values["try_seeds"])
            assert opts["--learning_rate_max"] == str(values["learning_rate_max"])
    with pytest.raises(KeyError):
        S.build_ace_args({}, True)


def test_arg_builder_matches_script_parsers():
    """Every option we pass must exist in the ACE-Zero script's argparse."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "scripts" / "acezero"
    for script, names in (("ace_zero_hybrid.py", S.HYBRID_ARGS), ("ace_zero.py", S.ITERATIVE_ARGS)):
        src = (root / script).read_text()
        for name in names:
            assert f"'--{name}'" in src, (script, name)


def test_estimator_uses_schedule(tmp_path, whatsapp_profile, monkeypatch):
    """perception wiring: computed values (not getattr fallbacks) reach the command line."""
    from hypersplat.pipeline.wrappers.perception import ACEZeroPoseEstimator
    whatsapp_profile.path = tmp_path / "scene_profile.json"
    whatsapp_profile.save()
    monkeypatch.setenv("HYPERSPLAT_PROFILE", str(whatsapp_profile.path))
    est = ACEZeroPoseEstimator(output_dir=tmp_path / "ace")
    est.video_info = {"avg_motion": 28.28}
    est.num_frames, est.image_width, est.image_height = 40, 640, 360
    est._compute_adaptive_params()
    args = est._ace_cli_args()
    opts = dict(zip(args[::2], args[1::2]))
    assert opts["--image_resolution"] == "360"
    assert opts["--repro_loss_soft_clamp"] == str(S.soft_clamp(28.28).value) != "50"
    saved = SceneProfile.load(whatsapp_profile.path)
    assert saved.get("decisions.ACE_IMAGE_RESOLUTION.value") == 360


# ---------------------------------------------------------------------------
# Pose filter
# ---------------------------------------------------------------------------
def test_filter_whatsapp_keeps_good_frames(whatsapp_profile):
    confs = list(whatsapp_profile.get("ace.conf").values())
    d = S.robust_min_confidence(confs, 1000)
    kept = [c for c in confs if c >= d.value]
    assert len(kept) >= 0.8 * len(confs)
    dropped = [c for c in confs if c < d.value]
    assert all(c < 3500 for c in dropped)  # only the low tail (median 5483)


def test_filter_drone_drops_nothing(drone_profile):
    confs = list(drone_profile.get("ace.conf").values())
    d = S.robust_min_confidence(confs, 1000)
    assert d.value == 1000  # MAD ~16 would cut good frames; never raise the base threshold
    assert all(c >= d.value for c in confs)


def test_filter_lowers_threshold_for_low_confidence_scene():
    confs = [600 + i for i in range(50)] + [100, 120]
    d = S.robust_min_confidence(confs, 1000)
    kept = [c for c in confs if c >= d.value]
    assert len(kept) == 50  # base 1000 would have dropped everything


def test_filter_never_drops_more_than_20_percent():
    confs = [100.0] * 40 + [5000.0] * 60
    d = S.robust_min_confidence(confs, 1000)
    assert sum(1 for c in confs if c < d.value) <= 20


def test_filter_poses_end_to_end(tmp_path, whatsapp_profile, monkeypatch, caplog):
    from hypersplat.pipeline.wrappers.perception import ACEZeroPoseEstimator
    whatsapp_profile.path = tmp_path / "scene_profile.json"
    whatsapp_profile.save()
    monkeypatch.setenv("HYPERSPLAT_PROFILE", str(whatsapp_profile.path))
    est = ACEZeroPoseEstimator(output_dir=tmp_path / "ace")
    est.acezero_output.mkdir(parents=True)
    conf = whatsapp_profile.get("ace.conf")
    lines = [f"{n} 1 0 0 0 0 0 0 593.4 {c}\n" for n, c in conf.items()] + ["lowconf.jpg 1 0 0 0 0 0 0 593.4 50\n"]
    (est.acezero_output / "poses_final.txt").write_text("".join(lines))
    est.poses = {n: None for n in list(conf) + ["lowconf.jpg"]}
    with caplog.at_level(logging.INFO, logger="hypersplat.params"):
        good, bad, _ = est._filter_poses()
    assert bad == 1 and good == len(conf)
    assert any("[param] MIN_REGISTRATION_CONFIDENCE=" in r.message for r in caplog.records)
    assert SceneProfile.load(whatsapp_profile.path).get("ace.min_conf_used") == 1000

    # An explicit user value is applied as-is
    est.min_confidence = 5000
    est.poses = {n: None for n in conf}
    good, bad, _ = est._filter_poses()
    assert bad == sum(1 for c in conf.values() if c < 5000)


def test_post_ace_strategy(whatsapp_profile, make_config):
    from hypersplat.pipeline.params.strategies import apply_stage, discover
    assert any(p.NAME == "MIN_REGISTRATION_CONFIDENCE" and p.STAGE == "post_ace" for p in discover())
    cfg = make_config()
    whatsapp_profile.set("ace.min_conf_used", 900)
    apply_stage("post_ace", whatsapp_profile, cfg)
    assert cfg.MIN_REGISTRATION_CONFIDENCE == 900
    cfg = make_config(MIN_REGISTRATION_CONFIDENCE=3000, USER_SET={"MIN_REGISTRATION_CONFIDENCE"})
    apply_stage("post_ace", whatsapp_profile, cfg)
    assert cfg.MIN_REGISTRATION_CONFIDENCE == 3000
