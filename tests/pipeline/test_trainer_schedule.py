"""CPU tests for the schedule-relative trainer helpers (examples/trainer_schedule.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from trainer_schedule import (  # noqa: E402
    PlateauDetector,
    WorldConvergence,
    early_stop_defaults,
    floater_prune_mask,
    resolve_trainer_schedule,
    schedule_relative,
)


def test_refine_stop_is_schedule_relative():
    short = resolve_trainer_schedule(7000)
    assert short.refine_stop_iter < 7000
    assert short.refine_stop_iter > 0.7 * 7000  # still refines most of the run
    assert resolve_trainer_schedule(2000).refine_stop_iter < 2000
    assert resolve_trainer_schedule(30_000).refine_stop_iter == 25_000
    assert resolve_trainer_schedule(100_000).refine_stop_iter == 25_000


def test_long_schedules_keep_todays_constants():
    s = resolve_trainer_schedule(30_000)
    assert (s.sh_degree_interval, s.depth_warmup_steps, s.depth_ramp_steps) == (1000, 100, 500)
    assert (s.floater_start, s.floater_every) == (2000, 400)


def test_short_schedule_scales_proportionally():
    s = resolve_trainer_schedule(7000)
    assert s.sh_degree_interval == schedule_relative(7000, 1000) == 233
    assert 3 * s.sh_degree_interval < 7000 * 0.15  # full SH well before the end
    assert s.floater_start < 7000 and s.floater_every == 400
    assert schedule_relative(10, 400, min_value=50) == 50


def test_early_stop_defaults_relative_and_explicit_wins():
    assert early_stop_defaults(7000, None, None) == (490, 2100)
    assert early_stop_defaults(30_000, None, None) == (2100, 9000)
    assert early_stop_defaults(7000, 123, 456) == (123, 456)


def _run(detector, losses):
    for step, loss in enumerate(losses):
        if detector.update(step, float(loss)):
            return step
    return None


def test_plateau_triggers_on_flat_curve():
    rng = np.random.default_rng(0)
    flat = 0.1 + 0.005 * rng.standard_normal(3000)
    det = PlateauDetector(patience=300, min_delta=1e-3, min_steps=1000)
    stop = _run(det, flat)
    assert stop is not None
    assert stop >= 1000


def test_plateau_does_not_trigger_on_decreasing_curve():
    rng = np.random.default_rng(0)
    steps = np.arange(7000)
    decreasing = 0.3 * np.exp(-steps / 3000) + 0.05 + 0.005 * rng.standard_normal(7000)
    det = PlateauDetector(patience=490, min_delta=1e-3, min_steps=2100)
    assert _run(det, decreasing) is None


def test_plateau_ignores_nan():
    det = PlateauDetector(patience=1, min_delta=1e-3, min_steps=0)
    assert det.update(0, float("nan")) is False
    assert det.ema is None


def _gaussians(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    iso = torch.rand(n, generator=g) * 0.1
    opac = 0.5 + 0.5 * torch.rand(n, generator=g)
    log_scales = torch.log(0.01 + 0.01 * torch.rand(n, 3, generator=g))
    return iso, opac, log_scales


@pytest.mark.parametrize("scene_scale", [1e-3, 1.0, 1e3])
def test_floater_removal_bounded_and_scale_invariant(scene_scale):
    n = 5000
    iso, opac, log_scales = _gaussians(n)
    iso = iso * scene_scale
    log_scales = log_scales + np.log(scene_scale)
    mask = floater_prune_mask(iso, opac, log_scales, max_frac=0.02)
    assert mask.dtype == torch.bool and mask.shape == (n,)
    assert mask.sum().item() <= int(0.02 * n)


def test_floater_mask_targets_outliers_first():
    n = 5000
    iso, opac, log_scales = _gaussians(n)
    outliers = torch.arange(10)
    iso[outliers] = 10.0  # far from everything
    log_scales[outliers] = torch.log(torch.tensor([5.0, 0.01, 0.01]))  # giant spikes
    mask = floater_prune_mask(iso, opac, log_scales, max_frac=0.02)
    assert mask[outliers].all()
    # Same outliers selected with a tiny budget (hard-rule hits rank first)
    mask_small = floater_prune_mask(iso, opac, log_scales, max_frac=10 / n)
    assert mask_small.sum().item() == 10 and mask_small[outliers].all()


def test_floater_mask_skips_already_dead():
    n = 2000
    iso, opac, log_scales = _gaussians(n)
    opac[:500] = 0.0  # killed on a previous pass
    mask = floater_prune_mask(iso, opac, log_scales, max_frac=0.02)
    assert not mask[:500].any()
    assert mask.sum().item() <= int(0.02 * 1500)


def test_floater_mask_without_isolation():
    iso, opac, log_scales = _gaussians(1000)
    mask = floater_prune_mask(None, opac, log_scales)
    assert mask.sum().item() <= 20


def test_stream_horizon_extends_uncapped_and_matches_static_when_idle():
    from trainer_schedule import StreamHorizon

    h = StreamHorizon(30_000, 25_000)
    assert h.extend == 10_000 and h.refine_stop_iter == 25_000
    assert h.lr_factor(15_000, 0.1) == pytest.approx(0.1 ** 0.5)  # ExponentialLR when idle
    assert not h.on_new_views(100)  # still inside the initial budget
    for step in range(0, 200_000, 50):  # views keep arriving: never capped
        h.on_new_views(step)
    assert h.value == 199_950 + 10_000 and h.refine_stop_iter == h.value - 5_000
    h.finish(200_000)
    assert h.value == 200_000 and not h.on_new_views(200_000)


def test_stream_horizon_optional_cap():
    from trainer_schedule import StreamHorizon

    h = StreamHorizon(7_000, 5_833, extend=2_000, cap=12_000)
    h.on_new_views(11_000)
    assert h.value == 12_000


# ---------------------------------------------------------------- WorldConvergence
def _run_world(conv, psnrs, *, start=500, every=500, quiet_from=0, gs=100_000, loss=0.05,
               map_idle=True, wait=None):
    """Feed one probe per psnr value; step of the first True (or None)."""
    for i, p in enumerate(psnrs):
        step = start + i * every
        if conv.update(step, psnr=p, loss_ema=loss, num_gs=gs, views_quiet_steps=step - quiet_from,
                       map_idle=map_idle, extra_wait=wait):
            return step
    return None


def test_world_converges_on_plateau_after_confirmation():
    conv = WorldConvergence(min_steps=2000, window=3, confirm=2, quiet_steps=1000)
    psnrs = [18, 20, 21.5, 21.9, 22.0, 22.05, 22.08, 22.09, 22.1, 22.1]
    step = _run_world(conv, psnrs)
    assert step is not None and step >= 2000
    assert conv.status["converged"] and "converged" in conv.status["reason"]
    # needs `confirm` consecutive passing probes, not just one
    assert conv.streak == 2


def test_world_waits_while_quality_still_rising():
    conv = WorldConvergence(min_steps=0, window=3, confirm=2, quiet_steps=0, target_psnr=40)
    assert _run_world(conv, [15 + i for i in range(12)]) is None
    assert "quality improving" in conv.status["reason"]


def test_world_modest_target_accepts_small_gains():
    # +0.3 dB per window: above min_gain (not a plateau) but under loose_gain above target
    kw = dict(min_steps=0, window=3, confirm=2, quiet_steps=0, min_gain=0.15, loose_gain=0.5)
    psnrs = [23 + 0.1 * i for i in range(10)]
    assert _run_world(WorldConvergence(target_psnr=22.0, **kw), psnrs) is not None
    assert _run_world(WorldConvergence(target_psnr=30.0, **kw), psnrs) is None


def test_world_blocked_by_coverage_growth_and_waits():
    flat = [25.0] * 12
    kw = dict(min_steps=0, window=3, confirm=2, quiet_steps=1500)
    # new views keep arriving (quiet counter never reaches quiet_steps)
    conv = WorldConvergence(**kw)
    for i, p in enumerate(flat):
        step = 500 + 500 * i
        assert not conv.update(step, psnr=p, loss_ema=0.05, num_gs=1000, views_quiet_steps=100,
                               map_idle=True)
    assert "new views" in conv.status["reason"]
    assert _run_world(WorldConvergence(**kw), flat, map_idle=False) is None
    assert _run_world(WorldConvergence(**kw), flat, wait="first Difix round") is None
    # Gaussian count still growing 10% per probe
    conv = WorldConvergence(**kw)
    for i, p in enumerate(flat):
        step = 500 + 500 * i
        assert not conv.update(step, psnr=p, loss_ema=0.05, num_gs=int(1000 * 1.1 ** i),
                               views_quiet_steps=step, map_idle=True)
    assert "model growing" in conv.status["reason"]


def test_world_data_change_restarts_windows():
    conv = WorldConvergence(min_steps=0, window=3, confirm=2, quiet_steps=0)
    for i in range(4):
        conv.update(500 * (i + 1), psnr=25.0, loss_ema=0.05, num_gs=1000, views_quiet_steps=5000,
                    map_idle=True)
    assert conv.streak == 1
    conv.on_data_change(2000)
    assert conv.streak == 0 and not conv.history
    assert not conv.update(2500, psnr=25.0, loss_ema=0.05, num_gs=1000, views_quiet_steps=500,
                           map_idle=True)
    assert "measuring" in conv.status["reason"]


def test_world_falls_back_to_loss_without_held_out_views():
    kw = dict(min_steps=0, window=3, confirm=2, quiet_steps=0)
    conv = WorldConvergence(**kw)
    losses = [0.2, 0.1, 0.06, 0.05, 0.0499, 0.0498, 0.0498, 0.0497, 0.0497]
    hit = None
    for i, l in enumerate(losses):
        if conv.update(500 * (i + 1), psnr=None, loss_ema=l, num_gs=1000, views_quiet_steps=10_000,
                       map_idle=True):
            hit = i
            break
    assert hit is not None and hit >= 5
    assert "no held-out views" in conv.status["quality"]


def test_world_growth_ignored_when_disabled():
    # MCMC-style: Gaussians grow 10 % per probe toward the cap while quality is flat
    conv = WorldConvergence(min_steps=0, window=3, confirm=2, quiet_steps=0, growth_tol=None)
    hit = None
    for i in range(12):
        if conv.update(500 * (i + 1), psnr=23.3, loss_ema=0.05, num_gs=int(1000 * 1.1 ** i),
                       views_quiet_steps=10_000, map_idle=True):
            hit = i
            break
    assert hit == 4  # window of 3 probes + 2 confirmations


def test_stream_horizon_settle_anneals_continuously_to_the_earlier_end():
    from trainer_schedule import StreamHorizon

    h = StreamHorizon(10_000, 8_333)
    before = h.lr_factor(2_500, 0.1)
    assert h.settle(2_500, 3_000) and h.settled and h.value == 5_500
    assert h.lr_factor(2_500, 0.1) == pytest.approx(before)  # no jump at the settle step
    assert h.lr_factor(5_500, 0.1) == pytest.approx(0.1)  # fully annealed at the new end
    assert h.lr_factor(4_000, 0.01) < h.lr_factor(4_000, 0.1)  # every final LR anneals with it
    assert h.anneal_progress(4_000) == pytest.approx(0.5)
    assert h.refine_stop_iter == 5_500 - h.refine_gap
    assert not h.settle(3_000, 1_000)  # once only


def test_stream_horizon_settle_never_extends_and_resumes_on_new_views():
    from trainer_schedule import StreamHorizon

    h = StreamHorizon(2_000, 1_667)
    assert not h.settle(1_500, 1_000) and not h.settled  # would end later than 2000
    h = StreamHorizon(10_000, 8_333, extend=3_000)
    h.settle(2_000, 2_000)
    assert h.on_new_views(3_000) and not h.settled and h.value == 6_000
    assert h.lr_factor(3_000, 0.1) == pytest.approx(0.1 ** 0.5)  # moving schedule again


def test_settle_defaults_are_schedule_relative():
    s2, s30 = resolve_trainer_schedule(2_000), resolve_trainer_schedule(30_000)
    assert s2.settle_tail_steps == 1_000 and s30.settle_tail_steps == 10_000
    assert s2.settle_quiet_steps == 300 and s30.settle_quiet_steps == 4_500


def test_world_converges_on_noisy_plateau():
    # The 10k stream run's probes: ~0.5 dB jitter around 20 dB with no trend. The old
    # best-minus-first test kept seeing a "gain" and never converged.
    noisy = [19.93, 20.87, 20.41, 19.67, 20.39, 20.8, 20.1, 20.09, 20.73, 20.43]
    conv = WorldConvergence(min_steps=0, window=3, confirm=2, quiet_steps=0, target_psnr=30)
    assert _run_world(conv, noisy) is not None
