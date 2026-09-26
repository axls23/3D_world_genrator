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
    assert s.floater_start < 7000 and s.floater_every >= 50
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
