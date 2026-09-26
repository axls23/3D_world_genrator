"""Schedule-relative trainer constants, plateau early stopping and floater thresholds.

Kept free of CUDA / gsplat imports so it can be unit tested on CPU
(tests/pipeline/test_trainer_schedule.py).

Most step constants in simple_trainer.py were tuned for the reference 30k-step
3DGS schedule. The pipeline typically trains ~7k steps, so those constants are
rescaled proportionally to ``max_steps`` (today's value is the 30k-equivalent and
also the ceiling, so long schedules keep the original behaviour).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

# Schedule that the original constants were tuned for.
REFERENCE_STEPS = 30_000


def schedule_relative(max_steps: int, value_at_ref: int, min_value: int = 1) -> int:
    """Rescale a step constant tuned for REFERENCE_STEPS to a ``max_steps`` schedule.

    Returns ``value_at_ref * max_steps / REFERENCE_STEPS`` clamped to
    ``[min_value, value_at_ref]``: shorter schedules get a proportionally earlier
    step, schedules >= REFERENCE_STEPS keep today's value.
    """
    scaled = int(round(value_at_ref * max_steps / REFERENCE_STEPS))
    return max(min_value, min(value_at_ref, scaled))


@dataclass
class TrainerSchedule:
    refine_stop_iter: int  # MCMC relocation/growth stops here (25k @ 30k)
    sh_degree_interval: int  # one more SH band every N steps (1000 @ 30k)
    depth_warmup_steps: int  # no depth loss before this step (100 @ 30k)
    depth_ramp_steps: int  # linear depth-lambda ramp length (500 @ 30k)
    floater_start: int  # online floater pruning starts after this step (2000 @ 30k)
    floater_every: int  # ... and runs every N steps (400 @ 30k)


def resolve_trainer_schedule(max_steps: int) -> TrainerSchedule:
    """Schedule-relative defaults for every step constant tied to the 30k schedule."""
    return TrainerSchedule(
        # 25k/30k ~= 83%: leave the last ~17% of training for relocation-free
        # convergence instead of relocating until the very last step.
        refine_stop_iter=schedule_relative(max_steps, 25_000),
        # sh_degree=3 is fully enabled after 3 intervals (10% of training).
        sh_degree_interval=schedule_relative(max_steps, 1_000),
        depth_warmup_steps=schedule_relative(max_steps, 100, min_value=0),
        depth_ramp_steps=schedule_relative(max_steps, 500),
        floater_start=schedule_relative(max_steps, 2_000),
        # Same number of pruning passes regardless of schedule length; floor keeps
        # the (CPU kNN) pass from running too often on very short schedules.
        floater_every=schedule_relative(max_steps, 400, min_value=50),
    )


def early_stop_defaults(
    max_steps: int, patience: Optional[int], min_steps: Optional[int]
) -> tuple:
    """Fill unset early-stopping patience / min_steps relative to ``max_steps``.

    patience ~= 7% of the schedule, min_steps ~= 30% (500 / 2000 at 7k steps).
    """
    if patience is None:
        patience = max(50, int(round(0.07 * max_steps)))
    if min_steps is None:
        min_steps = int(round(0.3 * max_steps))
    return patience, min_steps


class PlateauDetector:
    """Loss-plateau test on an EMA of the training loss.

    The per-step loss is very noisy (different image every step), so it is
    smoothed with an EMA. A step counts as an improvement when the EMA drops
    below ``best * (1 - min_delta)`` (relative). ``update`` returns True once
    ``step >= min_steps`` and no improvement happened for ``patience`` steps.
    """

    def __init__(self, patience: int, min_delta: float, min_steps: int, ema_beta: float = 0.99):
        self.patience = patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.ema_beta = ema_beta
        self.ema: Optional[float] = None
        self.best: Optional[float] = None
        self.best_step = 0

    def update(self, step: int, loss: float) -> bool:
        if not (loss == loss) or loss in (float("inf"), float("-inf")):
            return False  # ignore NaN/Inf samples
        if self.ema is None:
            self.ema = loss
        else:
            self.ema = self.ema_beta * self.ema + (1.0 - self.ema_beta) * loss
        if self.best is None or self.ema < self.best * (1.0 - self.min_delta):
            self.best = self.ema
            self.best_step = step
            return False
        return step >= self.min_steps and step - self.best_step >= self.patience


def _robust_score(x: torch.Tensor, lo_q: float = 0.5, hi_q: float = 0.99, sample: int = 100_000) -> torch.Tensor:
    """Score ``x`` relative to its own distribution.

    0 at/below the ``lo_q`` quantile (median), 1 at the ``hi_q`` quantile and
    growing linearly beyond it (not clamped above, so extreme outliers stay
    distinguishable from the rest of the top percentile). Quantiles are estimated on a random subsample (torch.quantile has an input
    size limit and is slow on millions of elements).
    """
    x = x.float()
    ref = x if x.numel() <= sample else x[torch.randint(0, x.numel(), (sample,), device=x.device)]
    q = torch.quantile(ref, torch.tensor([lo_q, hi_q], device=x.device, dtype=x.dtype))
    return torch.clamp((x - q[0]) / (q[1] - q[0] + 1e-8), min=0.0)


def floater_prune_mask(
    iso: Optional[torch.Tensor],
    opacities: torch.Tensor,
    log_scales: torch.Tensor,
    weights: Sequence[float] = (0.4, 0.3, 0.15, 0.15),
    threshold: float = 0.6,
    hard_threshold: float = 3.0,
    max_frac: float = 0.02,
    dead_opacity: float = 1e-4,
) -> torch.Tensor:
    """Select floater Gaussians relative to the model's own distributions.

    Args:
        iso: [N] mean kNN distance (None -> isolation term disabled).
        opacities: [N] activated opacities in [0, 1].
        log_scales: [N, 3] log-scales (``splats["scales"]``).
        weights: (isolation, low opacity, elongation, size) score weights.
        threshold: combined-score threshold (scores clamped to [0, 1]).
        hard_threshold: unclamped isolation / size score that prunes on its
            own: 3.0 = beyond the 99th percentile by twice the median-to-p99
            spread (sky floaters / giant spikes).
        max_frac: at most this fraction of the live Gaussians is pruned per
            call (highest combined scores first).
        dead_opacity: Gaussians at/below this opacity count as already pruned.

    Every score is percentile-based (median -> 0, 99th percentile -> 1), so the
    same thresholds hold regardless of scene scale or training stage. The
    budget is filled by hard-rule hits first, then by unclamped combined score.
    """
    n_total = opacities.shape[0]
    device = opacities.device
    opacities = opacities.reshape(n_total).float()
    # Already-killed Gaussians (opacity ~0) neither count towards the budget nor
    # skew the percentiles, otherwise they would be re-selected on every call.
    alive = opacities > dead_opacity
    result = torch.zeros(n_total, dtype=torch.bool, device=device)
    if int(alive.sum().item()) < 2:
        return result
    opacities = opacities[alive]
    log_scales = log_scales[alive].float()
    iso = iso[alive] if iso is not None else None
    N = opacities.shape[0]

    iso_score = _robust_score(iso.float()) if iso is not None else torch.zeros(N, device=device)
    # Low opacity relative to the median opacity (1st percentile -> 1).
    opacity_score = _robust_score(-opacities, lo_q=0.5, hi_q=0.99)
    # Elongation max/min computed in log space: log(max) - log(min).
    elongation_score = _robust_score(log_scales.max(dim=1).values - log_scales.min(dim=1).values)
    # Size: largest axis (log is monotonic so percentiles match exp-space ones).
    size_score = _robust_score(log_scales.max(dim=1).values)

    scores = (iso_score, opacity_score, elongation_score, size_score)
    floater_prob = sum(w * sc.clamp(max=1.0) for w, sc in zip(weights, scores))
    severity = sum(w * sc for w, sc in zip(weights, scores))

    to_prune = floater_prob > threshold
    # Hard rules: extreme isolation (sky floaters) / giant spikes
    hard = (iso_score > hard_threshold) | (size_score > hard_threshold)
    to_prune |= hard

    budget = int(max_frac * N)
    n_prune = int(to_prune.sum().item())
    if n_prune > budget:
        # Keep the worst `budget` candidates; hard-rule hits rank first.
        rank = torch.where(to_prune, severity + 1e3 * hard.float(), torch.full_like(severity, -1.0))
        keep = torch.topk(rank, budget).indices if budget > 0 else rank.new_zeros(0, dtype=torch.long)
        to_prune = torch.zeros(N, dtype=torch.bool, device=device)
        to_prune[keep] = True
    result[alive] = to_prune
    return result
