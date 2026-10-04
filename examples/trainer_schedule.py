"""Schedule-relative trainer constants, plateau / world-convergence early stopping and
floater thresholds.

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
    settle_quiet_steps: int  # stream: no new views this long = coverage settled (4500 @ 30k)
    settle_tail_steps: int  # stream: anneal length once settled (10000 @ 30k)
    settle_min_steps: int  # stream: never settle before this step (3000 @ 30k)


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
        # Interval stays 400: each pass is a CPU kNN over all Gaussians and kills up
        # to floater_max_frac, so short schedules get fewer passes, not denser ones.
        floater_every=400,
        settle_quiet_steps=schedule_relative(max_steps, 4_500, min_value=100),
        # The 2k A/B runs gain ~90% of their loss drop in the last 20% (the anneal); keep
        # at least 1000 steps of it so a settled short run still gets a full anneal.
        settle_tail_steps=max(1_000, schedule_relative(max_steps, 10_000)),
        settle_min_steps=schedule_relative(max_steps, 3_000, min_value=100),
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


class StreamHorizon:
    """Dynamic training length for stream runs.

    A live source has no frame count, so a fixed ``max_steps`` would cut training off while
    new views keep arriving. ``initial`` (cfg.max_steps) is only the first budget: every
    newly registered view pushes the end to at least ``step + extend``, so training runs
    until ``extend`` steps after the last new view (or a STOP). ``cap`` is an optional hard
    ceiling (None: uncapped). ``finish(end)`` pins the end (file source done + tail steps).

    Densification keeps running while the horizon moves: it stops ``refine_gap`` steps before
    the current end (the static schedule's gap, at most extend / 2), never earlier than the
    static ``refine_stop``. Unextended, both match the static schedule exactly.

    ``settle(step, tail)`` pulls the end in to ``step + tail`` once coverage has settled:
    the LR anneals from its current value to the final one over the tail instead of
    following a far-away end. Runs only improve while the LR anneals (the loss is flat
    before it), so a stream whose data stopped changing would otherwise spend thousands
    of steps on a plateau. New views on a settled (not finished) horizon resume it.
    """

    def __init__(self, initial: int, refine_stop: int, extend: Optional[int] = None,
                 cap: Optional[int] = None):
        self.initial = int(initial)
        self.value = self.initial
        self.extend = int(extend) if extend is not None else max(1000, self.initial // 3)
        self.cap = cap
        self.refine_stop0 = int(refine_stop)
        self.refine_gap = min(max(self.initial - self.refine_stop0, 0), self.extend // 2)
        self.finished = False
        self.anchor: Optional[tuple] = None  # (step, decay exponent) where settle() started the anneal

    def settle(self, step: int, tail: int) -> bool:
        """Coverage settled at ``step``: anneal to the end over ``tail`` steps (only ever
        earlier than the current end). True if the end moved."""
        if self.finished or self.anchor is not None:
            return False
        end = step + max(int(tail), 1)
        if end >= self.value:
            return False
        self.anchor = (step, self._exponent(step))
        self.value = end
        return True

    @property
    def settled(self) -> bool:
        return self.anchor is not None

    def anneal_progress(self, step: int) -> float:
        """Fraction of the settle anneal done at ``step`` (0 when not settled)."""
        if self.anchor is None:
            return 0.0
        s0 = self.anchor[0]
        return min(max(step - s0, 0) / max(self.value - s0, 1), 1.0)

    def on_new_views(self, step: int) -> bool:
        """New views registered at ``step``; True if the end moved."""
        if self.finished:
            return False
        self.anchor = None  # the scene changed again: back to the moving schedule
        target = step + self.extend
        if self.cap is not None:
            target = min(target, int(self.cap))
        if target <= self.value:
            return False
        self.value = target
        return True

    def finish(self, end: int):
        self.value = int(end)
        self.finished = True

    @property
    def refine_stop_iter(self) -> int:
        if self.anchor is not None:
            # Settled: same gap before the (earlier) end, but never before the settle step
            return max(self.anchor[0], self.value - self.refine_gap)
        if self.value == self.initial:
            return self.refine_stop0
        return max(self.refine_stop0, self.value - self.refine_gap)

    def _exponent(self, step: int) -> float:
        if self.anchor is None:
            return min(step / max(self.value, 1), 1.0)
        s0, e0 = self.anchor
        return e0 + (1.0 - e0) * self.anneal_progress(step)

    def lr_factor(self, step: int, final: float) -> float:
        """Exponential decay to ``final`` at the current end (ExponentialLR when unextended).
        After settle() it continues from the settle step's value, so the LR never jumps."""
        return final ** self._exponent(step)


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


def _trend(values: Sequence[float]) -> float:
    """Least-squares slope of ``values`` per sample (0 for fewer than two)."""
    n = len(values)
    if n < 2:
        return 0.0
    xm, ym = (n - 1) / 2.0, sum(values) / n
    return sum((i - xm) * (v - ym) for i, v in enumerate(values)) / sum((i - xm) ** 2 for i in range(n))


class WorldConvergence:
    """Early stop for stream runs: stop once the world is adequately reconstructed.

    A stream has no fixed length and its data keeps changing (new keyframes, online Difix
    pseudo views), so a plain loss plateau is meaningless. Instead ``update`` is fed a probe
    every few hundred steps and the run is converged when *all* of these hold for
    ``confirm`` consecutive probes:

      coverage   no newly registered views for ``quiet_steps`` and the ACE map is idle
                 (nothing pending, no refit running): the camera has shown what there is
      quality    held-out PSNR trend < ``min_gain`` dB over the last ``window`` probes, or
                 it is above ``target_psnr`` and gains < ``loose_gain`` dB (a modest world is
                 good enough); without held-out views, the EMA training loss improved
                 < ``loss_rel_gain`` relative over the window
      growth     the Gaussian count changed < ``growth_tol`` (relative) over the window;
                 None skips it - MCMC grows to its cap on a schedule whatever the scene needs
                 (new geometry already shows up as new views in the coverage test)

    and at least ``min_steps`` steps were trained. ``on_data_change`` (new views registered)
    restarts the quality / growth windows, since the target just moved. Difix rounds do not:
    held-out PSNR is measured on real frames, and rounds recur too often for a window to fit
    between them (the trainer waits for the first round instead, via ``extra_wait``).
    """

    def __init__(self, min_steps: int = 3000, window: int = 3, confirm: int = 2,
                 quiet_steps: int = 1500, target_psnr: float = 22.0, min_gain: float = 0.15,
                 loose_gain: float = 0.5, loss_rel_gain: float = 0.01,
                 growth_tol: Optional[float] = 0.02):
        self.min_steps = min_steps
        self.window = max(1, window)
        self.confirm = max(1, confirm)
        self.quiet_steps = quiet_steps
        self.target_psnr = target_psnr
        self.min_gain = min_gain
        self.loose_gain = loose_gain
        self.loss_rel_gain = loss_rel_gain
        self.growth_tol = growth_tol
        self.history: list = []          # (step, psnr or None, loss_ema, num_gs) since last change
        self.last_change = 0
        self.streak = 0
        self.status: dict = {"converged": False, "reason": "warming up"}

    def on_data_change(self, step: int):
        self.history.clear()
        self.last_change = step
        self.streak = 0

    def _quality(self) -> tuple:
        """(ok, human summary) over the last ``window`` + 1 probes.

        The gain is the least-squares trend over the window, not best-minus-first: probe PSNR
        jitters by ~0.5 dB between probes on a plateau, and the max of a few noisy samples
        always beats the first by more than ``min_gain``, so the old test never passed."""
        h = self.history[-(self.window + 1):]
        psnrs = [x[1] for x in h if x[1] is not None]
        if len(psnrs) == len(h):  # held-out views throughout the window
            gain = _trend(psnrs) * self.window
            level = sum(psnrs[-2:]) / len(psnrs[-2:])
            ok = gain < self.min_gain or (level >= self.target_psnr and gain < self.loose_gain)
            return ok, f"PSNR {psnrs[-1]:.2f} dB ({gain:+.2f} trend over {self.window} probes)"
        l0, l1 = h[0][2], h[-1][2]
        rel = (l0 - l1) / max(abs(l0), 1e-8)
        return rel < self.loss_rel_gain, f"loss {l1:.4f} ({-rel:+.1%}, no held-out views)"

    def update(self, step: int, *, psnr: Optional[float], loss_ema: Optional[float], num_gs: int,
               views_quiet_steps: int, map_idle: bool, extra_wait: Optional[str] = None) -> bool:
        """Record a probe; True when the world is converged (stop training)."""
        if loss_ema is None or not (loss_ema == loss_ema):
            return False
        self.history.append((step, psnr, float(loss_ema), int(num_gs)))
        coverage = views_quiet_steps >= self.quiet_steps and map_idle
        cov_txt = ("coverage settled" if coverage else
                   "map busy" if not map_idle else
                   f"new views {views_quiet_steps}/{self.quiet_steps} steps ago")
        if len(self.history) <= self.window:
            self.streak = 0
            self.status = {"converged": False, "reason": f"measuring ({len(self.history)}/"
                           f"{self.window + 1} probes); {cov_txt}"}
            return False
        quality, q_txt = self._quality()
        g0, g1 = self.history[-(self.window + 1)][3], self.history[-1][3]
        growth = abs(g1 - g0) / max(g0, 1)
        grown = self.growth_tol is None or growth < self.growth_tol
        all_ok = coverage and quality and grown and step >= self.min_steps and not extra_wait
        self.streak = self.streak + 1 if all_ok else 0
        converged = self.streak >= self.confirm
        waiting = [t for ok, t in ((coverage, cov_txt), (quality, "quality improving"),
                                   (grown, f"model growing {growth:.1%}"),
                                   (step >= self.min_steps, f"min {self.min_steps} steps"),
                                   (not extra_wait, extra_wait)) if not ok]
        self.status = {
            "converged": converged, "psnr": psnr, "quality": q_txt, "growth": round(growth, 4),
            "coverage": coverage, "streak": self.streak,
            "reason": (f"converged: {q_txt}, {cov_txt}"
                       + (f", model stable ({growth:.1%})" if self.growth_tol is not None else "") if converged else
                       f"converging {self.streak}/{self.confirm}: {q_txt}" if all_ok else
                       "waiting: " + ", ".join(waiting) + f"; {q_txt}"),
        }
        return converged


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
