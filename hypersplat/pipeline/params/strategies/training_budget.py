"""Training budget: Gaussian cap / resolution from a VRAM model, step counts from frame count.

Stage pre_train (gpu.* is refreshed right before training starts).

CAP_MAX / DATA_FACTOR - "vram-budget". Peak trainer VRAM is modelled as

    peak_mb ~= A * N_gaussians + B * (W/f) * (H/f) + C

and must fit within gpu.free_mb minus a safety margin. The trainer subsamples the initial
cloud down to cap_max (MCMC never removes Gaussians), so N_gaussians at the peak is the cap.
Calibration (RTX 4050 Laptop, 2026-09-26):
  - 1,153,899 initial Gaussians at 480x864, f=1 OOM'd at ~4.79 GB used while asking for
    360 MB more (~5.2 GB needed). A/B/C are fitted to this case, i.e. to a dense init cloud
    whose large early Gaussians produce many tile intersections.
  - 300k Gaussians at 640x360, f=1 peaked at 0.47 GB allocated; the model says ~1.9 GB, so
    it is ~2x pessimistic for well-converged scenes. That is deliberate - an OOM costs a run.
Resolution is kept (f=1) whenever a minimal cap still fits; f is raised to 2 then 4 only when
it doesn't. The cap also scales with how much supervision the scene provides (training
pixels), and short captures (<= SHORT_CAPTURE_FRAMES) are held to a few hundred thousand:
on the 40-frame drone scene 1M Gaussians gave 72% near-transparent, 92% needle-shaped
Gaussians, while 300k trained cleanly.

TRAINING_MAX_STEPS - "steps-per-image": ~STEPS_PER_IMAGE per training image, clamped.
TRAINING_EVAL_STEPS / TRAINING_SAVE_STEPS - [max/2, max] so the Governor always has val stats.
POSE_OPT_WARMUP - ~15% of max_steps.

Each derive recomputes from the same helpers (and honours a sibling the user set explicitly)
instead of reading a value a sibling strategy wrote, so module order doesn't matter.
Stdlib-only and py3.8-compatible.
"""

import math

from ..resolve import Decision
from . import Param

# --- VRAM model (MB) ---
MB_PER_GAUSSIAN = 0.0036   # params + Adam state + grads + projection/tile-intersection buffers
MB_PER_PIXEL = 0.0015      # render/loss/SSIM buffers per training pixel (batch size 1)
FIXED_OVERHEAD_MB = 500    # CUDA context, allocator slack, gsplat workspaces
SAFETY_MARGIN = 0.18       # fraction of free memory kept in reserve (other processes, spikes)

DATA_FACTORS = (1, 2, 4)   # the only image folders the pipeline creates (images, images_2, images_4)
MIN_CAP = 150_000          # below this, lowering resolution beats starving the model
ABS_MIN_CAP = 50_000       # floor when nothing fits even at the coarsest resolution
MAX_CAP = 1_000_000        # trainer default; never exceed it
CAP_STEP = 10_000

# --- content model ---
REF_CAP = 300_000              # worked well on the drone scene ...
REF_TRAIN_PIXELS = 8_000_000   # ... which had 35 train images x 640x360 ~= 8.1M pixels
SHORT_CAPTURE_FRAMES = 50
SHORT_CAPTURE_CAP = 300_000

# --- step budget ---
STEPS_PER_IMAGE = 150
MIN_STEPS, MAX_STEPS = 3000, 30000
WARMUP_FRACTION = 0.15


def _num(profile, key):
    v = profile.get(key) if profile is not None else None
    return v if isinstance(v, (int, float)) and v > 0 else None


def _user(config, name):
    """The user's explicit value for `name`, or None when it's on auto."""
    if name in getattr(config, "USER_SET", set()):
        return getattr(config, name, None)
    return None


def _n_train(profile, config):
    """Training image count: the dataset holds out every test_every-th image for val."""
    n = _num(profile, "frames.count")
    if n is None:
        return None
    n = int(n)
    test_every = getattr(config, "TEST_EVERY", 8) or 0
    n_test = math.ceil(n / test_every) if test_every > 0 else 0
    return max(1, n - n_test)


def estimate_peak_mb(n_gaussians, width, height, data_factor):
    """Modelled peak trainer VRAM (MB) for n_gaussians at (width/f)x(height/f)."""
    pixels = (width // data_factor) * (height // data_factor)
    return MB_PER_GAUSSIAN * n_gaussians + MB_PER_PIXEL * pixels + FIXED_OVERHEAD_MB


def _round_cap(n):
    return int(n // CAP_STEP * CAP_STEP)


def _content_cap(n_train, n_frames, width, height, f):
    """Gaussians the scene can usefully supervise: sqrt-scaled with training pixels."""
    pixels = n_train * (width // f) * (height // f)
    cap = REF_CAP * math.sqrt(pixels / REF_TRAIN_PIXELS)
    if n_frames <= SHORT_CAPTURE_FRAMES:
        cap = min(cap, SHORT_CAPTURE_CAP)
    return _round_cap(min(max(cap, MIN_CAP), MAX_CAP))


def _plan(profile, config):
    """(cap, data_factor, est_mb, free_mb, budget_mb, why) or None when signals are missing."""
    free = _num(profile, "gpu.free_mb")
    n_frames = _num(profile, "frames.count")
    width = _num(profile, "frames.width")
    height = _num(profile, "frames.height")
    if None in (free, n_frames, width, height):
        return None
    width, height, n_frames = int(width), int(height), int(n_frames)
    n_train = _n_train(profile, config)
    budget = free * (1.0 - SAFETY_MARGIN)

    user_cap = _user(config, "CAP_MAX")
    user_f = _user(config, "DATA_FACTOR")
    factors = [user_f] if user_f else list(DATA_FACTORS)

    def mem_cap(f):
        room = budget - MB_PER_PIXEL * (width // f) * (height // f) - FIXED_OVERHEAD_MB
        return room / MB_PER_GAUSSIAN

    if user_cap:
        # Cap fixed by the user: pick the finest resolution it fits at (else the coarsest,
        # which is the best we can do, and say so).
        why = "user cap"
        for f in factors:
            if estimate_peak_mb(user_cap, width, height, f) <= budget:
                break
        else:
            why = "user cap does NOT fit at any data_factor - OOM likely"
        return user_cap, f, estimate_peak_mb(user_cap, width, height, f), free, budget, why

    for f in factors:
        if mem_cap(f) >= MIN_CAP:
            break
    content = _content_cap(n_train, n_frames, width, height, f)
    fit = mem_cap(f)
    if fit >= content:
        cap, why = content, f"content cap ({n_train} train imgs)"
    elif fit >= MIN_CAP:
        cap, why = _round_cap(fit), "VRAM-limited"
    else:
        cap, why = max(_round_cap(fit), ABS_MIN_CAP), "VRAM-limited, below minimal cap"
    return cap, f, estimate_peak_mb(cap, width, height, f), free, budget, why


def _plan_reason(plan):
    cap, f, est, free, budget, why = plan
    return (f"free {free:.0f} MB (budget {budget:.0f} MB after {SAFETY_MARGIN:.0%} margin), "
            f"est {est / 1024:.1f} GB at cap {cap // 1000}k, f={f}; {why}")


def derive_cap_max(profile, config):
    plan = _plan(profile, config)
    if plan is None:
        return None
    return Decision(plan[0], "vram-budget", _plan_reason(plan))


def derive_data_factor(profile, config):
    plan = _plan(profile, config)
    if plan is None:
        return None
    return Decision(plan[1], "vram-budget", _plan_reason(plan))


def _max_steps(profile, config):
    """(max_steps, reason) - the user's value if set, else steps-per-image; None if unknown."""
    user = _user(config, "TRAINING_MAX_STEPS")
    if user:
        return int(user), f"user max_steps {int(user)}"
    n_train = _n_train(profile, config)
    if n_train is None:
        return None
    steps = min(max(STEPS_PER_IMAGE * n_train, MIN_STEPS), MAX_STEPS)
    steps = int(steps / 100.0 + 0.5) * 100
    return steps, (f"{n_train} train imgs x {STEPS_PER_IMAGE} steps, "
                   f"clamped [{MIN_STEPS}, {MAX_STEPS}]")


def derive_max_steps(profile, config):
    ms = _max_steps(profile, config)
    if ms is None:
        return None
    return Decision(ms[0], "steps-per-image", ms[1])


def _checkpoints(profile, config):
    ms = _max_steps(profile, config)
    if ms is None:
        return None
    steps, why = ms
    half = int(steps / 200.0 + 0.5) * 100 if steps >= 2000 else max(1, steps // 2)
    points = sorted({half, steps})
    return Decision(points, "half-and-end", f"[max/2, max] of max_steps {steps} ({why})")


def derive_warmup(profile, config):
    ms = _max_steps(profile, config)
    if ms is None:
        return None
    steps, why = ms
    warmup = int(steps * WARMUP_FRACTION / 10.0 + 0.5) * 10
    return Decision(warmup, "fraction-of-steps",
                    f"{WARMUP_FRACTION:.0%} of max_steps {steps} ({why})")


PARAMS = [
    Param("CAP_MAX", "pre_train", derive_cap_max),
    Param("DATA_FACTOR", "pre_train", derive_data_factor),
    Param("TRAINING_MAX_STEPS", "pre_train", derive_max_steps),
    Param("TRAINING_EVAL_STEPS", "pre_train", _checkpoints),
    Param("TRAINING_SAVE_STEPS", "pre_train", _checkpoints),
    Param("POSE_OPT_WARMUP", "pre_train", derive_warmup),
]
