"""ACE-Zero schedule: continuous functions of the measured scene instead of lookup tables.

Previously ACEZeroPoseEstimator bucketed avg_motion (>30, >15) and quality_mode into step
tables. Here every value is a smooth function of the signals ACE-Zero actually depends on:

    avg_motion   mean frame difference (0-255)   -> reprojection clamp, registration
                                                    confidence, seeds, rotation augmentation
    num_frames   extracted frames                 -> mapping iterations, seeds, data workers
    frame size   extracted frame short side       -> image_resolution (never upscaled)
    cpu_count, free VRAM                          -> worker counts, training buffer placement

quality_mode stays a preset: it picks the base of each schedule, which the signals then
scale. Every output is clamped to the envelope spanned by the old tables so behaviour stays
within what has been exercised before.

Also provides the robust pose-filter threshold (MIN_REGISTRATION_CONFIDENCE) and the single
ACE-Zero argument builder used by both the native and Docker command paths.

Pure, stdlib-only and py3.8-compatible (imported by the strategy auto-discovery).
"""

import math
from typing import Dict, List, Optional

from ..resolve import Decision

# ---------------------------------------------------------------------------
# Presets (old per-mode tables) and the envelope the old tables spanned
# ---------------------------------------------------------------------------
PRESETS = {
    "fast": dict(seed_iterations=3000, refit_iterations=10000, learning_rate_max=0.006,
                 cooldown_threshold=0.75, cooldown_iterations=2000, aug_rotation=5,
                 registration_threshold=0.95, hybrid_train_iterations=8000,
                 hybrid_pose_wait=1000),
    "balanced": dict(seed_iterations=5000, refit_iterations=18000, learning_rate_max=0.004,
                     cooldown_threshold=0.7, cooldown_iterations=4000, aug_rotation=10,
                     registration_threshold=0.99, hybrid_train_iterations=15000,
                     hybrid_pose_wait=2000),
    "quality": dict(seed_iterations=10000, refit_iterations=50000, learning_rate_max=0.003,
                    cooldown_threshold=0.6, cooldown_iterations=5000, aug_rotation=15,
                    registration_threshold=0.99, hybrid_train_iterations=25000,
                    hybrid_pose_wait=3000),
}

ENVELOPE = {
    "repro_loss_soft_clamp": (30, 50),
    "registration_confidence": (500, 1500),
    "try_seeds": (4, 12),
    "iterations_max": (5, 15),
    "seed_iterations": (3000, 10000),
    "refit_iterations": (10000, 50000),
    "hybrid_train_iterations": (8000, 25000),
    "hybrid_pose_wait": (1000, 3000),
    "learning_rate_max": (0.003, 0.006),
    "cooldown_iterations": (2000, 5000),
    "aug_rotation": (5, 15),
    "num_data_workers": (4, 16),
    "image_resolution": (120, 480),
}

# avg_motion range over which the old buckets switched (<=15 calm ... >30 shaky); the
# smooth ramp spans slightly beyond so both old thresholds sit inside it.
MOTION_LO, MOTION_HI = 10.0, 35.0
# Frame count at which the preset iteration counts apply unchanged.
REFERENCE_FRAMES = 100
FRAME_SCALE_RANGE = (0.7, 1.5)
# ACE training buffer: 10 dataset passes x 1024 samples/image, capped at 8M samples;
# ~1230 bytes/sample (512 fp16 features + poses/intrinsics/targets).
ACE_BUFFER_PASSES, ACE_SAMPLES_PER_IMAGE, ACE_BUFFER_MAX = 10, 1024, 8000000
ACE_BYTES_PER_SAMPLE = 1230
# VRAM ACE-Zero needs besides the buffer (network, depth model, activations).
ACE_VRAM_RESERVE_MB = 2500
# Rough VRAM per concurrent seed-mapping process.
SEED_WORKER_MB = 700
# Robust pose filter: drop only values below median - k * 1.4826 * MAD, and never
# more than MAX_DROP_FRACTION of the frames.
ROBUST_K = 3.0
MAX_DROP_FRACTION = 0.2
DEFAULT_MIN_CONFIDENCE = 1000
DEFAULT_IMAGE_RESOLUTION = 480


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _env(name, x):
    lo, hi = ENVELOPE[name]
    return _clamp(x, lo, hi)


def _round_to(x, step):
    return int(step * round(float(x) / step))


def preset(quality_mode: str) -> dict:
    return PRESETS.get(quality_mode, PRESETS["balanced"])


def motion_level(avg_motion: Optional[float]) -> float:
    """0 (static/calm) .. 1 (shaky) linear ramp of mean frame difference; 15 -> 0.2, 30 -> 0.8."""
    if avg_motion is None:
        avg_motion = 15.0  # old _analyze_video default
    return _clamp((float(avg_motion) - MOTION_LO) / (MOTION_HI - MOTION_LO), 0.0, 1.0)


def frame_scale(num_frames: int) -> float:
    """Mapping-iteration multiplier: sqrt(frames / 100), clamped to [0.7, 1.5]."""
    return _clamp(math.sqrt(max(1, num_frames) / float(REFERENCE_FRAMES)), *FRAME_SCALE_RANGE)


# ---------------------------------------------------------------------------
# Individual schedules (each returns a Decision)
# ---------------------------------------------------------------------------
def soft_clamp(avg_motion) -> Decision:
    """Tighter tanh reprojection clamp for shaky footage (more outlier correspondences)."""
    m = motion_level(avg_motion)
    value = int(round(_env("repro_loss_soft_clamp", 50 - 20 * m)))
    return Decision(value, "motion-ramp", f"avg_motion={_fmt(avg_motion)} -> level {m:.2f}")


def registration_confidence(avg_motion) -> Decision:
    """ACE-internal inlier-count threshold for registering a frame; stricter under motion."""
    m = motion_level(avg_motion)
    value = _round_to(_env("registration_confidence", 500 + 1000 * m), 10)
    return Decision(value, "motion-ramp", f"avg_motion={_fmt(avg_motion)} -> level {m:.2f}")


def try_seeds(avg_motion, num_frames, quality_mode) -> Decision:
    """Seed attempts grow with motion (harder init) and sqrt(frames); fast mode = single shot."""
    if quality_mode == "fast":
        return Decision(1, "preset", "fast mode trusts the first seed")
    m = motion_level(avg_motion)
    value = int(round(_env("try_seeds", 4 + 3 * m + math.sqrt(max(1, num_frames) / 30.0))))
    return Decision(value, "motion+frames", f"motion level {m:.2f}, {num_frames} frames")


def iterations_max(num_frames) -> Decision:
    """Registration rounds: 5 + log2(frames) in [5, 15]."""
    value = int(_env("iterations_max", int(5 + math.log2(max(1, num_frames)))))
    return Decision(value, "log-frames", f"{num_frames} frames")


def mapping_iterations(num_frames, quality_mode) -> Dict[str, Decision]:
    """Seed/refit/hybrid iterations: preset x sqrt(frames/100), with pose-wait, cooldown and
    peak LR following the resulting length (shorter 1-cycle schedules need a higher peak)."""
    p = preset(quality_mode)
    s = frame_scale(num_frames)
    why = f"{quality_mode} preset x frame scale {s:.2f} ({num_frames} frames)"
    out = {}
    for key in ("seed_iterations", "refit_iterations", "hybrid_train_iterations"):
        out[key] = Decision(_round_to(_env(key, p[key] * s), 500), "frames-scaled", why)
    train = out["hybrid_train_iterations"].value
    out["hybrid_pose_wait"] = Decision(
        _round_to(_env("hybrid_pose_wait", train * p["hybrid_pose_wait"] / p["hybrid_train_iterations"]), 100),
        "ratio", f"same fraction of {train} train iterations as the preset")
    out["cooldown_iterations"] = Decision(
        _round_to(_env("cooldown_iterations", train * p["cooldown_iterations"] / p["hybrid_train_iterations"]), 100),
        "ratio", f"same fraction of {train} train iterations as the preset")
    lr = _env("learning_rate_max", p["learning_rate_max"] * math.sqrt(p["hybrid_train_iterations"] / float(train)))
    out["learning_rate_max"] = Decision(round(lr, 5), "sqrt-length",
                                        f"preset LR x sqrt(preset/actual iterations), {train} iterations")
    return out


def aug_rotation(avg_motion, quality_mode) -> Decision:
    """Rotation augmentation: preset x (0.75 .. 1.25) with motion (handheld roll)."""
    m = motion_level(avg_motion)
    value = int(round(_env("aug_rotation", preset(quality_mode)["aug_rotation"] * (0.75 + 0.5 * m))))
    return Decision(value, "preset x motion", f"{quality_mode} preset, motion level {m:.2f}")


def image_resolution(frame_width, frame_height) -> Decision:
    """ACE short-side resolution: the frame short side, capped at 480 (never upscale)."""
    if not frame_width or not frame_height:
        return Decision(DEFAULT_IMAGE_RESOLUTION, "fallback", "frame size unknown")
    short = min(int(frame_width), int(frame_height))
    value = int(_env("image_resolution", short))
    return Decision(value, "frame-short-side", f"frames {frame_width}x{frame_height}, short side {short}")


def ace_buffer_mb(num_frames) -> float:
    samples = min(ACE_BUFFER_PASSES * max(1, num_frames) * ACE_SAMPLES_PER_IMAGE, ACE_BUFFER_MAX)
    return samples * ACE_BYTES_PER_SAMPLE / 2.0 ** 20


def num_data_workers(num_frames, cpu_count) -> Decision:
    """4 + frames/50 in [4, 16], leaving two cores free on small machines."""
    value = int(_env("num_data_workers", int(4 + max(1, num_frames) / 50)))
    if cpu_count:
        value = max(1, min(value, int(cpu_count) - 2))
    return Decision(value, "frames+cpu", f"{num_frames} frames, cpu_count={cpu_count}")


def seed_parallel_workers(seeds, cpu_count, free_mb, quality_mode) -> Decision:
    """Concurrent seed mappers: <= seeds, <= cpu/3, <= free VRAM / 700MB, <= 8."""
    if quality_mode == "fast":
        return Decision(1, "preset", "fast mode runs a single seed")
    value = min(8, max(1, seeds))
    if cpu_count:
        value = min(value, max(1, int(cpu_count) // 3))
    if free_mb:
        value = min(value, max(1, int(free_mb // SEED_WORKER_MB)))
    return Decision(value, "cpu+vram", f"{seeds} seeds, cpu_count={cpu_count}, free_mb={_fmt(free_mb)}")


def training_buffer_cpu(num_frames, free_mb, quality_mode) -> Decision:
    """Keep the ACE training buffer on the GPU unless it would not fit beside the reserve."""
    if quality_mode == "fast":
        return Decision(True, "preset", "fast/streaming mode leaves VRAM to 3DGS")
    need = ace_buffer_mb(num_frames)
    if not free_mb:
        return Decision(False, "fallback", f"free VRAM unknown; buffer ~{need:.0f}MB")
    on_cpu = need > float(free_mb) - ACE_VRAM_RESERVE_MB
    return Decision(on_cpu, "vram-budget",
                    f"buffer ~{need:.0f}MB for {num_frames} frames vs {free_mb:.0f}MB free "
                    f"- {ACE_VRAM_RESERVE_MB}MB reserve")


def num_head_blocks(num_frames, free_mb, buffer_on_cpu) -> Decision:
    """A second head block (more map capacity) for large scenes when VRAM headroom allows."""
    if num_frames < 150:
        return Decision(1, "frames", f"{num_frames} frames < 150")
    if free_mb:
        spare = float(free_mb) - ACE_VRAM_RESERVE_MB - (0 if buffer_on_cpu else ace_buffer_mb(num_frames))
        if spare < 500:
            return Decision(1, "vram-budget", f"{num_frames} frames but only {spare:.0f}MB spare VRAM")
    return Decision(2, "frames+vram", f"{num_frames} frames >= 150, free_mb={_fmt(free_mb)}")


def compute_schedule(avg_motion, num_frames, frame_width=None, frame_height=None,
                     quality_mode="balanced", cpu_count=None, free_mb=None) -> Dict[str, Decision]:
    """All ACE-Zero parameters for a run, keyed by the ACE-Zero option name."""
    num_frames = max(1, int(num_frames or 1))
    out = {
        "repro_loss_soft_clamp": soft_clamp(avg_motion),
        "registration_confidence": registration_confidence(avg_motion),
        "try_seeds": try_seeds(avg_motion, num_frames, quality_mode),
        "iterations_max": iterations_max(num_frames),
        "aug_rotation": aug_rotation(avg_motion, quality_mode),
        "image_resolution": image_resolution(frame_width, frame_height),
        "num_data_workers": num_data_workers(num_frames, cpu_count),
    }
    out.update(mapping_iterations(num_frames, quality_mode))
    p = preset(quality_mode)
    out["cooldown_threshold"] = Decision(p["cooldown_threshold"], "preset", quality_mode)
    out["registration_threshold"] = Decision(p["registration_threshold"], "preset", quality_mode)
    out["seed_parallel_workers"] = seed_parallel_workers(out["try_seeds"].value, cpu_count, free_mb, quality_mode)
    out["training_buffer_cpu"] = training_buffer_cpu(num_frames, free_mb, quality_mode)
    out["num_head_blocks"] = num_head_blocks(num_frames, free_mb, out["training_buffer_cpu"].value)
    return out


# ---------------------------------------------------------------------------
# Command-line builder (native and Docker paths share it)
# ---------------------------------------------------------------------------
HYBRID_ARGS = ("hybrid_train_iterations", "hybrid_pose_wait", "learning_rate_max",
               "training_buffer_cpu", "num_data_workers", "num_head_blocks", "refinement",
               "refinement_ortho", "pose_refinement_lr", "cooldown_iterations",
               "cooldown_threshold", "aug_rotation", "image_resolution",
               "registration_confidence", "repro_loss_soft_clamp")
ITERATIVE_ARGS = ("iterations_max", "seed_iterations", "refit_iterations", "learning_rate_max",
                  "try_seeds", "seed_parallel_workers", "training_buffer_cpu", "num_data_workers",
                  "num_head_blocks", "refinement", "refinement_ortho", "pose_refinement_wait",
                  "pose_refinement_lr", "cooldown_iterations", "cooldown_threshold",
                  "aug_rotation", "registration_threshold", "image_resolution",
                  "registration_confidence", "repro_loss_soft_clamp")


def build_ace_args(values: dict, hybrid: bool) -> List[str]:
    """['--name', 'value', ...] for ace_zero_hybrid.py (hybrid) or ace_zero.py.

    Raises KeyError for a missing value instead of silently substituting a default.
    """
    args = []
    for name in (HYBRID_ARGS if hybrid else ITERATIVE_ARGS):
        args += ["--" + name, str(values[name])]
    return args


# ---------------------------------------------------------------------------
# Pose filter: robust percentile threshold
# ---------------------------------------------------------------------------
def _median(values):
    v = sorted(values)
    n = len(v)
    mid = n // 2
    return v[mid] if n % 2 else 0.5 * (v[mid - 1] + v[mid])


def robust_min_confidence(confs, base_threshold=DEFAULT_MIN_CONFIDENCE) -> Decision:
    """Effective pose-filter threshold: min(base, median - k*MAD), never dropping > 20%.

    The base threshold is an absolute inlier count; on scenes where ACE-Zero's confidence is
    low across the board it would discard good frames, so it is lowered to the scene's own
    outlier boundary. It is never raised: consistently high-confidence scenes (drone,
    MAD ~16) keep every frame.
    """
    confs = sorted(float(c) for c in confs)
    if not confs:
        return Decision(base_threshold, "fallback", "no ACE-Zero confidences")
    med = _median(confs)
    mad = 1.4826 * _median([abs(c - med) for c in confs])
    outlier = med - ROBUST_K * mad
    threshold = min(float(base_threshold), outlier)
    # Cap the drop: the threshold may not exceed the MAX_DROP_FRACTION quantile.
    max_drop = int(MAX_DROP_FRACTION * len(confs))
    cap = confs[max_drop]
    capped = threshold > cap
    threshold = min(threshold, cap)
    value = int(math.floor(threshold))
    dropped = sum(1 for c in confs if c < value)
    reason = (f"min(base {base_threshold}, median {med:.0f} - {ROBUST_K:g}*MAD {mad:.0f} = {outlier:.0f})"
              f"{', capped at 20% drop' if capped else ''}; drops {dropped}/{len(confs)}")
    return Decision(value, "robust-percentile", reason)


def _derive_min_conf(profile, config):
    """post_ace: keep config (Difix confidence weighting) in line with the filter used."""
    used = profile.get("ace.min_conf_used") if profile is not None else None
    if used is not None:
        return Decision(used, "robust-percentile", "threshold used by the ACE-Zero pose filter")
    conf = profile.get("ace.conf") if profile is not None else None
    if not conf:
        return None
    base = getattr(config, "MIN_REGISTRATION_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
    return robust_min_confidence(conf.values(), base)


def _fmt(x):
    return "n/a" if x is None else f"{float(x):.1f}"


# Auto-discovery (strategies/__init__.py)
NAME = "MIN_REGISTRATION_CONFIDENCE"
STAGE = "post_ace"
derive = _derive_min_conf
