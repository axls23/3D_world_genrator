"""Frame extraction: sampling rate (FPS) and extracted-frame resolution.

FPS used to be a constant (10 in the manager, 2 in perception, 1.5 in the API) regardless of
clip length: a 20 s drone clip at 2 fps gave 40 frames (too sparse -> needle artifacts), while
a 9 s handheld clip at 15 fps gave 139 frames (good). Here the rate is chosen so the clip
yields a target frame count, scaled by how fast the camera/scene moves.

The extraction width used to be a fixed 640 px. ACE-Zero resizes internally (480 px short
side) and the trainer's data_factor sets training resolution, so extraction only needs to not
throw detail away: keep the source resolution up to a VRAM-derived long-side budget.

Stdlib-only and py3.8-compatible.
"""

from typing import Optional

from ..resolve import Decision, resolve

NAME = "FPS"
STAGE = "pre_ace"

# Frame-count targets. ~150 frames gave good coverage on the WhatsApp clip (139); ACE-Zero cost
# grows ~linearly with frames, so cap the total to keep it tractable.
TARGET_FRAMES = 150
MAX_FRAMES = 300
MIN_FPS = 0.5
# Motion signal (video.avg_motion: mean abs frame difference, 160x90 gray) of a "typical" clip;
# drone flyover = 19.2, handheld WhatsApp = 28.3. Faster motion needs denser sampling for
# parallax overlap between neighbouring frames; slower motion needs fewer frames.
MOTION_REF = 20.0
MOTION_SCALE_RANGE = (0.6, 1.5)

# Extraction long-side budgets by total VRAM (MB): (min total_mb, long side px)
LONG_SIDE_FALLBACK = 640
LONG_SIDE_BUDGETS = ((11000, 1280), (5000, 960), (0, 640))


def derive(profile, config) -> Optional[Decision]:
    """FPS from video.duration / video.fps / video.avg_motion ("target-frames-parallax")."""
    if profile is None:
        return None
    duration = profile.get("video.duration")
    if not duration or duration <= 0:
        return None
    src_fps = profile.get("video.fps") or 0.0
    motion = profile.get("video.avg_motion")

    scale = 1.0
    if motion is not None and motion > 0:
        lo, hi = MOTION_SCALE_RANGE
        scale = min(hi, max(lo, (motion / MOTION_REF) ** 0.5))
    target = min(MAX_FRAMES, TARGET_FRAMES * scale)

    fps = max(MIN_FPS, target / duration)
    if src_fps > 0:
        fps = min(fps, src_fps)
    capped = fps * duration > MAX_FRAMES  # very long clips: frame cap beats the fps floor
    if capped:
        fps = MAX_FRAMES / duration
    fps = round(fps, 2)
    fps = max(fps, 0.01)

    motion_txt = f"avg_motion={motion:.1f}" if motion is not None else "avg_motion=n/a"
    reason = (f"target {target:.0f} frames ({motion_txt} -> x{scale:.2f}) over "
              f"{duration:.1f}s, src {src_fps:.2f} fps -> ~{fps * duration:.0f} frames")
    if capped:
        reason += f" (capped at {MAX_FRAMES})"
    return Decision(fps, "target-frames-parallax", reason)


def derive_long_side(profile) -> Optional[Decision]:
    """Max long side of extracted frames from gpu.total_mb, never above the source."""
    if profile is None:
        return None
    total_mb = profile.get("gpu.total_mb")
    if not total_mb:
        return None
    budget = next(px for mb, px in LONG_SIDE_BUDGETS if total_mb >= mb)
    src_long = max(profile.get("video.width") or 0, profile.get("video.height") or 0)
    value = min(budget, src_long) if src_long > 0 else budget
    value -= value % 2
    src_txt = f"source long side {src_long}px" if src_long else "source size n/a"
    return Decision(value, "vram-pixel-budget",
                    f"gpu.total_mb={total_mb} -> budget {budget}px, {src_txt}, never upscale")


def max_long_side(profile) -> int:
    """Resolve and log EXTRACT_LONG_SIDE (fallback 640 when there is no profile/GPU signal)."""
    return resolve("EXTRACT_LONG_SIDE", None, lambda: derive_long_side(profile),
                   LONG_SIDE_FALLBACK, profile)


def scale_filter(long_side: int) -> str:
    """ffmpeg scale filter: long side <= long_side, never upscale, even dimensions."""
    L = int(long_side)
    w = f"if(gte(iw,ih),trunc(min({L},iw)/2)*2,-2)"
    h = f"if(gte(iw,ih),-2,trunc(min({L},ih)/2)*2)"
    return f"scale='{w}':'{h}'"


def fitted_size(width: int, height: int, long_side: int):
    """(w, h) that scale_filter produces for a width x height source (for tests/logging)."""
    if width >= height:
        w = (min(long_side, width) // 2) * 2
        h = int(round(height * w / width / 2.0)) * 2
    else:
        h = (min(long_side, height) // 2) * 2
        w = int(round(width * h / height / 2.0)) * 2
    return w, h
