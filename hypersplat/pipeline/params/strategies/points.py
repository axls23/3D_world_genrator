"""Initial point cloud size: a GPU-memory point budget instead of a fixed 2% of all pixels.

generate_points_wsl.py unprojects monocular depth for every registered frame and keeps a random
fraction of the pixels as the 3DGS starting cloud. A fixed 2% scales with W*H*N: the drone
(40 x 640x360) gave 184,610 points, the WhatsApp clip (139 x 480x864) gave 1,153,899 and OOM'd
a 6 GB GPU. Instead we pick a target point count from free GPU memory and derive the fraction
from it, so the cloud size tracks what the trainer can hold (roughly the CAP_MAX ballpark; the
trainer still enforces its cap).

Helpers only (no NAME/STAGE/derive): point generation runs inside the ACE-Zero stage, in the
separate py3.8 env, before any strategy stage fires. Stdlib-only and py3.8-compatible.
"""

import os
from pathlib import Path
from typing import Optional

from ..resolve import Decision, resolve

PARAM_NAME = "INIT_POINT_SUBSAMPLE"
DEFAULT_SUBSAMPLE = 0.02        # previous hard-coded fraction; fallback without a profile
MAX_SUBSAMPLE = 0.05            # denser than 1 point per ~20 px only duplicates neighbouring depth
MIN_POINTS, MAX_POINTS = 50000, 1000000
# Calibrated on the RTX 4050 Laptop (6 GB): ~5.5 GB free -> ~300k points, which trained at
# 0.47 GB peak on the drone scene and leaves headroom for densification up to the cap.
POINTS_PER_FREE_MB = 300000 / 5500.0


def target_points(profile) -> Optional[Decision]:
    """Point budget from GPU memory (free, else total), clamped to [MIN_POINTS, MAX_POINTS]."""
    if profile is None:
        return None
    free_mb = profile.get("gpu.free_mb")
    total_mb = profile.get("gpu.total_mb")
    mem_mb, which = (free_mb, "free") if free_mb else (total_mb, "total")
    if not mem_mb or mem_mb <= 0:
        return None
    target = int(min(MAX_POINTS, max(MIN_POINTS, mem_mb * POINTS_PER_FREE_MB)))
    return Decision(target, "point-budget", f"{mem_mb:.0f} MB GPU {which}")


def derive_subsample(profile, width: int, height: int, n_frames: int) -> Optional[Decision]:
    """Fraction of the W*H*N unprojected pixels to keep so the cloud hits the point budget."""
    budget = target_points(profile)
    pixels = int(width) * int(height) * int(n_frames)
    if budget is None or pixels <= 0:
        return None
    fraction = min(MAX_SUBSAMPLE, budget.value / float(pixels))
    capped = " (density-capped)" if fraction >= MAX_SUBSAMPLE else ""
    return Decision(round(fraction, 6), "point-budget",
                    f"target {budget.value} pts / ({width}x{height} x {n_frames} frames = "
                    f"{pixels} px){capped}; {budget.reason}")


def resolve_subsample(user_value: Optional[float], profile, width: int, height: int,
                      n_frames: int) -> float:
    """Explicit --subsample > point budget > DEFAULT_SUBSAMPLE; logs and records the decision."""
    return resolve(PARAM_NAME, user_value,
                   lambda: derive_subsample(profile, width, height, n_frames),
                   DEFAULT_SUBSAMPLE, profile)


def pick_poses_file(poses_file: os.PathLike) -> Path:
    """Prefer the confidence-filtered poses written by the pipeline's pose filter.

    poses_final_filtered.txt is only (re)written when some poses were dropped, so a copy older
    than poses_final.txt is stale from a previous ACE-Zero run and ignored.
    """
    poses_file = Path(poses_file)
    filtered = poses_file.with_name("poses_final_filtered.txt")
    if poses_file.name == "poses_final.txt" and filtered.exists():
        if not poses_file.exists() or filtered.stat().st_mtime >= poses_file.stat().st_mtime:
            return filtered
    return poses_file
