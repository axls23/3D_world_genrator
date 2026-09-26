"""Camera intrinsics: measure, then refine.

1. Initial focal (FOCAL_INIT_PX), in extracted-frame pixels:
   - "metadata-35mm": the video carries a 35mm-equivalent focal length (e.g. Apple
     `com.apple.quicktime.camera.focal_length.35mmEquivalent`, EXIF-style
     `FocalLengthIn35mmFormat`) -> f_px = f35 / 36mm * long_side_px
     (36 mm is the long side of a 35mm frame, so this holds for portrait and landscape).
   - "hfov-70": otherwise ONE shared heuristic, 70 degree FOV across the long side:
     f_px = max(W, H) / (2 tan 35deg) (the long side, so portrait phone clips are not
     given a focal ~40% too short). ACE-Zero's own hybrid init receives this value as a hint
     instead of applying its separate 70%-of-diagonal rule.
2. Refined focal (FOCAL_PX): ACE-Zero optimises the focal during mapping
   (refine_calibration); the median of its per-frame focals is what the COLMAP export uses.
3. Stale cameras.bin: runs that reuse an existing sparse/0 (--colmap-input / skip-ACE) may
   carry a cameras.bin written from the initial guess. If its focal differs from ACE's by
   more than STALE_TOLERANCE it is rewritten (cameras.bin.bak keeps the original).

Principal point: stays at (W/2, H/2). ACE-Zero does not estimate it and common video
containers carry no principal-point metadata, so nothing here changes it; a rewrite of a
stale cameras.bin preserves whatever principal point the file already had.

No NAME/STAGE/derive on purpose: the focal is not a PipelineConfig attribute, so this module
is not auto-applied; perception.py and manager.py call these functions directly.
Stdlib-only and py3.8-compatible (also imported from the ACE-Zero py3.8 env).
"""

import json
import math
import shutil
import struct
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from ..resolve import Decision

HFOV_DEG = 70.0
STALE_TOLERANCE = 0.02  # relative focal mismatch that marks a cameras.bin as stale
FILM_LONG_SIDE_MM = 36.0

# Tag names (lower-cased) that carry a 35mm-equivalent focal length, in format or stream tags
FOCAL_35MM_TAGS = (
    "com.apple.quicktime.camera.focal_length.35mmequivalent",
    "focallengthin35mmformat",
    "focallengthin35mmfilm",
    "focal_length_35mm",
    "focal_length_35mm_equivalent",
)

# COLMAP camera model id -> (name, number of params, number of focal params)
COLMAP_MODELS = {
    0: ("SIMPLE_PINHOLE", 3, 1),
    1: ("PINHOLE", 4, 2),
    2: ("SIMPLE_RADIAL", 4, 1),
    3: ("RADIAL", 5, 1),
    4: ("OPENCV", 8, 2),
    5: ("OPENCV_FISHEYE", 8, 2),
    6: ("FULL_OPENCV", 12, 2),
    7: ("FOV", 5, 2),
    8: ("SIMPLE_RADIAL_FISHEYE", 4, 1),
    9: ("RADIAL_FISHEYE", 5, 1),
    10: ("THIN_PRISM_FISHEYE", 12, 2),
}


# ----------------------------------------------------------------------------
# Initial focal: metadata, else the shared 70-degree-HFOV heuristic
# ----------------------------------------------------------------------------
def heuristic_focal_px(width: float, height: float = 0) -> float:
    """Focal (px) for a 70 degree FOV across the long image side."""
    return max(width, height) / (2.0 * math.tan(math.radians(HFOV_DEG / 2.0)))


def _parse_mm(value) -> Optional[float]:
    """'26', '26.0', '26 mm', 26 -> 26.0; anything non-positive or unparsable -> None."""
    try:
        mm = float(str(value).lower().replace("mm", "").strip())
    except (TypeError, ValueError):
        return None
    return mm if 0 < mm < 2000 else None


def parse_focal_35mm(ffprobe_info: dict) -> Optional[float]:
    """35mm-equivalent focal length (mm) from `ffprobe -show_format -show_streams` JSON."""
    tag_dicts = [(ffprobe_info.get("format") or {}).get("tags") or {}]
    for stream in ffprobe_info.get("streams") or []:
        if stream.get("codec_type") in (None, "video"):
            tag_dicts.append(stream.get("tags") or {})
    for tags in tag_dicts:
        for key, value in tags.items():
            if key.lower() in FOCAL_35MM_TAGS:
                mm = _parse_mm(value)
                if mm is not None:
                    return mm
    return None


def probe_focal_35mm(video_path) -> Optional[float]:
    """Run ffprobe on the video and return its 35mm-equivalent focal (mm), if tagged."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
             str(video_path)], capture_output=True, text=True, timeout=30).stdout
        return parse_focal_35mm(json.loads(out or "{}"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def focal_35mm_to_px(f35_mm: float, width: float, height: float) -> float:
    """35mm-equivalent focal (mm) -> px for a width x height image (orientation-agnostic)."""
    return f35_mm / FILM_LONG_SIDE_MM * max(width, height)


def initial_focal(width: int, height: int, f35_mm: Optional[float] = None) -> Decision:
    """Initial focal for extracted frames of width x height (px)."""
    if f35_mm:
        return Decision(round(focal_35mm_to_px(f35_mm, width, height), 2), "metadata-35mm",
                        f"video tag {f35_mm:g}mm 35mm-equiv -> {f35_mm:g}/36*{max(width, height)}px")
    return Decision(round(heuristic_focal_px(width, height), 2), "hfov-70",
                    f"no focal metadata; 70deg FOV across the {max(width, height)}px long side")


def refined_focal(focals: List[float], initial_px: float) -> Optional[Decision]:
    """Median of ACE-Zero's per-frame refined focals, or None when there are none."""
    if not focals:
        return None
    values = sorted(focals)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
    return Decision(round(median, 2), "ace-refined",
                    f"median of {len(values)} ACE-Zero focals; initial guess {initial_px:.2f}px")


# ----------------------------------------------------------------------------
# Stale cameras.bin check
# ----------------------------------------------------------------------------
def read_colmap_cameras(cameras_bin) -> List[Tuple[int, int, int, int, List[float]]]:
    """[(camera_id, model_id, width, height, params)] from a COLMAP cameras.bin.

    Layout: uint64 num_cameras, then per camera int32 id, int32 model, uint64 width,
    uint64 height, float64 params[n(model)]. Raises ValueError on unknown models/truncation.
    """
    data = Path(cameras_bin).read_bytes()
    (n,) = struct.unpack_from("<Q", data, 0)
    offset, cams = 8, []
    for _ in range(n):
        cam_id, model, w, h = struct.unpack_from("<iiQQ", data, offset)
        offset += 24
        if model not in COLMAP_MODELS:
            raise ValueError(f"unknown COLMAP camera model {model}")
        n_params = COLMAP_MODELS[model][1]
        params = list(struct.unpack_from(f"<{n_params}d", data, offset))
        offset += 8 * n_params
        cams.append((cam_id, model, w, h, params))
    return cams


def write_colmap_cameras(cameras_bin, cams) -> None:
    with open(cameras_bin, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cam_id, model, w, h, params in cams:
            f.write(struct.pack("<iiQQ", cam_id, model, w, h))
            f.write(struct.pack(f"<{len(params)}d", *params))


def is_stale(camera_focal: float, ace_focal: float, tolerance: float = STALE_TOLERANCE) -> bool:
    """True when a cameras.bin focal differs from the ACE-Zero focal by more than tolerance."""
    if not ace_focal or ace_focal <= 0:
        return False
    return abs(camera_focal - ace_focal) / ace_focal > tolerance


def fix_stale_cameras(cameras_bin, ace_focal: float, ace_width: Optional[int] = None,
                      tolerance: float = STALE_TOLERANCE) -> Optional[Decision]:
    """Rewrite cameras.bin with the ACE-Zero focal when its focal is stale.

    ace_width: width (px) of the images ACE-Zero ran on; the ACE focal is rescaled to each
    camera's width when they differ. Backs up the original to cameras.bin.bak (only when no
    backup exists yet). Returns the Decision applied, or None when nothing was changed.
    """
    cameras_bin = Path(cameras_bin)
    if not cameras_bin.exists() or not ace_focal or ace_focal <= 0:
        return None
    cams = read_colmap_cameras(cameras_bin)
    fixed, old = [], []
    for cam_id, model, w, h, params in cams:
        target = ace_focal * (w / ace_width) if ace_width else ace_focal
        if is_stale(params[0], target, tolerance):
            old.append(params[0])
            params = list(params)
            for i in range(COLMAP_MODELS[model][2]):
                params[i] = target
        fixed.append((cam_id, model, w, h, params))
    if not old:
        return None
    backup = cameras_bin.with_name(cameras_bin.name + ".bak")
    if not backup.exists():
        shutil.copy2(str(cameras_bin), str(backup))
    write_colmap_cameras(cameras_bin, fixed)
    new = fixed[0][4][0]
    return Decision(round(new, 2), "stale-rewrite",
                    f"cameras.bin focal {old[0]:.2f}px vs ACE-Zero {new:.2f}px "
                    f"(>{tolerance:.0%} off); original kept at {backup.name}")
