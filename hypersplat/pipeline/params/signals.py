"""Lightweight signal readers used to fill the SceneProfile.

Stdlib-only at import time and py3.8-compatible; optional deps (torch, cv2) are imported
lazily inside the functions that need them.
"""

import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_pose_file(data_dir: Path) -> Optional[Path]:
    """ACE-Zero poses_final.txt at the dataset root, in acezero_output/, or anywhere below."""
    data_dir = Path(data_dir)
    for cand in (data_dir / "poses_final.txt", data_dir / "acezero_output" / "poses_final.txt"):
        if cand.exists():
            return cand
    found = sorted(data_dir.rglob("poses_final.txt"))
    return found[0] if found else None


def read_ace_poses(data_dir: Path) -> Optional[dict]:
    """Per-frame ACE-Zero focal and confidence from poses_final.txt.

    Line format: <image_path> qw qx qy qz tx ty tz <focal> <confidence>
    Returns {"focal": {name: f}, "conf": {name: c}, "focal_median": f, "path": str} or None.
    """
    pose_file = find_pose_file(data_dir)
    if pose_file is None:
        return None
    focal, conf = {}, {}
    for line in pose_file.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 10:
            name = Path(parts[0]).name
            focal[name] = float(parts[8])
            conf[name] = float(parts[9])
    if not focal:
        return None
    values = sorted(focal.values())
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
    return {"focal": focal, "conf": conf, "focal_median": median, "path": str(pose_file)}


def count_points3d(points_bin: Path) -> Optional[int]:
    """Number of points in a COLMAP points3D.bin (the leading uint64)."""
    points_bin = Path(points_bin)
    if not points_bin.exists() or points_bin.stat().st_size < 8:
        return None
    with open(points_bin, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def image_folder_stats(image_dir: Path) -> Optional[Tuple[int, int, int]]:
    """(count, width, height) of the images in a folder, or None."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return None
    files = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith("pseudo_"))
    if not files:
        return None
    try:
        from PIL import Image
        with Image.open(files[0]) as im:
            w, h = im.size
    except Exception:
        return None
    return len(files), w, h


def gpu_memory() -> Optional[Tuple[float, float]]:
    """(free_mb, total_mb) of GPU 0 via torch, else nvidia-smi, else None."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 2**20, total / 2**20
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        free, total = (float(x) for x in out.splitlines()[0].split(","))
        return free, total
    except Exception:
        return None


def read_val_stats(result_dir: Path) -> List[dict]:
    """Validation stats written by the trainer, sorted by numeric step."""
    stats_dir = Path(result_dir) / "stats"
    rows = []
    for f in stats_dir.glob("val_step*.json") if stats_dir.is_dir() else []:
        m = re.search(r"val_step(\d+)", f.stem)
        if not m:
            continue
        try:
            row = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        row["step"] = int(m.group(1))
        rows.append(row)
    return sorted(rows, key=lambda r: r["step"])


def probe_video(video_path: Path) -> Optional[Dict[str, float]]:
    """fps, frame_count, width, height, duration and mean frame-difference motion (cv2).

    Mirrors ACEZeroPoseEstimator._analyze_video so the result can be recorded before
    pose estimation runs.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    # Sample up to 30 evenly spaced frame pairs for motion
    diffs, prev = [], None
    step = max(1, frame_count // 30)
    for idx in range(0, max(frame_count, 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(gray - prev).mean()))
        prev = gray
    cap.release()
    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if fps > 0 else 0.0,
        "avg_motion": float(sum(diffs) / len(diffs)) if diffs else None,
    }
