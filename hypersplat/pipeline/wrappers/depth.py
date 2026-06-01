"""# utils/depth_estimator.py

Small utility to run a depth-estimation pipeline over a folder of images
and produce a simple Open3D PLY point cloud for debugging/initialization.

This file was reformatted and made a few robustness improvements:
- consistent device handling for transformers.pipeline (int device index or -1)
- removed duplicate checks
- safer handling of pipeline output variants
- deterministic subsampling seed for reproducibility
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

def get_depth_anything_pipeline(device_idx: int = 0):
    """Load the Depth Anything V2 model pipeline.

    device_idx: int
        0 or positive for GPU device index, -1 for CPU.
    """
    device_arg = device_idx if isinstance(device_idx, int) else 0
    print(f"Loading Depth Anything V2 model on device {device_arg}...")
    try:
        # Use the "small" variant for speed/memory efficiency in this demo
        return pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device_arg,
        )
    except Exception as exc:  # pragma: no cover - fallback at runtime
        print(f"Error loading Depth Anything model: {exc}")
        # Fallback to older DPT if Depth Anything fails (compatibility)
        print("Falling back to DPT-Hybrid...")
        return pipeline(
            task="depth-estimation", model="Intel/dpt-hybrid-midas", device=device_arg
        )

def estimate_depth_and_create_ply(
    images_dir: Path,
    output_ply: Path,
    camera_file: Optional[Path] = None,
    device: str = "cuda",
    downsample_rate: int = 1,
):
    """Estimate depth for images and project to a 3D point cloud.

    This function tries to be robust to different pipeline outputs and to
    missing COLMAP intrinsics by using a reasonable default focal length.
    """
    if str(device) == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = "cpu"

    # Determine device index for transformers.pipeline: 0 for first GPU, -1 for CPU
    device_idx = 0 if device == "cuda" and torch.cuda.is_available() else -1

    depth_pipe = get_depth_anything_pipeline(device_idx=device_idx)

    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    image_files = image_files[:: max(1, downsample_rate)]

    if not image_files:
        print("No images found!")
        return

    print(f"Processing {len(image_files)} images for dense initialization...")

    # Simple intrinsic estimation if no file provided (HFOV ~ 70 deg)
    try:
        sample_img = Image.open(image_files[0])
        w, h = sample_img.size
    except Exception as exc:
        raise ValueError(f"Could not read image: {image_files[0]} error: {exc}")

    # Approximate focal lengths
    fx = w * 0.8
    fy = w * 0.8
    cx = w / 2.0
    cy = h / 2.0

    all_points = []
    all_colors = []

    rng = np.random.default_rng(0)

    for img_path in tqdm(image_files):
        try:
            pil_image = Image.open(img_path).convert("RGB")
            frame_rgb = np.array(pil_image)

            depth_output = depth_pipe(pil_image)

            # Handle different pipeline return formats
            if isinstance(depth_output, dict):
                # Some HF pipelines return {'depth': <PIL.Image>}
                if "depth" in depth_output:
                    depth_map = np.array(depth_output["depth"])  # type: ignore[index]
                elif "predictions" in depth_output:
                    # Sometimes nested
                    depth_map = np.array(depth_output["predictions"][0])
                else:
                    # Fallback: try to convert entire dict to array (unlikely)
                    raise RuntimeError("Unexpected depth pipeline output format: dict")
            elif isinstance(depth_output, list):
                # Some versions return a list with first element containing 'depth'
                first = depth_output[0]
                if isinstance(first, dict) and "depth" in first:
                    depth_map = np.array(first["depth"])  # type: ignore[index]
                else:
                    depth_map = np.array(first)
            else:
                # Could be a PIL.Image or a numpy array
                try:
                    depth_map = np.array(depth_output)
                except Exception:
                    raise RuntimeError("Could not interpret depth pipeline output")

        except Exception as exc:
            print(f"Skipping frame {img_path} due to error: {exc}")
            continue

        # Resize depth to match image if needed
        if depth_map.shape[:2] != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize depth to a reasonable scene range for initialization
        Z = depth_map.astype(float)
        Z_min, Z_max = float(Z.min()), float(Z.max())
        if Z_max - Z_min > 0.0:
            Z = (Z - Z_min) / (Z_max - Z_min)
            Z = Z * 5.0 + 0.1

        u, v = np.meshgrid(np.arange(w), np.arange(h))

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        colors = frame_rgb.reshape(-1, 3) / 255.0

        # Subsample deterministically
        mask = rng.random(points.shape[0]) < 0.01

        all_points.append(points[mask])
        all_colors.append(colors[mask])

    if not all_points:
        print("No valid points were generated, aborting.")
        return

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    print(f"Creating PLY with {all_points.shape[0]} points...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.io.write_point_cloud(str(output_ply), pcd)
    print(f"Saved dense init points to {output_ply}")


class DepthEstimator:
    """Unified depth estimator class wrapping underlying model backends."""
    def __init__(self, model: str = "zoedepth", device: str = "cuda"):
        from scripts.acezero.dataset_io import get_depth_model
        self.model_type = model
        self.device = device
        self.model = get_depth_model(model_type=model)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        from scripts.acezero.dataset_io import estimate_depth
        return estimate_depth(self.model, image_rgb)

    def unload(self):
        if hasattr(self, 'model'):
            del self.model
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--output_ply", required=True, type=Path)
    parser.add_argument("--camera_file", type=Path)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()

    estimate_depth_and_create_ply(
        args.images_dir, args.output_ply, args.camera_file, downsample_rate=args.downsample
    )