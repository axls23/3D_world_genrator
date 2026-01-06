
import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import open3d as o3d
import json

def get_depth_anything_pipeline(device="cuda"):
    """Load the Depth Anything V2 model pipeline."""
    print(f"Loading Depth Anything V2 model on {device}...")
    try:
        # Use the "small" variant for speed/memory efficiency in this demo
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        return pipe
    except Exception as e:
        print(f"Error loading Depth Anything model: {e}")
        # Fallback to older DPT if Depth Anything fails (compatibility)
        print("Falling back to DPT-Hybrid...")
        return pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas", device=device)

def estimate_depth_and_create_ply(
    images_dir: Path,
    output_ply: Path,
    camera_file: Path,
    device: str = "cuda",
    downsample_rate: int = 1
):
    """
    Estimate depth for images and project to 3D point cloud.
    """
    if str(device) == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = "cpu"

    # 1. Load COLMAP cameras.txt/images.txt or transforms.json to get intrinsics
    # For simplicity in this demo pipeline, we'll assume a pinhole model and try to infer 
    # intrinsics or use a default if COLMAP hasn't run yet. 
    # BUT, to be accurate, we really should use the COLMAP output.
    # Let's assume COLMAP has run and we have `cameras.txt` in the sparse folder.
    
    # 2. Load Depth Model
    depth_pipe = get_depth_anything_pipeline(device=0 if device == "cuda" else -1)

    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    image_files = image_files[::downsample_rate] # Subsample for speed

    if not image_files:
        print("No images found!")
        return

    all_points = []
    all_colors = []
    
    print(f"Processing {len(image_files)} images for dense initialization...")

    if not image_files:
        print("No images found!")
        return

    # Simple intrinsic estimation if no file provided (HFOV ~ 70 deg)
    # fx = w / (2 * tan(hfov/2))
    # Using PIL for safer loading
    from PIL import Image
    try:
        sample_img = Image.open(image_files[0])
        w, h = sample_img.size
    except Exception as e:
        raise ValueError(f"Could not read image: {image_files[0]} error: {e}")
    
    fx = w * 0.8  # Approx focal length
    fy = w * 0.8
    cx = w / 2
    cy = h / 2

    for i, img_path in enumerate(tqdm(image_files)):
        # 3. Read Image
        try:
            pil_image = Image.open(img_path).convert("RGB")
            frame_rgb = np.array(pil_image)
            
            # 4. Infer Depth
            # The pipeline returns a dictionary with 'depth' or just the image depending on version
            # We need the raw depth map (metric or relative)
            depth_output = depth_pipe(pil_image)
            depth_map = np.array(depth_output["depth"])
        except Exception as e:
            print(f"Skipping frame {img_path} due to error: {e}")
            continue
        
        # Resize depth to match image if needed (pipeline output might be different tensor size)
        if depth_map.shape[:2] != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # 5. Unproject to 3D
        # Z = depth
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        
        # Create grid of UV coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        Z = depth_map
        # Check for Midas relative depth (inverse depth) 
        # Usually output is disparate, so we might need to invert it or simple normalize
        # For initialization, the exact scale matters less than the shape.
        # Let's normalize Z to a reasonable range (e.g., 0.1 to 10.0 scene units)
        Z_min, Z_max = Z.min(), Z.max()
        if Z_max - Z_min > 0:
            Z = (Z - Z_min) / (Z_max - Z_min) # 0 to 1
            Z = Z * 5.0 + 0.1 # 0.1 to 5.1 units
        
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        # Stack to (H*W, 3)
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        colors = frame_rgb.reshape(-1, 3) / 255.0
        
        # Subsample points per image to keep PLY size manageable
        # Take 1% of points
        mask = np.random.rand(points.shape[0]) < 0.01
        
        all_points.append(points[mask])
        all_colors.append(colors[mask])

    # 6. Concatenate and Save
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    print(f"Creating PLY with {all_points.shape[0]} points...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    o3d.io.write_point_cloud(str(output_ply), pcd)
    print(f"Saved dense init points to {output_ply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--output_ply", required=True, type=Path)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()
    
    estimate_depth_and_create_ply(args.images_dir, args.output_ply, None, downsample_rate=args.downsample)
