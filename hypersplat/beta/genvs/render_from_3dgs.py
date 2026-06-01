"""
render_from_3dgs.py — Standalone 3DGS Renderer

Loads a trained 3DGS checkpoint and renders RGB images at specified camera poses.
Used by refine_loop.py to create geometrically consistent feedback images for GeNVS.
"""

import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm

# Add project root for gsplat imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "examples"))

from gsplat.rendering import rasterization


def load_3dgs_checkpoint(ckpt_path: str, device: str = "cuda"):
    """
    Load a 3DGS checkpoint saved by simple_trainer.py.
    
    Returns:
        splats: dict of parameter tensors (means, scales, quats, opacities, sh0, shN)
        step: training step at which the checkpoint was saved
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    splats = {}
    for k, v in ckpt["splats"].items():
        splats[k] = v.to(device)
    
    step = ckpt.get("step", -1)
    print(f"[3DGS Renderer] Loaded checkpoint from step {step}")
    print(f"[3DGS Renderer] {splats['means'].shape[0]} Gaussians")
    
    return splats, step


def render_at_pose(splats, c2w, K, width, height, sh_degree=3, device="cuda"):
    """
    Render a single image from the 3DGS model at a given camera pose.
    
    Args:
        splats: dict of Gaussian parameters
        c2w: [4, 4] camera-to-world matrix (numpy or tensor)
        K: [3, 3] intrinsics matrix (numpy or tensor)
        width: int
        height: int
        sh_degree: int
        
    Returns:
        rgb: [H, W, 3] numpy array (uint8, 0-255)
        depth: [H, W] numpy array (float32)
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).float()
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    
    c2w = c2w.to(device).unsqueeze(0)  # [1, 4, 4]
    K = K.to(device).unsqueeze(0)      # [1, 3, 3]
    
    means = splats["means"]       # [N, 3]
    quats = splats["quats"]       # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N]
    
    if "sh0" in splats:
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]
    else:
        colors = splats.get("colors", torch.zeros(means.shape[0], 1, 3, device=device))
    
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.inverse(c2w),  # rasterization expects world-to-camera
        Ks=K,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+ED",
        near_plane=0.01,
        far_plane=1e10,
    )
    
    rgb = torch.clamp(render_colors[0, ..., :3], 0.0, 1.0)
    depth = render_colors[0, ..., 3]
    
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
    depth_np = depth.cpu().numpy()
    
    return rgb_np, depth_np


def generate_gap_filling_poses(existing_c2ws, num_new=10):
    """
    Generate new camera poses that fill gaps in the existing trajectory.
    Uses midpoint interpolation between the most distant neighboring cameras.
    
    Args:
        existing_c2ws: [N, 4, 4] numpy array of camera-to-world matrices
        num_new: number of new poses to generate
        
    Returns:
        new_poses: list of [4, 4] numpy arrays
        source_indices: list of (idx1, idx2) pairs that were interpolated
    """
    N = len(existing_c2ws)
    centers = existing_c2ws[:, :3, 3]  # [N, 3]
    
    # Compute pairwise distances
    dists = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)  # [N, N]
    np.fill_diagonal(dists, 0)
    
    # Find nearest neighbor for each camera
    np.fill_diagonal(dists, np.inf)
    nn_dists = np.min(dists, axis=1)  # [N]
    nn_indices = np.argmin(dists, axis=1)  # [N]
    
    new_poses = []
    source_pairs = []
    
    # Sort by gap size (largest first)
    gap_order = np.argsort(-nn_dists)
    
    for i in range(min(num_new, N)):
        idx1 = gap_order[i % N]
        idx2 = nn_indices[idx1]
        
        # Interpolate position (midpoint)
        pos = 0.5 * (existing_c2ws[idx1, :3, 3] + existing_c2ws[idx2, :3, 3])
        
        # Interpolate rotation (simple average, re-orthogonalize)
        R1 = existing_c2ws[idx1, :3, :3]
        R2 = existing_c2ws[idx2, :3, :3]
        R_avg = 0.5 * (R1 + R2)
        
        # Re-orthogonalize via SVD
        U, _, Vt = np.linalg.svd(R_avg)
        R_new = U @ Vt
        if np.linalg.det(R_new) < 0:
            R_new = -R_new
        
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_new
        pose[:3, 3] = pos
        
        new_poses.append(pose)
        source_pairs.append((int(idx1), int(idx2)))
    
    return new_poses, source_pairs


def main():
    parser = argparse.ArgumentParser(description="Render images from a trained 3DGS checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to 3DGS checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, help="Original COLMAP dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save rendered images")
    parser.add_argument("--num_views", type=int, default=10, help="Number of novel views to render")
    parser.add_argument("--width", type=int, default=0, help="Render width (0 = auto from dataset)")
    parser.add_argument("--height", type=int, default=0, help="Render height (0 = auto from dataset)")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # 1. Load checkpoint
    splats, step = load_3dgs_checkpoint(args.ckpt, device)
    
    # 2. Load existing poses from the COLMAP dataset
    # We need to know the camera intrinsics and existing poses
    try:
        from datasets.colmap import Parser
        colmap_parser = Parser(data_dir=args.data_dir, factor=1, normalize=False)
        existing_c2ws = colmap_parser.camtoworlds  # [N, 4, 4]
        
        # Get intrinsics from first camera
        first_cam_id = colmap_parser.camera_ids[0]
        K = colmap_parser.Ks_dict[first_cam_id].copy()
        width, height = colmap_parser.imsize_dict[first_cam_id]
        
        if args.width > 0:
            scale_x = args.width / width
            K[0, :] *= scale_x
            width = args.width
        if args.height > 0:
            scale_y = args.height / height
            K[1, :] *= scale_y
            height = args.height
            
    except Exception as e:
        print(f"[Warning] Could not load COLMAP parser: {e}")
        print("[Warning] Using fallback: rendering at existing checkpoint poses only")
        return
    
    # 3. Generate novel poses
    print(f"[3DGS Renderer] Generating {args.num_views} gap-filling poses...")
    new_poses, source_pairs = generate_gap_filling_poses(existing_c2ws, args.num_views)
    
    # 4. Render
    print(f"[3DGS Renderer] Rendering {len(new_poses)} views at {width}x{height}...")
    
    # Write ACE-format pose file for GeNVS consumption
    pose_file = out_dir / "poses_3dgs_rendered.txt"
    
    with torch.no_grad():
        with open(pose_file, "w") as f_pose:
            for i, (pose, (idx1, idx2)) in enumerate(tqdm(
                zip(new_poses, source_pairs), total=len(new_poses), desc="Rendering"
            )):
                rgb, depth = render_at_pose(
                    splats, pose, K, width, height,
                    sh_degree=args.sh_degree, device=device
                )
                
                # Save image
                filename = f"3dgs_render_{i:05d}.jpg"
                save_path = images_dir / filename
                imageio.imwrite(str(save_path), rgb)
                
                # Save depth
                depth_path = out_dir / "depths" 
                depth_path.mkdir(exist_ok=True)
                np.save(str(depth_path / f"3dgs_render_{i:05d}.npy"), depth)
                
                # Write pose in ACE-Zero format:
                # <image_path> <qw> <qx> <qy> <qz> <tx> <ty> <tz> <focal_length> <confidence>
                # ACE uses world-to-camera
                w2c = np.linalg.inv(pose)
                
                # Rotation matrix to quaternion
                from scipy.spatial.transform import Rotation
                R = w2c[:3, :3]
                quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                tx, ty, tz = w2c[:3, 3]
                focal = float(K[0, 0])
                
                f_pose.write(
                    f"{str(save_path.absolute())} "
                    f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{tx:.8f} {ty:.8f} {tz:.8f} "
                    f"{focal:.4f} 1.0\n"
                )
    
    print(f"[3DGS Renderer] Done. Saved {len(new_poses)} images to {images_dir}")
    print(f"[3DGS Renderer] Poses written to {pose_file}")
    
    # Also save metadata for the refine loop
    meta = {
        "checkpoint": args.ckpt,
        "step": step,
        "num_renders": len(new_poses),
        "width": width,
        "height": height,
        "source_pairs": source_pairs,
    }
    with open(out_dir / "render_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
