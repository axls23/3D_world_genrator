
import sys
import os
import argparse
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hypersplat.beta.genvs.pipeline import GeNVSPipeline
from hypersplat.beta.genvs.dataset import GenVSDataset
from scripts.acezero.dataset_io import write_pose_to_pose_file

def generate_trajectory(poses, num_views=20):
    """
    Generate a simple spiral or interpolation trajectory based on input poses.
    For simplicity: Interpolate between random pairs or create a circle around mean center.
    """
    # Simple: Random Orbit around mean center
    poses_np = np.array([p.cpu().numpy() for p in poses])
    centers = poses_np[:, :3, 3] # [N, 3]
    mean_center = np.mean(centers, axis=0) # [3]
    
    # Estimate radius
    dists = np.linalg.norm(centers - mean_center, axis=1)
    radius = np.mean(dists)
    
    # Generic spiral logic (simplified)
    # Just return existing poses for now if we want to "densify" near them, 
    # OR better: Randomly perturb existing poses to create "nearby" views.
    # Refinement usually needs "filling gaps".
    # Strategy: Perturb each existing pose slightly (e.g. 5 degrees, 10cm)
    
    new_poses = []
    parent_indices = []
    
    # Mode: "vr_sphere" - extensive coverage
    # Generate partial sphere (upper hemisphere) looking at mean center
    
    # Vectors
    up = torch.tensor([0., 0., 1.]) # Assume Z up? ACE is usually Z forward or Y down? 
    # ACE/COLMAP: Z is viewing direction.
    # Let's rely on LookAt logic.
    
    def look_at(eye, target, up=torch.tensor([0., -1., 0.], dtype=torch.float32)):
        # OpenCV Style: Forward is +Z, Y is Down
        fwd = target - eye
        fwd = fwd / torch.norm(fwd) # Target direction
        
        z_axis = fwd # Camera Z points AT target
        
        # We use Up=[0, -1, 0] because camera Y points down
        x_axis = torch.cross(up, z_axis, dim=0)
        x_axis = x_axis / torch.norm(x_axis)
        y_axis = torch.cross(z_axis, x_axis, dim=0) # This will be the "Down" vector
        
        # 4x4 matrix (Cam-to-World)
        # Columns are X, Y, Z axes in World Space
        R = torch.stack([x_axis, y_axis, z_axis], dim=1) 
        pose = torch.eye(4, dtype=torch.float32)
        pose[:3, :3] = R
        pose[:3, 3] = eye
        return pose

    # Simple circular path at mean height
    theta = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    for t in theta:
        # X, Y circle
        x = mean_center[0] + radius * np.cos(t)
        y = mean_center[1] + radius * np.sin(t)
        z = mean_center[2] # Maintain height
        
        eye = torch.tensor([x, y, z], dtype=torch.float32)
        target = torch.tensor(mean_center, dtype=torch.float32)
        
        # Determine "Up" from existing poses?
        # Just use global Y up or Z up?
        # Let's assume standard Y-down camera (COLMAP/OpenCV).
        # We construct LookAt.
        
        pose = look_at(eye, target, up=torch.tensor([0., -1., 0.], dtype=torch.float32)) # Y downish
        
        # But we need to use a Source Image for conditioning.
        # Pick NEAREST sourcepose to this new pose?
        # Find nearest
        dists = np.linalg.norm(centers - eye.numpy(), axis=1)
        src_idx = np.argmin(dists)
        
        new_poses.append(pose)
        parent_indices.append(src_idx)
        
    return new_poses, parent_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, help="ACE output directory")
    parser.add_argument("--num_views", type=int, default=10, help="Number of new views to generate")
    parser.add_argument("--output_dir", type=str, default="results/genvs_refined")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()
    
    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # 1. Load Pipeline
    print(f"[Inference] Loading pipeline from {args.checkpoint}...")
    pipeline = GeNVSPipeline(device=device, model_channels=64)
    
    # Load weights
    ckpt = torch.load(args.checkpoint, map_location=device)
    pipeline.encoder.load_state_dict(ckpt['encoder'])
    pipeline.renderer.load_state_dict(ckpt['renderer'])
    pipeline.unet.load_state_dict(ckpt['unet'])
    pipeline.volume_struct.load_state_dict(ckpt['volume']) # Frustum often has buffers
    
    # 2. Load Dataset (Training Source)
    # We use this to get "Source" views to condition on.
    dataset = GenVSDataset(args.data_dir, image_size=args.image_size)
    
    # 3. Generate New Poses
    print(f"[Inference] Generating {args.num_views} target poses...")
    all_poses = [item['pose'] for item in dataset.items]
    target_poses, source_indices = generate_trajectory(all_poses, num_views=args.num_views)
    
    # 4. Inference Loop
    print("[Inference] Starting generation...")
    
    # Log file for new poses
    pose_file_path = out_dir / "poses_generated.txt"
    f_pose = open(pose_file_path, 'w')
    
    with torch.no_grad():
        for i, (tgt_pose, src_idx) in tqdm(enumerate(zip(target_poses, source_indices)), total=args.num_views):
            # Source View
            src_item = dataset.load_view(src_idx)
            src_img = src_item['image'].unsqueeze(0).to(device) # [1, 3, H, W]
            src_pose = src_item['pose'].unsqueeze(0).to(device) # [1, 4, 4]
            src_K = src_item['K'].unsqueeze(0).to(device)
            
            # Target
            # Target Poses: [1, 4, 4]
            # Target Ks: [1, 3, 3] (reuse source)
            
            # NORMALIZE TARGET POSE
            tgt_pose_norm = tgt_pose.clone()
            tgt_pose_norm[:3, 3] = (tgt_pose[:3, 3] - dataset.scene_center) * dataset.scene_scale
            
            tgt_pose_input = tgt_pose_norm.unsqueeze(0).to(device)
            
            result = pipeline.sample_batch(
                src_img,
                src_pose,
                src_K,
                tgt_pose_input,
                src_K, # Use source intrinsics for target
                num_steps=50, 
            )
            
            # Result is [-1, 1]. Convert to [0, 255]
            gen_img = result[0].cpu().permute(1, 2, 0).numpy() # [H, W, 3]
            gen_img = (gen_img * 0.5 + 0.5) * 255
            gen_img = np.clip(gen_img, 0, 255).astype(np.uint8)
            
            # Save Image
            filename = f"gen_frame_{i:05d}.jpg"
            save_path = images_dir / filename
            Image.fromarray(gen_img).save(save_path)
            
            # Save Pose (ACE format)
            # Needs to back-convert pose to World-to-Camera for the file, if `write_pose_to_pose_file` expects that?
            # `write_pose_to_pose_file` docstring: "pose: ... world-to-cam".
            # Our `tgt_pose` is Cam-to-World (dataset loaded as c2w).
            # So we invert it.
            w2c = tgt_pose.inverse()
            focal = src_item['K'][0, 0].item() # Use source focal
            
            # Wait, `write_pose` uses absolute path for image file?
            # Yes. `Path(rgb_file)`.
            write_pose_to_pose_file(
                f_pose,
                str(save_path.absolute()),
                w2c.cpu().numpy(),
                confidence=1.0, # Synthetic confidence
                focal_length=focal
            )
            
    f_pose.close()
    print(f"[Inference] Done. Saved {args.num_views} images to {images_dir}")
    print(f"[Inference] Saved poses to {pose_file_path}")

if __name__ == "__main__":
    main()
