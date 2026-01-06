import argparse
import os
import sys
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from plyfile import PlyData

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from gsplat.rendering import rasterization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# COORDINATE SYSTEM CONFIGURATION (Critical Gap #3 Fix)
# ============================================================================

class CoordinateConfig:
    """Detect and configure coordinate system conventions"""
    
    @staticmethod
    def detect_convention(means: torch.Tensor) -> str:
        """Detect Y-up vs Z-up based on point cloud distribution"""
        # Compute variance along each axis
        var_y = means[:, 1].var().item()
        var_z = means[:, 2].var().item() if means.shape[1] > 2 else 0
        
        # If Y has much more variance than Z, likely Y-up (standard GL/OpenGL)
        # If Z has more variance, likely Z-up (some CAD/GIS systems)
        if var_y > var_z * 1.5:
            return "y_up"
        elif var_z > var_y * 1.5:
            return "z_up"
        else:
            return "y_up"  # Default to Y-up (standard)
    
    @staticmethod
    def get_up_vector(convention: str, device: torch.device) -> torch.Tensor:
        """Get the up vector for the given convention"""
        if convention == "z_up":
            return torch.tensor([0.0, 0.0, 1.0], device=device)
        else:  # y_up
            return torch.tensor([0.0, 1.0, 0.0], device=device)
    
    @staticmethod
    def get_top_view_config(convention: str, center: torch.Tensor, distance: float, device: torch.device) -> tuple:
        """Get eye position and up vector for top-down view"""
        if convention == "z_up":
            # For Z-up, look down from above along Z
            eye = center + torch.tensor([0.0, 0.001, distance], device=device)
            up = torch.tensor([0.0, 1.0, 0.0], device=device)  # Y is forward
        else:  # y_up
            # For Y-up, look down from above along Y
            eye = center + torch.tensor([0.0, distance, 0.001], device=device)
            up = torch.tensor([0.0, 0.0, 1.0], device=device)  # Z is forward
        return eye, up

def load_ply(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Loads a PLY file and returns the Gaussian Splat parameters as tensors."""
    plydata = PlyData.read(path)
    
    xyz = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']), axis=1)
    opacities = plydata['vertex']['opacity']
    
    scale_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.stack([plydata['vertex'][n] for n in scale_names], axis=1)
    
    rot_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.stack([plydata['vertex'][n] for n in rot_names], axis=1)
    
    # SH features
    features_dc = np.stack([plydata['vertex'][n] for n in ['f_dc_0', 'f_dc_1', 'f_dc_2']], axis=1)
    features_dc = features_dc.reshape(-1, 1, 3)
    
    extra_f_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.stack([plydata['vertex'][n] for n in extra_f_names], axis=1)
    num_extra_features = len(extra_f_names)
    num_extra_coeffs = num_extra_features // 3
    features_extra = features_extra.reshape(-1, num_extra_coeffs, 3)
    
    features = np.concatenate((features_dc, features_extra), axis=1)
    
    return {
        "means": torch.from_numpy(xyz).float().to(device),
        "scales": torch.from_numpy(scales).float().to(device),
        "quats": torch.from_numpy(rots).float().to(device),
        "opacities": torch.from_numpy(opacities).float().to(device),
        "colors": torch.from_numpy(features).float().to(device)
    }

def get_look_at_matrix(eye, center, up):
    z_axis = F.normalize(center - eye, dim=0)
    x_axis = F.normalize(torch.linalg.cross(z_axis, up), dim=0)
    y_axis = F.normalize(torch.linalg.cross(x_axis, z_axis), dim=0)
    
    # View matrix (World to Camera)
    # R = [x_axis, y_axis, z_axis]^T
    # t = -R * eye
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    t = -torch.matmul(R, eye)
    
    view_mat = torch.eye(4, device=eye.device)
    view_mat[:3, :3] = R
    view_mat[:3, 3] = t
    
    # In gsplat/nerfview conventions, we often use c2w or w2c. 
    # rasterization expects viewmats as [C, 4, 4] (World to Camera)
    return view_mat

def get_projection_matrix(fov_y, aspect_ratio, near, far):
    tan_half_fov = math.tan(fov_y / 2)
    proj_mat = torch.zeros(4, 4)
    proj_mat[0, 0] = 1 / (aspect_ratio * tan_half_fov)
    proj_mat[1, 1] = 1 / tan_half_fov
    proj_mat[2, 2] = -(far + near) / (far - near)
    proj_mat[2, 3] = -(2 * far * near) / (far - near)
    proj_mat[3, 2] = -1
    return proj_mat

def render_chunk(chunk_data, output_path, device, resolution=(512, 512)):
    means = chunk_data["means"]
    scales = chunk_data["scales"]
    quats = chunk_data["quats"]
    opacities = chunk_data["opacities"]
    colors = chunk_data["colors"]
    
    # Normalize quats, exp scales, sigmoid opacities
    quats = F.normalize(quats, p=2, dim=-1)
    scales = torch.exp(scales)
    opacities = torch.sigmoid(opacities)
    
    # Calculate center and radius of the chunk
    center = means.mean(dim=0)
    # Simple bounding box radius
    radius = (means.max(dim=0)[0] - means.min(dim=0)[0]).max() / 2.0
    if radius == 0: radius = 1.0
    
    # Detect coordinate system
    convention = CoordinateConfig.detect_convention(means)
    logger.info(f"Detected coordinate convention: {convention}")
    
    # Setup cameras
    # We'll render 4 views: Front, Back, Left, Right (or just rotated around vertical axis)
    # Distance should be enough to see the whole chunk
    distance = radius * 2.5
    fov_y = math.radians(60)
    
    up_vector = CoordinateConfig.get_up_vector(convention, device)
    
    views = []
    # 4 cardinal directions around vertical axis
    angles = [0, 90, 180, 270]
    for angle in angles:
        rad = math.radians(angle)
        eye_x = center[0] + distance * math.sin(rad)
        eye_z = center[2] + distance * math.cos(rad)
        
        if convention == "y_up":
            eye_y = center[1]  # Same height
            eye = torch.tensor([eye_x, eye_y, eye_z], device=device)
        else:  # z_up
            eye_y = center[1]  # Orbit around Y for Z-up
            eye = torch.tensor([eye_x, eye_y, eye_z], device=device)
        
        view_mat = get_look_at_matrix(eye, center, up_vector)
        views.append(view_mat)
        
    # Add a top-down view with proper configuration
    eye_top, up_top = CoordinateConfig.get_top_view_config(convention, center, distance, device)
    views.append(get_look_at_matrix(eye_top, center, up_top))

    width, height = resolution
    K = torch.tensor([
        [width / (2 * math.tan(fov_y / 2)), 0, width / 2],
        [0, height / (2 * math.tan(fov_y / 2)), height / 2],
        [0, 0, 1]
    ], device=device)
    
    rendered_images = []
    failed_views = []
    
    for i, view_mat in enumerate(views):
        try:
            # Debug shapes (commented out for performance)
            # logger.info(f"Shapes: means={means.shape}, quats={quats.shape}, scales={scales.shape}, opacities={opacities.shape}, colors={colors.shape}")
            
            render_colors, render_alphas, _ = rasterization(
                means, quats, scales, opacities, colors,
                view_mat[None], # [1, 4, 4]
                K[None],        # [1, 3, 3]
                width, height,
                sh_degree=0, # Use low degree for speed/robustness
                backgrounds=torch.zeros(1, 3, device=device), # Black background
                packed=False
            )
            
            img = render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()
            img_uint8 = (img * 255).astype(np.uint8)
            
            save_path = output_path / f"view_{i}.png"
            imageio.imsave(save_path, img_uint8)
            rendered_images.append(str(save_path))
        except Exception as e:
            logger.error(f"Rasterization failed for view {i}: {e}")
            failed_views.append(i)
            # CRITICAL FIX: Skip failed view instead of crashing entire chunk
            # Create a blank placeholder image
            blank_img = np.zeros((height, width, 3), dtype=np.uint8)
            save_path = output_path / f"view_{i}_failed.png"
            try:
                imageio.imsave(save_path, blank_img)
                logger.info(f"  Created placeholder for failed view {i}")
            except:
                pass  # If even placeholder fails, just skip
    
    # Log summary
    if failed_views:
        logger.warning(f"Failed to render {len(failed_views)}/{len(views)} views: {failed_views}")
    if rendered_images:
        logger.info(f"Successfully rendered {len(rendered_images)}/{len(views)} views")

    return rendered_images

def main():
    parser = argparse.ArgumentParser(description="Render thumbnails for PLY chunks.")
    parser.add_argument("--input_dir", required=True, help="Directory containing chunk PLY files")
    parser.add_argument("--output_dir", default="chunk_renders", help="Directory to save renders")
    args = parser.parse_args()
    
    # ====================================================================
    # PATH VALIDATION (Critical Gap #3 Fix)
    # ====================================================================
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        sys.exit(1)
    
    logger.info("✅ Path validation passed")
    logger.info(f"   Input: {input_dir}")
    logger.info(f"   Output: {output_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check for manifest
    manifest_path = input_dir / "manifest.json"
    chunks_to_process = []
    
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        for chunk_id, info in manifest["chunks"].items():
            chunks_to_process.append((chunk_id, input_dir / info["file"]))
    else:
        for f in input_dir.glob("*.ply"):
            chunks_to_process.append((f.stem, f))
            
    logger.info(f"Found {len(chunks_to_process)} chunks to render.")
    
    for chunk_id, ply_path in chunks_to_process:
        chunk_out_dir = output_dir / chunk_id
        chunk_out_dir.mkdir(exist_ok=True)
        
        # Skip if already rendered
        if len(list(chunk_out_dir.glob("*.png"))) >= 5:
            logger.info(f"Skipping {chunk_id} (already rendered)")
            continue
            
        logger.info(f"Rendering {chunk_id}...")
        try:
            chunk_data = load_ply(str(ply_path), device)
            if len(chunk_data["means"]) == 0:
                logger.warning(f"Chunk {chunk_id} is empty.")
                continue
                
            render_chunk(chunk_data, chunk_out_dir, device)
        except Exception as e:
            logger.error(f"Failed to render {chunk_id}: {e}")

if __name__ == "__main__":
    main()
