"""
Inject Novel Views into COLMAP Dataset for 3DGS Training

This script takes:
1. Original COLMAP dataset (from ACE-Zero)
2. Novel view images (from GeNVS-Lite)

And creates an augmented COLMAP dataset with synthetic back-view cameras.

Usage:
    python inject_back_views.py \
        --colmap-dir demo/test_output/acezero_output \
        --novel-views novel_views \
        --output-dir augmented_dataset
"""

import argparse
import os
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json

# Import COLMAP read/write utilities
# Import COLMAP read/write utilities
import sys
try:
    # Try absolute import first (when part of a package)
    from genvs.utils.colmap_read_write_model import (
        read_model,
        write_model,
        Camera,
        Image,
        rotmat2qvec,
        qvec2rotmat
    )
except ImportError:
    # Fallback to local import (when run as script)
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from utils.colmap_read_write_model import (
            read_model,
            write_model,
            Camera,
            Image,
            rotmat2qvec,
            qvec2rotmat
        )
    except ImportError:
        # Fallback for when 'utils' is shadowed by another module (e.g. gsplat.examples.utils)
        sys.path.insert(0, str(Path(__file__).parent / "utils"))
        from colmap_read_write_model import (
            read_model,
            write_model,
            Camera,
            Image,
            rotmat2qvec,
            qvec2rotmat
        )


def compute_scene_center(images: Dict) -> np.ndarray:
    """Compute the center of the scene from camera positions."""
    positions = []
    for img in images.values():
        # Camera position in world coordinates
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        cam_pos = -R.T @ t
        positions.append(cam_pos)
    return np.mean(positions, axis=0)


def mirror_camera_pose(qvec: np.ndarray, tvec: np.ndarray, scene_center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror a camera pose across the scene center to create a "back view" camera.
    
    The idea: 
    - Front camera at position P looks towards scene center C
    - Back camera at position P' = 2*C - P looks back towards C
    - Rotation is flipped 180 degrees around the up axis (Y)
    """
    # Get camera position in world coordinates
    R = qvec2rotmat(qvec)
    cam_pos = -R.T @ tvec
    
    # Mirror position across scene center
    new_cam_pos = 2 * scene_center - cam_pos
    
    # Rotate camera 180 degrees around Y axis (turn around)
    R_180_y = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ], dtype=np.float64)
    
    new_R = R_180_y @ R
    
    # Compute new tvec
    new_tvec = -new_R @ new_cam_pos
    
    # Convert back to quaternion
    new_qvec = rotmat2qvec(new_R)
    
    return new_qvec, new_tvec


def inject_back_views(
    colmap_dir: Path,
    novel_views_dir: Path,
    output_dir: Path,
    camera_offset: float = 0.0
):
    """
    Inject novel back-view images into the COLMAP dataset.
    
    Args:
        colmap_dir: Original COLMAP directory (with sparse/0/)
        novel_views_dir: Directory containing novel view images
        output_dir: Where to save the augmented dataset
        camera_offset: Additional offset for back cameras (default 0)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output structure
    sparse_out = output_dir / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    
    # Read original COLMAP model
    sparse_in = colmap_dir / "sparse" / "0"
    if not sparse_in.exists():
        sparse_in = colmap_dir / "sparse"
    
    cameras, images, points3D = read_model(str(sparse_in))
    
    if cameras is None:
        raise ValueError(f"Could not read COLMAP model from {sparse_in}")
    
    print(f"[Inject] Read {len(cameras)} cameras, {len(images)} images")
    
    # Copy original images
    orig_images_dir = colmap_dir / "images"
    for img_file in orig_images_dir.glob("*"):
        if img_file.is_file():
            shutil.copy2(img_file, images_out / img_file.name)
    print(f"[Inject] Copied original images to {images_out}")
    
    # Compute scene center
    scene_center = compute_scene_center(images)
    print(f"[Inject] Scene center: {scene_center}")
    
    # Get novel view images
    novel_images = sorted(novel_views_dir.glob("*.png"))
    if not novel_images:
        novel_images = sorted(novel_views_dir.glob("*.jpg"))
    
    if not novel_images:
        raise ValueError(f"No novel view images found in {novel_views_dir}")
    
    print(f"[Inject] Found {len(novel_images)} novel views to inject")
    
    # Get reference camera (use the first one for intrinsics)
    ref_camera_id = list(cameras.keys())[0]
    ref_camera = cameras[ref_camera_id]
    
    # Get reference image for pose (use the middle one)
    ref_image_id = list(images.keys())[len(images) // 2]
    ref_image = images[ref_image_id]
    
    # Create new camera and image entries for novel views
    max_image_id = max(images.keys())
    max_camera_id = max(cameras.keys())
    
    new_images = dict(images)  # Copy existing
    new_cameras = dict(cameras)  # Copy existing
    
    for i, novel_img_path in enumerate(novel_images):
        # Copy novel view image
        new_img_name = f"novel_back_{i:04d}.png"
        shutil.copy2(novel_img_path, images_out / new_img_name)
        
        # Create synthetic back-view pose
        # Distribute novel views evenly around the back hemisphere
        angle_offset = (i / len(novel_images)) * np.pi * 0.5  # 90 degree spread
        
        # Use reference pose as base
        base_qvec = ref_image.qvec
        base_tvec = ref_image.tvec
        
        # Mirror it
        new_qvec, new_tvec = mirror_camera_pose(base_qvec, base_tvec, scene_center)
        
        # Add slight rotation variation for better coverage
        # Small perturbation around Y axis
        perturb_angle = (angle_offset - np.pi * 0.25)  # Center around 0
        R_perturb = np.array([
            [np.cos(perturb_angle), 0, np.sin(perturb_angle)],
            [0, 1, 0],
            [-np.sin(perturb_angle), 0, np.cos(perturb_angle)]
        ])
        R_new = R_perturb @ qvec2rotmat(new_qvec)
        new_qvec = rotmat2qvec(R_new)
        
        # Create new image entry
        new_image_id = max_image_id + i + 1
        new_image = Image(
            id=new_image_id,
            qvec=new_qvec,
            tvec=new_tvec,
            camera_id=ref_camera_id,  # Use same camera intrinsics
            name=new_img_name,
            xys=np.zeros((0, 2)),  # No 2D points for synthetic views
            point3D_ids=np.array([], dtype=np.int64)
        )
        new_images[new_image_id] = new_image
    
    print(f"[Inject] Created {len(novel_images)} synthetic back-view cameras")
    
    # Write augmented model (binary format for pycolmap compatibility)
    write_model(new_cameras, new_images, points3D, str(sparse_out), ext=".bin")
    
    # IMPORTANT: Copy original points3D.bin for SFM initialization (don't overwrite with empty)
    orig_points_file = sparse_in / "points3D.bin"
    if orig_points_file.exists():
        shutil.copy2(orig_points_file, sparse_out / "points3D.bin")
        print(f"[Inject] Copied original points3D.bin for SFM init")
    
    print(f"[Inject] Written binary COLMAP files to {sparse_out}")
    
    # Copy downsampled image folders if they exist
    # Also add novel images to each downsampled folder
    for factor in ["2", "4", "8"]:
        src_folder = colmap_dir / f"images_{factor}"
        if src_folder.exists():
            dst_folder = output_dir / f"images_{factor}"
            if not dst_folder.exists():
                shutil.copytree(src_folder, dst_folder)
            # Copy novel images to this folder too (same size for now)
            for novel_img_path in novel_images:
                new_img_name = f"novel_back_{novel_images.index(novel_img_path):04d}.png"
                shutil.copy2(images_out / new_img_name, dst_folder / new_img_name)
    
    print(f"[Inject] Augmented dataset saved to {output_dir}")
    print(f"[Inject] Total images: {len(new_images)} ({len(images)} original + {len(novel_images)} novel)")
    
    # Save injection metadata
    metadata = {
        "original_images": len(images),
        "novel_views": len(novel_images),
        "total_images": len(new_images),
        "scene_center": scene_center.tolist(),
    }
    with open(output_dir / "injection_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Inject Novel Views into COLMAP Dataset")
    parser.add_argument("--colmap-dir", type=str, required=True, help="Original COLMAP directory")
    parser.add_argument("--novel-views", type=str, required=True, help="Novel views directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--camera-offset", type=float, default=0.0, help="Camera offset for back views")
    
    args = parser.parse_args()
    
    inject_back_views(
        colmap_dir=Path(args.colmap_dir),
        novel_views_dir=Path(args.novel_views),
        output_dir=Path(args.output_dir),
        camera_offset=args.camera_offset
    )


if __name__ == "__main__":
    main()
