#!/usr/bin/env python3
"""
Convert ACE-Zero outputs (poses_final.txt, pc_final.ply) to COLMAP binary format.
Usage: python convert_ace_to_colmap.py --results_dir <ace_output_dir> --images_dir <images_dir>
"""

import argparse
import sys
import struct
import numpy as np
from pathlib import Path
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_poses_final(pose_file):
    """
    Read ACE-Zero poses_final.txt
    Format: image_name qw qx qy qz tx ty tz focal_length confidence
    Returns: list of dicts
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 10:
                continue
            
            # ACE-Zero stores World-to-Camera poses
            entry = {
                'name': Path(tokens[0]).name,  # Ensure just filename
                'qw': float(tokens[1]),
                'qx': float(tokens[2]),
                'qy': float(tokens[3]),
                'qz': float(tokens[4]),
                'tx': float(tokens[5]),
                'ty': float(tokens[6]),
                'tz': float(tokens[7]),
                'focal_length': float(tokens[8]),
                'confidence': float(tokens[9])
            }
            poses.append(entry)
    return poses

def write_cameras_bin(pk_poses, images_dir, output_path):
    """
    Write cameras.bin
    ACE-Zero typically assumes a single camera with a shared focal length.
    """
    if not pk_poses:
        return
    
    # Use first valid image to get dimensions
    first_img_name = pk_poses[0]['name']
    img_path = Path(images_dir) / first_img_name
    
    if not img_path.exists():
        # Try searching if not direct (e.g. if poses have full paths)
        found = list(Path(images_dir).glob(f"**/{first_img_name}"))
        if found:
            img_path = found[0]
        else:
            logger.error(f"Image {first_img_name} not found in {images_dir}")
            return

    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"Could not read image {img_path}")
        return

    height, width = img.shape[:2]
    focal_length = pk_poses[0]['focal_length']
    
    # SIMPLE_PINHOLE: f, cx, cy
    params = [focal_length, width / 2, height / 2]
    
    logger.info(f"Writing cameras.bin (1 camera, {width}x{height}, f={focal_length:.2f})")
    
    with open(output_path, "wb") as f:
        # Camera_ID, Model_ID (1=PINHOLE), Width, Height, Params...
        # Wait, PINHOLE is model_id=1. SIMPLE_PINHOLE is 0.
        # PINHOLE takes fx, fy, cx, cy. ACE gives one f, so fx=fy.
        # SIMPLE_PINHOLE takes f, cx, cy.
        # Let's use PINHOLE (model_id=1) for compatibility.
        
        camera_id = 1
        model_id = 1 # PINHOLE
        
        # Binary format: Camera_ID (uint32), Model_ID (int32), W (uint64), H (uint64), Params (doubles)
        # But wait, COLMAP binary format is simpler?
        # Specification: https://colmap.github.io/format.html#binary-file-format
        # CAMERA_ID (uint32), MODEL_ID (int32), WIDTH (uint64), HEIGHT (uint64), PARAMS[] (double)
        
        f.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
        
        # PINHOLE params: fx, fy, cx, cy
        f.write(struct.pack("<dddd", focal_length, focal_length, width/2, height/2))

def write_images_bin(pk_poses, output_path):
    """
    Write images.bin
    Format: Image_ID, QW, QX, QY, QZ, TX, TY, TZ, Camera_ID, Name
    Review: https://colmap.github.io/format.html#images-bin
    """
    logger.info(f"Writing images.bin ({len(pk_poses)} images)")
    
    name_to_id = {pose['name']: i+1 for i, pose in enumerate(pk_poses)}
    
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(pk_poses)))
        
        for i, pose in enumerate(pk_poses):
            image_id = i + 1
            camera_id = 1
            
            # Header: Image_ID (uint32), QW, QX, QY, QZ, TX, TY, TZ (doubles), Camera_ID (uint32)
            f.write(struct.pack("<IdddddddI", 
                image_id, 
                pose['qw'], pose['qx'], pose['qy'], pose['qz'],
                pose['tx'], pose['ty'], pose['tz'],
                camera_id
            ))
            
            # Name + null terminator
            name_bytes = pose['name'].encode("utf-8") + b"\x00"
            f.write(name_bytes)
            
            # Points2D count (uint64) - we have none (0)
            f.write(struct.pack("<Q", 0))

def write_points3D_bin(ply_file, output_path):
    """
    Convert PLY to points3D.bin
    """
    logger.info(f"Reading {ply_file}...")
    
    try:
        import trimesh
        mesh = trimesh.load(ply_file)
        if isinstance(mesh, trimesh.Scene):
            # Sometimes single obj loaded as scene
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                logger.error("Loaded PLY scene is empty")
                return

        points = mesh.vertices
        colors = mesh.visual.vertex_colors[:, :3] if hasattr(mesh.visual, 'vertex_colors') else np.zeros_like(points)
        
    except ImportError:
        logger.warning("Trimesh not found. Falling back to basic PLY parsing.")
        # Basic PLY parser
        points = []
        colors = []
        header_end = False
        with open(ply_file, 'rb') as f:
            while not header_end:
                line = f.readline().strip()
                if line == b"end_header":
                    header_end = True
            
            # Assume binary little endian float32 x,y,z usually? 
            # Or ascii? ACE export uses trimesh export which defaults to binary if not specified? 
            # Or export_point_cloud.py calls `cloud.export(opt.output_file)`. Trimesh defaults to binary ply.
            # Parsing binary PLY without library is painful. 
            # Let's hope trimesh is there (it is in the environment usually).
            logger.error("Trimesh required to parse binary PLY. Please `pip install trimesh`.")
            return

    logger.info(f"Writing points3D.bin ({len(points)} points)")
    
    with open(output_path, "wb") as f:
        # Num_Points (uint64)
        f.write(struct.pack("<Q", len(points)))
        
        for i in range(len(points)):
            point_id = i + 1
            xyz = points[i]
            rgb = colors[i]
            
            # Point3D_ID (uint64), X, Y, Z (double), R, G, B (uint8), Error (double), TrackLen (uint64)
            f.write(struct.pack("<QdddBBBdQ", 
                point_id, 
                xyz[0], xyz[1], xyz[2], 
                int(rgb[0]), int(rgb[1]), int(rgb[2]), 
                0.0, # Error
                0    # Track Length
            ))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="ACE-Zero output directory (containing poses_final.txt)")
    parser.add_argument("--images_dir", required=True, help="Directory containing original images")
    parser.add_argument("--output_dir", help="Directory to save COLMAP binaries (default: results_dir/colmap)")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    images_path = Path(args.images_dir)
    
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = results_path / "colmap"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Read Poses
    poses_file = results_path / "poses_final.txt"
    if not poses_file.exists():
        logger.error(f"Poses file not found: {poses_file}")
        return
    
    poses = read_poses_final(poses_file)
    logger.info(f"Loaded {len(poses)} poses")
    
    # 2. Write Cameras & Images
    write_cameras_bin(poses, images_path, output_path / "cameras.bin")
    write_images_bin(poses, output_path / "images.bin")
    
    # 3. Write Points
    ply_file = results_path / "pc_final.ply"
    if ply_file.exists():
        write_points3D_bin(ply_file, output_path / "points3D.bin")
    else:
        logger.warning(f"Point cloud {ply_file} not found. Running generate_points_wsl.py logic recommended if dense cloud needed.")
        # Create empty points3D.bin
        with open(output_path / "points3D.bin", "wb") as f:
            f.write(struct.pack("<Q", 0))

if __name__ == "__main__":
    main()
