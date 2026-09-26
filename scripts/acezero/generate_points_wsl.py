
import sys
import argparse
import numpy as np
import torch
import cv2
import struct
import logging
from pathlib import Path
# from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import dataset_io
sys.path.append(str(Path(__file__).parent))
import dataset_io

def write_points3D_binary(points3D, path):
    """
    Write 3D points to a COLMAP binary file.
    points3D: list of (xyz, rgb, error) tuples
    """
    path = Path(path)
    with open(path, "wb") as f:
        # Header: # of points (uint64)
        f.write(struct.pack("Q", len(points3D)))
        
        # Data: point3D_id, x, y, z, r, g, b, error, track_len, track_idx, track_img_idx
        # We dummy out track info since we don't have it
        for i, (xyz, rgb) in enumerate(points3D):
            point3d_id = i + 1
            f.write(struct.pack("Q", point3d_id))
            f.write(struct.pack("ddd", xyz[0], xyz[1], xyz[2]))
            f.write(struct.pack("BBB", int(rgb[0]), int(rgb[1]), int(rgb[2])))
            f.write(struct.pack("d", 0.0)) # error
            f.write(struct.pack("Q", 0))   # track_len (0 = no track info)
            # No track content follows if len is 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("poses_file", help="Path to poses_final.txt (ACE format)")
    parser.add_argument("images_root", help="Root directory containing images")
    parser.add_argument("output_path", help="Path to write points3D.bin")
    parser.add_argument("--subsample", type=float, default=0.02, help="Fraction of points to keep (0.01 = 1%)")
    args = parser.parse_args()

    # Load poses using existing utility
    # Confidence 0 to take all poses that made it to final file
    try:
        rgb_files, poses, focal_lengths, _ = dataset_io.load_dataset_ace(args.poses_file, confidence_threshold=0)
    except Exception as e:
        logger.error(f"Failed to load poses: {e}")
        return

    # Load Depth Model
    logger.info("Loading ZoeDepth model...")
    model = dataset_io.get_depth_model()
    
    all_points_xyz = []
    all_points_rgb = []
    
    logger.info(f"Processing {len(rgb_files)} frames...")
    
    for i in range(len(rgb_files)):
        if i % 10 == 0:
            logger.info(f"Processing frame {i}/{len(rgb_files)}")
        img_rel_path = rgb_files[i]
        pose = poses[i] # Camera-to-World 4x4
        fl = focal_lengths[i]
        
        # Resolve image path
        # rgb_files[i] might be absolute or relative. 
        # ACE usually stores absolute paths in WSL. We might need to re-root them.
        # But for now, assume we use the filename to find it in images_root
        img_name = Path(img_rel_path).name
        img_path = Path(args.images_root) / img_name
        
        if not img_path.exists():
            # Try original path
            img_path = Path(img_rel_path)
            if not img_path.exists():
                logger.warning(f"Image not found: {img_name}")
                continue
                
        # Read Image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Infer Depth
        depth = dataset_io.estimate_depth(model, img_rgb)
        
        # Resize depth to match image if needed (Zoe usually outputs high res, but verify)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h))
            
        # Unproject
        cx, cy = w / 2, h / 2
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        Z = depth
        X = (u - cx) * Z / fl
        Y = (v - cy) * Z / fl
        
        # Camera Points (N, 3)
        points_cam = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        colors = img_rgb.reshape(-1, 3)
        
        # Apply random subsampling
        mask = np.random.rand(points_cam.shape[0]) < args.subsample
        points_cam = points_cam[mask]
        colors = colors[mask]
        
        if len(points_cam) == 0:
            continue
            
        # Transform to World
        # pose is CamToWorld
        # P_world = R * P_cam + t
        # Or simply P_world = (Pose @ P_cam_homog.T).T
        
        R = pose[:3, :3].cpu().numpy()
        t = pose[:3, 3].cpu().numpy()
        
        points_world = (R @ points_cam.T).T + t
        
        all_points_xyz.append(points_world)
        all_points_rgb.append(colors)

    if not all_points_xyz:
        logger.error("No points generated!")
        return

    # Concatenate
    final_xyz = np.concatenate(all_points_xyz, axis=0)
    final_rgb = np.concatenate(all_points_rgb, axis=0)
    
    logger.info(f"Writing {len(final_xyz)} points to {args.output_path}...")
    
    # Zip for writer
    points_data = list(zip(final_xyz, final_rgb))
    write_points3D_binary(points_data, args.output_path)
    logger.info("Done.")

if __name__ == "__main__":
    main()
