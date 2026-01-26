"""
Recover COLMAP sparse files from poses_final.txt and images.
This fixes the ZeroDivisionError caused by cameras.bin having zero dimensions.
"""
from PIL import Image
import struct
import numpy as np
from pathlib import Path

def recover_colmap_sparse(base_dir: Path):
    """Regenerate cameras.bin, images.bin, points3D.bin from poses and images."""
    
    images_dir = base_dir / "images"
    sparse_dir = base_dir / "sparse" / "0"
    poses_file = base_dir / "poses_final.txt"
    
    # Ensure sparse dir exists
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get image dimensions from first image
    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    
    first_img = Image.open(image_files[0])
    w, h = first_img.size
    print(f"[INFO] Image dimensions: {w}x{h}")
    
    # Format: filename qw qx qy qz tx ty tz confidence focal
    poses = {}
    focals = []
    if poses_file.exists():
        with open(poses_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 10:
                    name = Path(tokens[0]).name
                    qw, qx, qy, qz = float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
                    tx, ty, tz = float(tokens[5]), float(tokens[6]), float(tokens[7])
                    focal_val = float(tokens[9])
                    poses[name] = (qw, qx, qy, qz, tx, ty, tz)
                    focals.append(focal_val)
        
        # Use mean focal from file if available, otherwise fallback to estimation
        if focals:
            focal = sum(focals) / len(focals)
            print(f"[INFO] Using mean focal length from pose file: {focal:.2f}")
        else:
            # Estimate focal length (~70 deg HFOV)
            focal = w / (2 * np.tan(np.radians(35)))
            print(f"[INFO] Estimated focal length (fallback): {focal:.2f}")
        
        print(f"[INFO] Parsed {len(poses)} poses from {poses_file}")
    else:
        raise FileNotFoundError(f"Poses file not found: {poses_file}")
    
    # 3. Write cameras.bin (PINHOLE model)
    cameras_path = sparse_dir / "cameras.bin"
    with open(cameras_path, 'wb') as f:
        f.write(struct.pack('<Q', 1))  # num_cameras = 1
        f.write(struct.pack('<I', 1))  # camera_id = 1
        f.write(struct.pack('<i', 1))  # model = PINHOLE (1)
        f.write(struct.pack('<Q', w))  # width
        f.write(struct.pack('<Q', h))  # height
        f.write(struct.pack('<d', focal))  # fx
        f.write(struct.pack('<d', focal))  # fy
        f.write(struct.pack('<d', w/2))    # cx
        f.write(struct.pack('<d', h/2))    # cy
    print(f"[INFO] Wrote cameras.bin: PINHOLE {w}x{h}, f={focal:.2f}")
    
    # 4. Write images.bin
    images_path = sparse_dir / "images.bin"
    with open(images_path, 'wb') as f:
        # Only include images that have poses
        valid_images = [(name, poses[name]) for name in sorted(poses.keys()) if name in poses]
        f.write(struct.pack('<Q', len(valid_images)))  # num_images
        
        for i, (name, pose) in enumerate(valid_images, 1):
            qw, qx, qy, qz, tx, ty, tz = pose
            
            f.write(struct.pack('<I', i))  # image_id
            f.write(struct.pack('<d', qw))  # qw
            f.write(struct.pack('<d', qx))  # qx
            f.write(struct.pack('<d', qy))  # qy
            f.write(struct.pack('<d', qz))  # qz
            f.write(struct.pack('<d', tx))  # tx
            f.write(struct.pack('<d', ty))  # ty
            f.write(struct.pack('<d', tz))  # tz
            f.write(struct.pack('<I', 1))   # camera_id = 1
            f.write(name.encode('utf-8') + b'\x00')  # image name (null-terminated)
            f.write(struct.pack('<Q', 0))   # num_points2D = 0 (no feature points)
    
    print(f"[INFO] Wrote images.bin: {len(valid_images)} images")
    
    # 5. Generate synthetic points3D.bin from camera positions
    # Create a point cloud around camera centers for initialization
    points3d_path = sparse_dir / "points3D.bin"
    
    # Extract camera centers from poses (C = -R^T * t)
    camera_centers = []
    from scipy.spatial.transform import Rotation
    for name, pose in poses.items():
        qw, qx, qy, qz, tx, ty, tz = pose
        # Convert quat to rotation matrix
        r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz])
        # Camera center in world coordinates
        center = -r.T @ t
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    scene_center = np.mean(camera_centers, axis=0)
    # Use standard deviation to define the "cloud of interest"
    scene_scale = np.std(camera_centers) * 1.5 if len(camera_centers) > 1 else 1.0
    
    # Generate random points around scene center
    num_points = 10000  # Initial point cloud size
    np.random.seed(42)
    points = scene_center + np.random.randn(num_points, 3) * scene_scale
    colors = np.random.randint(100, 200, size=(num_points, 3), dtype=np.uint8)  # Gray-ish colors
    
    with open(points3d_path, 'wb') as f:
        f.write(struct.pack('<Q', num_points))  # num_points
        
        for i in range(num_points):
            point3d_id = i + 1
            xyz = points[i]
            rgb = colors[i]
            error = 1.0  # Reprojection error (dummy)
            track_len = 0  # No image associations
            
            f.write(struct.pack('<Q', point3d_id))  # point3D_id
            f.write(struct.pack('<ddd', xyz[0], xyz[1], xyz[2]))  # x, y, z
            f.write(struct.pack('<BBB', rgb[0], rgb[1], rgb[2]))  # r, g, b
            f.write(struct.pack('<d', error))  # error
            f.write(struct.pack('<Q', track_len))  # track length
    
    print(f"[INFO] Wrote points3D.bin: {num_points} synthetic points around scene center")
    
    print(f"\n[SUCCESS] COLMAP sparse files recovered to {sparse_dir}")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path(r"C:\Users\sxhil_25660\3D_world_genrator\demo\test_run_genvs_gpu_max_seed2089\acezero_output")
    
    recover_colmap_sparse(base_dir)
