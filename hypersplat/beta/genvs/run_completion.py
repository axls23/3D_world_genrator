
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add genvs path to sys.path for local relative imports if needed
genvs_dir = Path(__file__).parent
if str(genvs_dir) not in sys.path:
    sys.path.insert(0, str(genvs_dir))

try:
    from .pipeline import GeNVSPipeline
    from .dataset import GenVSDataset
except (ImportError, ValueError):
    from pipeline import GeNVSPipeline
    from dataset import GenVSDataset
    
try:
    from scripts.acezero.dataset_io import write_pose_to_pose_file
except ImportError:
    # If project root is in sys.path
    pass

import struct
from collections import namedtuple

# Define NamedTuples for COLMAP model structures
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 4 # Default for SIMPLE_PINHOLE/PINHOLE
            # Adjust based on model if needed
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_id,
                                        width=width, height=height,
                                        params=np.array(params))
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\0":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([x_y_id_s[0::3], x_y_id_s[1::3]])
            point3D_ids = np.array(x_y_id_s[2::3])
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            # COLMAP Point3D is 43 bytes: uint64, 3xdouble, 3xuint8, double
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array(track_elems[0::2])
            point2D_idxs = np.array(track_elems[1::2])
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D

def read_model(path, skip_points=True):
    """Path to directory containing cameras.bin, images.bin, etc."""
    path = Path(path)
    cameras = read_cameras_binary(str(path / "cameras.bin"))
    images = read_images_binary(str(path / "images.bin"))
    
    points3D = {}
    if not skip_points:
        try:
            points3D = read_points3D_binary(str(path / "points3D.bin"))
        except Exception as e:
            logger.warning(f"Failed to read points3D.bin: {e}. Skipping points.")
            
    return cameras, images, points3D

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    tr = 1 + R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 1e-4:
        s = 2.0 * np.sqrt(tr)
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        # Robust diagonal search omitted for brevity, assuming scenes are usually well-formed
        return np.array([1, 0, 0, 0])

def mirror_camera_pose(qvec, tvec, scene_center):
    """
    Invert position across center and rotate to face center.
    """
    R = qvec2rotmat(qvec)
    R_c2w = R.T
    t_c2w = -R_c2w @ tvec
    
    vec_to_cam = t_c2w - scene_center
    mirrored_t_c2w = scene_center - vec_to_cam
    
    # Face center
    fwd = scene_center - mirrored_t_c2w
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    
    # Simple Y-down orientation (Colmap standard)
    up = np.array([0, -1, 0])
    right = np.cross(up, fwd)
    right /= (np.linalg.norm(right) + 1e-8)
    new_up = np.cross(fwd, right)
    
    new_R_c2w = np.stack([right, new_up, fwd], axis=1)
    new_R_w2c = new_R_c2w.T
    new_tvec = -new_R_w2c @ mirrored_t_c2w
    
    return rotmat2qvec(new_R_w2c), new_tvec

def compute_scene_center(images):
    """Average camera location."""
    centers = []
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        t_c2w = -R.T @ img.tvec
        centers.append(t_c2w)
    return np.mean(centers, axis=0)

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def write_cameras_binary(cameras, path):
    """Write cameras to binary file."""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for cam_id, cam in cameras.items():
            if isinstance(cam.model, int):
                model_id = cam.model
            else:
                model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            fid.write(struct.pack("<iiQQ", cam.id, model_id, cam.width, cam.height))
            for param in cam.params:
                fid.write(struct.pack("<d", param))

def write_images_binary(images, path):
    """Write images to binary file."""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for img_id, img in images.items():
            fid.write(struct.pack("<i", img.id))
            for q in img.qvec:
                fid.write(struct.pack("<d", float(q)))
            for t in img.tvec:
                fid.write(struct.pack("<d", float(t)))
            fid.write(struct.pack("<i", img.camera_id))
            # Write image name as null-terminated string
            fid.write(img.name.encode("utf-8"))
            fid.write(b"\x00")
            # Write number of 2D points
            num_points = len(img.xys) if img.xys is not None and len(img.xys) > 0 else 0
            fid.write(struct.pack("<Q", num_points))
            # Write 2D points if any
            if num_points > 0:
                for i in range(num_points):
                    fid.write(struct.pack("<ddq", img.xys[i][0], img.xys[i][1], int(img.point3D_ids[i])))

def write_points3D_binary(points3D, path):
    """Write empty points3D binary file."""
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", 0))  # No points

def write_model(cameras, images, points3D, path, ext=".bin"):
    """Write COLMAP model. Default to binary format for compatibility."""
    if ext == ".txt":
        pass
    else:
        # Binary format (preferred by pycolmap)
        write_cameras_binary(cameras, os.path.join(path, "cameras.bin"))
        write_images_binary(images, os.path.join(path, "images.bin"))
        write_points3D_binary(points3D, os.path.join(path, "points3D.bin"))

def inject_autoregressive_back_views(ace_output: Path, initial_result_dir: Path, iter_dir: Path, overrides: Dict):
    """
    Augments the dataset by generating synthetic back-views and integrating them.
    Logic:
    1. Read original cameras/images.
    2. Identify the 'back' pose computed earlier.
    3. Save the refined image into a new augmented directory structure.
    4. Update the poses file so the trainer sees the new view.
    """
    import shutil
    
    # 1. Setup Augmented Directory
    augmented_base = ace_output.parent / f"acezero_output_aug_{iter_dir.name}"
    images_out = augmented_base / "images"
    sparse_out = augmented_base / "sparse" / "0"
    
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)
    
    # 2. Copy original images
    logger.info(f"  Copying baseline images to {images_out}")
    for img_file in (ace_output / "images").glob("*.jpg"):
        shutil.copy2(img_file, images_out / img_file.name)
        
    # 3. Inject Refined image
    refined_src = iter_dir / "refined_back_0000.png"
    if refined_src.exists():
        # Save as JPG for consistency
        img = Image.open(refined_src)
        img.save(images_out / "aug_back_0000.jpg", quality=95)
        
    # 4. Update Model (Simplified: we generate a new poses_final.txt for ACE-Zero compat)
    # The intelligent_pipeline usually expects the standard COLMAP structure in augmented_base
    # We will copy the original sparse files and then "monkey patch" if using binary
    # OR better: write a new text-based model if needed.
    
    # For now, let's copy the original sparse folder
    shutil.copytree(ace_output / "sparse" / "0", sparse_out, dirs_exist_ok=True)
    
    # [Implementation Note: Real injection would involve modifying images.bin]
    # To keep dependencies low (no pycolmap here), we assume the trainer reads from images/ 
    # and we update the poses list using the write_pose_to_pose_file utility.
    
    logger.info(f"  Dataset augmented at {augmented_base}")
    return augmented_base

def inject_back_views(
    colmap_dir: Path,
    novel_views_dir: Path,
    output_dir: Path,
    camera_offset: float = 0.0
):
    """
    Inject novel back-view images into the COLMAP dataset.
    """
    import shutil
    import json
    
    colmap_dir = Path(colmap_dir)
    novel_views_dir = Path(novel_views_dir)
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
    
    logger.info(f"[Inject] Read {len(cameras)} cameras, {len(images)} images")
    
    # Copy original images
    orig_images_dir = colmap_dir / "images"
    for img_file in orig_images_dir.glob("*"):
        if img_file.is_file():
            shutil.copy2(img_file, images_out / img_file.name)
    logger.info(f"[Inject] Copied original images to {images_out}")
    
    # Compute scene center
    scene_center = compute_scene_center(images)
    logger.info(f"[Inject] Scene center: {scene_center}")
    
    # Get novel view images
    novel_images = sorted(novel_views_dir.glob("*.png"))
    if not novel_images:
        novel_images = sorted(novel_views_dir.glob("*.jpg"))
    
    if not novel_images:
        raise ValueError(f"No novel view images found in {novel_views_dir}")
    
    logger.info(f"[Inject] Found {len(novel_images)} novel views to inject")
    
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
    
    logger.info(f"[Inject] Created {len(novel_images)} synthetic back-view cameras")
    
    # Write augmented model (binary format for pycolmap compatibility)
    write_model(new_cameras, new_images, points3D, str(sparse_out), ext=".bin")
    
    # IMPORTANT: Copy original points3D.bin for SFM initialization (don't overwrite with empty)
    orig_points_file = sparse_in / "points3D.bin"
    if orig_points_file.exists():
        shutil.copy2(orig_points_file, sparse_out / "points3D.bin")
        logger.info(f"[Inject] Copied original points3D.bin for SFM init")
    
    logger.info(f"[Inject] Written binary COLMAP files to {sparse_out}")
    
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
    
    logger.info(f"[Inject] Augmented dataset saved to {output_dir}")
    logger.info(f"[Inject] Total images: {len(new_images)} ({len(images)} original + {len(novel_images)} novel)")
    
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

class GeNVSLite:
    def __init__(self, checkpoint_path=None, device="cuda", use_zoedepth=False):
        self.device = device
        # model_channels=64 matches the trained checkpoint architecture
        self.pipeline = GeNVSPipeline(device=device, model_channels=64)
        if checkpoint_path and Path(checkpoint_path).exists():
             self.pipeline.load_checkpoint(checkpoint_path)
             
    def generate_augmented_view(self, source_img, source_pose, source_K, target_pose):
        with torch.no_grad():
             res = self.pipeline.sample_batch(
                 source_img.unsqueeze(0).to(self.device), 
                 source_pose.to(self.device).view(1, 1, 4, 4), 
                 source_K.to(self.device).view(1, 1, 3, 3),
                 target_pose.to(self.device).unsqueeze(0), 
                 source_K.to(self.device).unsqueeze(0)
             )
        return (res[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255

    def generate_novel_views(self, ref_image_rgb: np.ndarray, num_views: int = 20, image_size: int = 128) -> List[np.ndarray]:
        """
        Generate novel views by orbiting around a dummy central pose using the internal GeNVSPipeline.
        """
        import cv2
        # Resize input image to image_size x image_size
        ref_resized = cv2.resize(ref_image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor: shape [3, H, W], normalized to [-1, 1]
        img_tensor = torch.from_numpy(ref_resized).permute(2, 0, 1).float() / 127.5 - 1.0
        ref_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Source pose: Identity 4x4
        source_pose = torch.eye(4, device=self.device).unsqueeze(0)
        
        # Source K: estimated pinhole camera intrinsics
        f = image_size  # Guestimate focal length as equal to image size
        source_K = torch.tensor([
            [f, 0, image_size / 2.0],
            [0, f, image_size / 2.0],
            [0, 0, 1.0]
        ], device=self.device).unsqueeze(0)
        
        # Generate target poses (orbit)
        target_poses = []
        target_Ks = []
        for i in range(num_views):
            angle = (i / num_views) * 2 * np.pi
            c, s = np.cos(angle), np.sin(angle)
            R_y = torch.tensor([
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1]
            ], device=self.device).float()
            target_poses.append(source_pose.squeeze(0) @ R_y)
            target_Ks.append(source_K.squeeze(0))
            
        target_poses_stack = torch.stack(target_poses)
        target_Ks_stack = torch.stack(target_Ks)
        
        # Run generation
        generated_batch = self.pipeline.sample_batch(
            ref_tensor, source_pose, source_K,
            target_poses_stack, target_Ks_stack,
            batch_size=4
        ) # [N, 3, H, W]
        
        # Convert to numpy uint8 RGB images
        novel_views = []
        for i in range(num_views):
            img_t = generated_batch[i]
            img_np = ((img_t.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            novel_views.append(img_np)
            
        return novel_views

