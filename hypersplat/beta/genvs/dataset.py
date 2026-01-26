
import sys
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Add project root to path for imports
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from scripts.acezero import dataset_io
except ImportError:
    # Fallback if run from different CWD
    sys.path.append(str(project_root / "scripts"))
    from acezero import dataset_io

class GenVSDataset(Dataset):
    """
    Dataset for GeNVS training.
    Loads poses from ACE output (poses_final.txt) and images.
    Returns:
        dictionary containing:
        - target_view: {image, pose, K}
        - source_views: {image, pose, K} (Single or Multiple)
    """
    def __init__(self, 
                 root_dir, 
                 pose_file="acezero_output/poses_final.txt", 
                 image_dir="acezero_output/images",
                 image_size=256,
                 num_source_views=1,
                 validation=False):
        
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.num_source_views = num_source_views
        
        # Resolve paths
        self.pose_file_path = self.root_dir / pose_file
        self.local_image_dir = self.root_dir / image_dir
        
        if not self.pose_file_path.exists():
            raise FileNotFoundError(f"Pose file not found: {self.pose_file_path}")
            
        print(f"[GeNVSDataset] Loading poses from {self.pose_file_path}")
        
        # Load dataset using ACE util
        # Note: ACE loads World-to-Cam but converts to Cam-to-World internally (inverse)
        # We assume 0 confidence threshold to get all poses
        rgb_files, poses, focal_lengths, _ = dataset_io.load_dataset_ace(
            str(self.pose_file_path), confidence_threshold=0
        )
        
        # Calculate Scene Center and Scale
        all_centers = torch.stack([p[:3, 3] for p in poses])
        self.scene_center = torch.mean(all_centers, dim=0)
        # Scale: max distance from center -> map to radius ~2.0 (fits in 10.0 frustum)
        dists = torch.norm(all_centers - self.scene_center, dim=1)
        self.scene_scale = 2.0 / (torch.max(dists) + 1e-5)
        
        print(f"[GeNVSDataset] Scene Norm: Center={self.scene_center.numpy()}, Scale={self.scene_scale.item():.2f}")
        
        self.items = []
        
        # Pre-calculate relative poses for sampling
        poses_stack = torch.stack(poses)
        self.neighbors = self.compute_nearest_neighbors(poses_stack)
        
        for i in range(len(rgb_files)):
            # Handle path correction
            # Raw: /tmp/acezero_images_.../frame_00001.jpg
            # Target: root_dir/image_dir/frame_00001.jpg
            filename = Path(rgb_files[i]).name
            local_path = self.local_image_dir / filename
            
            if not local_path.exists():
                # Try finding it recursively? Or maybe just in 'images' folder?
                # Fallback to root images if not found in specific folder
                fallback = self.root_dir / "images" / filename
                if fallback.exists():
                    local_path = fallback
                else:
                    # Skip missing files
                    print(f"Warning: Image not found {local_path} or {fallback}")
                    continue
            
            # Create intrinsics matrix (3x3)
            fl = focal_lengths[i]
            # Assume image center principal point (will be refined during loading if image size known)
            # For now store FL
            
            self.items.append({
                "path": str(local_path),
                "pose": poses[i], # c2w 4x4 tensor
                "focal": fl
            })
            
        print(f"[GeNVSDataset] Loaded {len(self.items)} valid frames")
        
        
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True), # Shortest edge -> image_size
            T.CenterCrop(image_size),             # Crop center
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1]
        ])

    def compute_nearest_neighbors(self, poses, k=5):
        """
        Compute nearest neighbors for each pose based on distance and look-at angle.
        poses: [N, 4, 4]
        Returns: [N, k] indices
        """
        N = poses.shape[0]
        centers = poses[:, :3, 3] # [N, 3]
        # Z-forward vector (assuming standard convention, ACE/COLMAP usually +Z is look-at)
        # Note: If ACE provides World-To-Cam, we inverted it earlier? 
        # Wait, load_dataset_ace returns cam-to-world (poses).
        forward_vectors = poses[:, :3, 2] # [N, 3]

        # 1. Distances (N x N)
        dists = torch.cdist(centers, centers) 

        # 2. Angles (N x N)
        # Dot product of normalized vectors
        # normalize just in case
        fwd_norm = torch.nn.functional.normalize(forward_vectors, dim=1)
        dot_prods = torch.mm(fwd_norm, fwd_norm.transpose(0, 1))

        # 3. Combined Cost
        # We want small dist and large dot product (close to 1)
        # Filter: if angle > 60 deg (dot < 0.5), infinite cost
        mask = dot_prods < 0.5 
        dists[mask] = float('inf')

        # Self is huge cost
        dists.fill_diagonal_(float('inf'))

        # Top K nearest
        # Helper to get valid k
        valid_k = min(k, N-1)
        if valid_k < 1:
            return torch.zeros(N, k, dtype=torch.long)
            
        _, indices = torch.topk(dists, k=valid_k, dim=1, largest=False)
        return indices

    def __len__(self):
        return len(self.items)

    def load_view(self, idx):
        item = self.items[idx]
        
        # Load Image
        try:
            img = Image.open(item['path']).convert("RGB")
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            # Return a blank/random standard tensor to avoid crashing
            return None 
            
        W, H = img.size
        
        # Intrinsics
        
        # Intrinsics Modification for Resize + Crop
        # 1. Resize: shortest edge becomes image_size
        scale_factor = self.image_size / min(W, H)
        new_W = int(W * scale_factor)
        new_H = int(H * scale_factor)
        
        # 2. Crop: center (image_size x image_size)
        offset_x = (new_W - self.image_size) // 2
        offset_y = (new_H - self.image_size) // 2
        
        # New K
        K = torch.eye(3)
        K[0, 0] = item['focal'] * scale_factor
        K[1, 1] = item['focal'] * scale_factor
        K[0, 2] = (W / 2.0) * scale_factor - offset_x
        K[1, 2] = (H / 2.0) * scale_factor - offset_y
        
        img_tensor = self.transform(img)
        
        # Apply Scene Normalization to Pose
        pose = item['pose'].clone()
        pose[:3, 3] = (pose[:3, 3] - self.scene_center) * self.scene_scale
        
        # Use original OpenCV (X-Right, Y-Down, Z-Forward) convention
        # No flip needed as we are aligning the entire pipeline to ACE-Zero/CV
        
        return {
            "image": img_tensor, # [3, H, W]
            "pose": pose, # [4, 4] Normalized
            "K": K # [3, 3] Adapted
        }

    def __getitem__(self, idx):
        # Target View (retry if load fails)
        target_view = self.load_view(idx)
        retry_count = 0
        while target_view is None and retry_count < 5:
            idx = random.randint(0, len(self.items) - 1)
            target_view = self.load_view(idx)
            retry_count += 1
        
        if target_view is None:
            # Fallback
            return None # Caught by DataLoader
        
        # Multi-View Selection (Section 5.3)
        # Randomly pick n in {1, 2, 3}
        n = random.randint(1, 3)
        valid_neighbors = self.neighbors[idx]
        
        # Sample n indices from valid neighbors
        source_indices = []
        if len(valid_neighbors) > 0:
            # Shuffle neighbors to pick diverse views
            indices_pool = valid_neighbors.tolist()
            random.shuffle(indices_pool)
            source_indices = indices_pool[:n]
        
        source_images = []
        source_poses = []
        source_Ks = []
        
        for s_idx in source_indices:
            view = self.load_view(s_idx)
            if view:
                source_images.append(view['image'])
                source_poses.append(view['pose'])
                source_Ks.append(view['K'])
        
        while len(source_images) < 3:
            # Repeat existing or use target
            if source_images:
                source_images.append(source_images[0])
                source_poses.append(source_poses[0])
                source_Ks.append(source_Ks[0])
            else:
                source_images.append(target_view['image'])
                source_poses.append(target_view['pose'])
                source_Ks.append(target_view['K'])
            
        return {
            "target_image": target_view['image'],
            "target_pose": target_view['pose'],
            "target_K": target_view['K'],
            "source_images": torch.stack(source_images), # [N_src, 3, H, W]
            "source_poses": torch.stack(source_poses),   # [B_src, 4, 4]
            "source_Ks": torch.stack(source_Ks)        # [B_src, 3, 3]
        }

if __name__ == "__main__":
    # Test dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="demo/test_run_genvs_gpu_max_seed2089")
    args = parser.parse_args()
    
    ds = GenVSDataset(args.root)
    item = ds[0]
    print("Source Pose:", item['source_pose'].shape)
    print("Target Image:", item['target_image'].shape)
    print("Pass!")
