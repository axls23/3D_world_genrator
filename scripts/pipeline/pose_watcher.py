import os
import time
import struct
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def read_poses_final(pose_file):
    poses = []
    if not Path(pose_file).exists():
        return poses
    with open(pose_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 10:
                continue
            entry = {
                'name': Path(tokens[0]).name,
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

class PoseWatcher:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.acezero_out = self.data_dir / "acezero_output"
        self.sparse_dir = self.data_dir / "sparse" / "0"
        self.images_dir = self.data_dir / "images"
        
        self.last_processed_file = None
        self.last_processed_mtime = 0.0
        self.has_new = False
        
        # Initial check
        self.check_for_updates()

    def check_for_updates(self) -> bool:
        """Find the latest poses_iteration*.txt or poses_final.txt and convert it if new."""
        if not self.acezero_out.exists():
            return False
            
        # Look for poses_*.txt (including iteration0_seed* and iteration*) or poses_final.txt
        files = list(self.acezero_out.glob("poses_*.txt"))
        final_pose = self.acezero_out / "poses_final.txt"
        if final_pose.exists() and final_pose not in files:
            files.append(final_pose)
            
        if not files:
            return False
            
        # Get the most recently modified pose file
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        stat = latest_file.stat()
        
        # If it's a new file or has been updated
        if latest_file != self.last_processed_file or stat.st_mtime > self.last_processed_mtime:
            logger.info(f"[PoseWatcher] Found new/updated pose file: {latest_file.name}")
            
            # Read poses
            poses = read_poses_final(latest_file)
            if not poses:
                return False
                
            # Write to sparse/0/ directory
            self.sparse_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Write cameras.bin
            import cv2
            first_img_name = poses[0]['name']
            img_path = self.images_dir / first_img_name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    focal_length = poses[0]['focal_length']
                    with open(self.sparse_dir / "cameras.bin", "wb") as f:
                        f.write(struct.pack("<iiQQ", 1, 1, w, h))
                        f.write(struct.pack("<dddd", focal_length, focal_length, w/2, h/2))
            
            # 2. Write images.bin
            with open(self.sparse_dir / "images.bin", "wb") as f:
                f.write(struct.pack("<Q", len(poses)))
                for i, pose in enumerate(poses):
                    image_id = i + 1
                    camera_id = 1
                    f.write(struct.pack("<IdddddddI", 
                        image_id, 
                        pose['qw'], pose['qx'], pose['qy'], pose['qz'],
                        pose['tx'], pose['ty'], pose['tz'],
                        camera_id
                    ))
                    name_bytes = pose['name'].encode("utf-8") + b"\x00"
                    f.write(name_bytes)
                    f.write(struct.pack("<Q", 0))
                    
            # 3. Ensure points3D.bin exists
            points_bin = self.sparse_dir / "points3D.bin"
            if not points_bin.exists():
                with open(points_bin, "wb") as f:
                    f.write(struct.pack("<Q", 0))
                    
            self.last_processed_file = latest_file
            self.last_processed_mtime = stat.st_mtime
            self.has_new = True
            return True
            
        return False

    def has_new_data(self) -> bool:
        """Called by simple_trainer.py to check if there is new data to load."""
        self.check_for_updates()
        if self.has_new:
            self.has_new = False
            return True
        return False

class StreamingDataManager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.watcher = PoseWatcher(self.data_dir)

    def start_watching(self):
        logger.info(f"[StreamingDataManager] Started watching {self.data_dir}")
        self.watcher.check_for_updates()
