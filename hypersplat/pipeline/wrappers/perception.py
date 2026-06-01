"""
ACE-Zero Pose Estimator Wrapper for 3DGS Pipeline (Docker Support)

This module replaces COLMAP with ACE-Zero (Niantic, ECCV 2024) for camera pose estimation.
It outputs data in COLMAP format compatible with gsplat's simple_trainer.py.

Key Features:
- Lightweight: Runs on 8-12GB VRAM GPUs
- Scalable: Handles thousands of images efficiently
- Docker Support: Runs via `docker exec acezero` on Windows, natively on Linux

Requirements:
    - ACE-Zero repository at scripts/acezero/
    - Docker container `acezero` running (docker compose up -d)
"""

import os
import sys
import subprocess
import shutil
import struct
import logging
import json
import numpy as np
import uuid
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import stat
import time

logger = logging.getLogger(__name__)

# Path to ACE-Zero installation relative to project root
# hypersplat/pipeline/wrappers/perception.py -> wrappers -> pipeline -> hypersplat -> root -> scripts -> acezero
# Wait, ACEZERO_ROOT is scripts/acezero as determined earlier.
# And project root is .parent.parent.parent.parent
ACEZERO_ROOT = Path(__file__).parent.parent.parent.parent / "scripts" / "acezero"

class ACEZeroPoseEstimator:
    """Wrapper for ACE-Zero pose estimation."""
    
    # Supported depth models: 'depth_anything' (fast, ~30ms), 'zoedepth' (metric, ~200ms), 'midas' (lightweight)
    DEPTH_MODELS = ['depth_anything', 'zoedepth', 'midas']
    
    def __init__(self, output_dir: Path, acezero_root: Optional[Path] = None, use_depth_init: bool = True,
                 pose_refinement: str = "mlp", pose_refinement_wait: int = 5000, 
                 pose_refinement_lr: float = 0.001, refinement_ortho: str = "gram-schmidt",
                 quality_mode: str = "balanced", min_confidence: int = 1000,
                 depth_model: str = "depth_anything", use_docker: bool = False,
                 use_hybrid: bool = True):
        self.output_dir = Path(output_dir).resolve()  # Ensure absolute path
        self.acezero_root = acezero_root or ACEZERO_ROOT
        self.use_depth_init = use_depth_init
        self.use_docker = use_docker
        self.use_hybrid = use_hybrid  # Single-shot hybrid mode (SIFT init + single ACE pass)
        
        # Pose refinement settings (significantly improves pose quality for 3DGS)
        # Options: 'mlp' (best quality), 'naive' (direct backprop), 'none'
        self.pose_refinement = pose_refinement
        self.pose_refinement_wait = pose_refinement_wait  # Freeze poses for first N iters (stability)
        self.pose_refinement_lr = pose_refinement_lr
        self.refinement_ortho = refinement_ortho  # 'gram-schmidt' or 'procrustes'
        
        # Adaptive ACE-Zero settings
        self.quality_mode = quality_mode  # 'fast', 'balanced', 'quality'
        self.min_confidence = min_confidence  # Filter poses below this confidence
        self.video_info = {}  # Populated by _analyze_video()
        
        # New adaptive options defaults
        self.cooldown_iterations = 5000
        self.cooldown_threshold = 0.7
        self.aug_rotation = 15
        self.registration_threshold = 0.99
        
        # Depth model selection: 'depth_anything' (default, fast), 'zoedepth' (metric), 'midas' (lightweight)
        self.depth_model = depth_model if depth_model in self.DEPTH_MODELS else 'depth_anything'
        logger.info(f"Depth estimator: {self.depth_model}, Docker routing: {self.use_docker}")
        
        self.images_dir = self.output_dir / "images"
        self.sparse_dir = self.output_dir / "sparse" / "0"
        self.acezero_output = self.output_dir / "acezero_output"
        self.streaming_status_file = self.output_dir / "streaming_status.json"
        
        logger.info(f"ACEZeroPoseEstimator init: output={self.output_dir}, acezero_root={self.acezero_root}")
        
        self.image_width = 0
        self.image_height = 0
        self.focal_length = 0.0
        
        self.poses: Dict[str, np.ndarray] = {}  # image_name -> 4x4 c2w matrix
        self.points: Optional[np.ndarray] = None
        
        # Detection of Container/Linux environment
        self.is_linux = sys.platform != 'win32'
        self.is_docker = os.path.exists('/.dockerenv')
        
        # Python path depends on environment
        # The lean Dockerfile installs deps into the base PyTorch image,
        # so the correct Python is /opt/conda/bin/python (NOT ace0 env).
        self.use_wsl = False
        if self.is_docker:
            # Running inside the container itself
            self.docker_python = os.environ.get("HYPERSPLAT_PYTHON", "/opt/conda/bin/python")
        elif self.is_linux:
            # Native Linux (bare metal, no docker)
            self.docker_python = os.environ.get("HYPERSPLAT_PYTHON", "python")
        else:
            if self.use_docker:
                # Windows host → commands routed via `docker exec acezero`
                self.docker_python = "/opt/conda/bin/python"
            else:
                # Windows host → commands routed via WSL
                self.use_wsl = True
                self.docker_python = "/home/axls23/miniconda3/envs/ace0/bin/python"

    @staticmethod
    def _remove_readonly(func, path, _):
        """Helper for shutil.rmtree to handle read-only files on Windows."""
        import os
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def _write_streaming_status(self, iteration: int, pose_file: str, is_running: bool, total_images: int = 0):
        """Write streaming status for the pose watcher to read."""
        status = {
            "iteration": iteration,
            "pose_file": pose_file,
            "is_running": is_running,
            "total_images": total_images,
            "timestamp": time.time(),
            "sparse_dir": str(self.sparse_dir),
            "images_dir": str(self.images_dir)
        }
        try:
            with open(self.streaming_status_file, "w") as f:
                json.dump(status, f)
        except Exception as e:
            logger.warning(f"Failed to write streaming status: {e}")
    
    def _analyze_video(self, video_path: str) -> dict:
        """Analyze video to determine optimal ACE-Zero parameters."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"Cannot open video for analysis: {video_path}")
            return {'fps': 30, 'frame_count': 100, 'avg_motion': 15, 'resolution': 640, 'width': 640, 'height': 480}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Sample frames to estimate motion/blur (Optimized)
        # Instead of random seeking (slow), read a contiguous chunk from middle
        start_frame = frame_count // 3
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        prev_gray = None
        frames_to_read = 30
        
        for _ in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_gray = gray
        
        cap.release()
        
        avg_motion = np.mean(motion_scores) if motion_scores else 15
        
        info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'avg_motion': avg_motion,
            'resolution': max(width, height),
        }
        
        logger.info(f"Video Analysis: {duration:.1f}s, {frame_count} frames, motion={avg_motion:.1f}")
        return info
    
    def _compute_adaptive_params(self):
        """Compute ACE-Zero parameters based on video analysis and quality mode."""
        avg_motion = self.video_info.get('avg_motion', 15)
        frame_count = self.video_info.get('frame_count', 100)
        duration = self.video_info.get('duration', 10)
        
        # === Motion-based adjustments ===
        if avg_motion > 30:
            self.repro_loss_soft_clamp = 30
            self.registration_confidence = 1500
            self.adaptive_try_seeds = 10
        elif avg_motion > 15:
            self.repro_loss_soft_clamp = 40
            self.registration_confidence = 1000
            self.adaptive_try_seeds = 7
        else:
            self.repro_loss_soft_clamp = 50
            self.registration_confidence = 500
            self.adaptive_try_seeds = 5
        
        # === Quality mode adjustments ===
        if self.quality_mode == "fast":
            self.seed_iterations = 3000   # Coarse seed (V2 optimized)
            self.refit_iterations = 10000  # Lean final mapping
            self.learning_rate_max = 0.006 # Higher LR for faster convergence
            self.cooldown_threshold = 0.75
            self.cooldown_iterations = 2000
            self.aug_rotation = 5
            self.registration_threshold = 0.95
        elif self.quality_mode == "quality":
            self.seed_iterations = 10000  # High-quality seed
            self.refit_iterations = 50000 # Exhaustive mapping
            self.learning_rate_max = 0.003
            self.cooldown_threshold = 0.6
            self.cooldown_iterations = 5000
            self.aug_rotation = 15
            self.registration_threshold = 0.99
        else: # balanced
            self.seed_iterations = 5000
            self.refit_iterations = 18000
            self.learning_rate_max = 0.004
            self.cooldown_threshold = 0.7
            self.cooldown_iterations = 4000
            self.aug_rotation = 10
            self.registration_threshold = 0.99
        
        # Scale iterations by expected frames
        expected_frames = getattr(self, 'num_frames', frame_count // 10)
        # The original `base_iterations` logic is now replaced by `seed_iterations` and `refit_iterations`
        # The `adaptive_iterations` variable is no longer directly used for ACE-Zero's main loop,
        # as ACE-Zero now uses `seed_iterations` and `refit_iterations` for its two main phases.
        # Keeping `adaptive_iterations` for potential future use or if other parts of the code still reference it.
        self.adaptive_iterations = self.seed_iterations + self.refit_iterations # Sum of seed and refit iterations
        self.adaptive_iterations = min(100000, self.adaptive_iterations)
        
        # === Hybrid mode parameters ===
        if self.use_hybrid:
            if self.quality_mode == "fast":
                self.hybrid_train_iterations = 8000
                self.hybrid_pose_wait = 1000
            elif self.quality_mode == "quality":
                self.hybrid_train_iterations = 25000
                self.hybrid_pose_wait = 3000
            else:  # balanced
                self.hybrid_train_iterations = 15000
                self.hybrid_pose_wait = 2000
        
        logger.info(f"Adaptive Params [mode={self.quality_mode}, motion={avg_motion:.1f}]: "
                    f"repro_clamp={self.repro_loss_soft_clamp}, reg_conf={self.registration_confidence}, "
                    f"seeds={self.adaptive_try_seeds}, iters={self.adaptive_iterations}")
        if self.use_hybrid:
            logger.info(f"Hybrid Mode: train_iters={self.hybrid_train_iterations}, pose_wait={self.hybrid_pose_wait}")
    
    def _filter_poses(self) -> tuple:
        """Filter low-confidence poses from self.poses dict.
        
        This is the SINGLE authoritative filter in the pipeline.
        It removes entries from self.poses so that write_colmap_format()
        only outputs good poses to the binary files consumed by 3DGS.
        
        Returns: (good_count, bad_count, filtered_path_or_None)
        """
        pose_file = self.acezero_output / "poses_final.txt"
        if not pose_file.exists():
            return (0, 0, None)
        
        # Build a set of image names that fail the confidence check
        bad_names = set()
        good_lines = []
        
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10:
                    conf = float(parts[-1])
                    if conf < self.min_confidence:
                        bad_names.add(Path(parts[0]).name)
                    else:
                        good_lines.append(line)
                else:
                    good_lines.append(line)
        
        # Actually remove bad poses from self.poses dict
        if bad_names:
            before_count = len(self.poses)
            for name in bad_names:
                self.poses.pop(name, None)
            after_count = len(self.poses)
            removed = before_count - after_count
            
            # Write filtered text file for debugging/provenance
            filtered_path = self.acezero_output / "poses_final_filtered.txt"
            with open(filtered_path, 'w') as f:
                f.writelines(good_lines)
            logger.info(f"Pose Filtering: {after_count} kept, {removed} removed (conf < {self.min_confidence})")
            return (after_count, removed, str(filtered_path))
        
        return (len(self.poses), 0, None)
        
    def process_video(self, video_path: str, fps: float = 2.0, streaming: bool = False) -> bool:
        """Run the full ACE-Zero pipeline: Frames -> Poses -> COLMAP Format"""
        try:
            self._setup_directories()
            self.fps = fps  # Store for parameter calculation
            
            # 0. Analyze video for adaptive parameters
            self.video_info = self._analyze_video(video_path)
            
            # 1. Extract Frames
            num_frames = self.extract_frames(video_path, fps)
            if num_frames == 0:
                logger.error("No frames extracted.")
                return False
            self.num_frames = num_frames  # Store for parameter calculation
            
            # 1.1 Compute adaptive parameters based on video analysis
            self._compute_adaptive_params()
            
            # Note: Depth estimation is now handled by ACE-Zero using Depth Anything V2
            # (via our modified dataset_io.py that gets synced to WSL before running)
            # This is 6x faster than ZoeDepth (~30ms vs ~200ms per frame)
                
            # 2. Run ACE-Zero (Train + Register) - iterations calculated based on frame count
            success = self.run_ace_zero()
            if not success:
                logger.error("ACE-Zero training/reg failed.")
                return False
            
            # 2.1 Filter low-confidence poses
            good, bad, filtered = self._filter_poses()
            if bad > 0:
                logger.info(f"Removed {bad} low-confidence poses")
                
            # 3. Write COLMAP Format
            self.write_colmap_format()
            
            # 4. Generate Points (Optional)
            self.generate_initial_points()
            
            # 5. Downsample for Training (2x, 4x)
            self.create_downsampled_images(factors=[2, 4])
            
            # 6. Export Config
            self.dump_config(str(self.acezero_output / "cfg_acezero.yml"))
            
            return True
        except Exception as e:
            logger.error(f"ACE-Zero Pipeline Error: {e}")
            return False

    def _setup_directories(self):
        for p in [self.output_dir, self.images_dir, self.sparse_dir, self.acezero_output]:
            p.mkdir(parents=True, exist_ok=True)
    
    def _precompute_depths(self):
        """Pre-compute depth maps for all images to free VRAM for ACE training.
        
        This runs ZoeDepth once, saves depths as .npy files, then the model
        is unloaded. ACE-Zero will load cached depths instead of recomputing.
        """
        if sys.platform != 'win32' and not self.is_docker:
            # On Linux native
            from hypersplat.pipeline.steps.depth import precompute_depths
            precompute_depths(self.images_dir)
            return
        
        logger.info("Pre-computing depths via Docker (frees ~2GB VRAM)...")
        
        docker_images_dir = self._windows_to_docker_path(str(self.images_dir.resolve()))
        docker_acezero = self._windows_to_docker_path(str(self.acezero_root.resolve()))
        
        # depth_model options: 'depth_anything' (fast, ~30ms), 'zoedepth' (metric, ~200ms), 'midas' (~100ms)
        depth_model_choice = self.depth_model
        precompute_cmd = f'''
cd {docker_acezero} && {self.docker_python} -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
from pathlib import Path
import torch

images_dir = Path('{docker_images_dir}')
depths_dir = images_dir.parent / 'depths'
depths_dir.mkdir(exist_ok=True)

image_files = sorted(images_dir.glob('*.jpg'))

if not image_files:
    print('No images found')
    exit(0)

# User-selected depth model: {depth_model_choice}
requested_model = '{depth_model_choice}'
model = None
model_type = None

def load_depth_anything():
    global model, model_type
    from depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {{'vits': {{'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}}}
    model = DepthAnythingV2(**model_configs['vits'])
    weights_path = Path.home() / '.cache' / 'depth_anything_v2' / 'depth_anything_v2_vits.pth'
    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth'
        print('Downloading Depth Anything V2 weights...')
        torch.hub.download_url_to_file(url, str(weights_path))
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model = model.to('cuda').eval()
    model_type = 'depth_anything'
    print(f'Depth Anything V2 loaded (~30ms/frame)')

def load_zoedepth():
    global model, model_type
    import dataset_io
    model = dataset_io.get_depth_model()
    model_type = 'zoedepth'
    print(f'ZoeDepth loaded (~200ms/frame, metric depth)')

def load_midas():
    global model, model_type
    model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', trust_repo=True)
    model = model.to('cuda').eval()
    model_type = 'midas'
    print(f'MiDaS loaded (~100ms/frame)')

# Load requested model with fallback chain
print(f'Loading {{requested_model}} for {{len(image_files)}} images...')
print(f'Depths will be saved to: {{depths_dir}}')

try:
    if requested_model == 'depth_anything':
        load_depth_anything()
    elif requested_model == 'zoedepth':
        load_zoedepth()
    elif requested_model == 'midas':
        load_midas()
    else:
        load_depth_anything()  # Default
except Exception as e:
    print(f'{{requested_model}} failed: {{e}}, trying fallback...')
    try:
        if requested_model != 'zoedepth':
            load_zoedepth()
        else:
            load_depth_anything()
    except Exception as e2:
        print(f'Fallback failed: {{e2}}, using MiDaS...')
        load_midas()

# MiDaS transform (only needed for MiDaS)
midas_transform = None
if model_type == 'midas':
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    midas_transform = midas_transforms.dpt_transform

with torch.no_grad():
    for i, img_path in enumerate(image_files):
        depth_path = depths_dir / f'{{img_path.stem}}.npy'
        if depth_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if model_type == 'depth_anything':
            depth = model.infer_image(img_rgb)
        elif model_type == 'zoedepth':
            import dataset_io
            depth = dataset_io.estimate_depth(model, img_rgb)
        elif model_type == 'midas':
            input_batch = midas_transform(img_rgb).to('cuda')
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
            depth = 1.0 / (prediction.cpu().numpy() + 1e-6)  # MiDaS outputs inverse depth
        
        np.save(str(depth_path), depth.astype(np.float32))
        
        if (i + 1) % 10 == 0:
            print(f'Processed {{i + 1}}/{{len(image_files)}}')

print(f'Unloading {{model_type}} model...')
del model
torch.cuda.empty_cache()
print('Done! VRAM freed for ACE training.')
"
'''
        try:
            self._run_docker_command(precompute_cmd.strip(), timeout=600)
            logger.info("Depth pre-computation complete.")
        except Exception as e:
            logger.warning(f"Depth pre-computation failed: {e}. Will use inline ZoeDepth.")
            
    def _run_docker_command(self, cmd_str: str, check: bool = True, capture_output: bool = False, timeout: int = 7200) -> subprocess.CompletedProcess:
        """Helper to run commands natively in Docker.
        
        Args:
            timeout: Maximum time in seconds to wait (default 7200 = 2 hours)
        """
        if not self.is_docker:
            if getattr(self, 'use_wsl', False):
                wsl_root = self._windows_to_docker_path(str(self.acezero_root))
                full_cmd = f"cd {wsl_root} && {cmd_str}"
                logger.info(f"WSL Exec: {full_cmd[:150]}...")
                try:
                    cmd_list = ["wsl", "-d", "FedoraLinux-43", "bash", "-c", full_cmd]
                    result = subprocess.run(
                        cmd_list,
                        check=False,
                        stdout=subprocess.PIPE if capture_output else None,
                        stderr=subprocess.PIPE if capture_output else None,
                        timeout=timeout,
                        text=True if capture_output else False
                    )
                except subprocess.TimeoutExpired:
                    logger.error(f"WSL Command timed out after {timeout} seconds.")
                    if check: raise
                    return subprocess.CompletedProcess(["wsl"], 124, stdout="", stderr="")
                    
                if result.returncode != 0 and check:
                    logger.error(f"WSL Command Failed (Return Code: {result.returncode}).")
                    if capture_output:
                        logger.error(f"Output: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, cmd_str)
                return subprocess.CompletedProcess(result.args, result.returncode, stdout=result.stdout if capture_output else "", stderr=result.stderr if capture_output else "")
            else:
                # We are running on host but invoking the Docker container `acezero`
                logger.info(f"Docker Exec: {cmd_str[:150]}...")
                try:
                    cmd_list = ["docker", "exec", "acezero", "bash", "-c", cmd_str]
                    result = subprocess.run(
                        cmd_list,
                        check=False,
                        stdout=subprocess.PIPE if capture_output else None,
                        stderr=subprocess.PIPE if capture_output else None,
                        timeout=timeout,
                        text=True if capture_output else False
                    )
                except subprocess.TimeoutExpired:
                    logger.error(f"Docker Command timed out after {timeout} seconds.")
                    if check: raise
                    return subprocess.CompletedProcess(["docker"], 124, stdout="", stderr="")
                    
                if result.returncode != 0 and check:
                    logger.error(f"Docker Command Failed (Return Code: {result.returncode}).")
                    if capture_output:
                        logger.error(f"Output: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, cmd_str)
                return subprocess.CompletedProcess(result.args, result.returncode, stdout=result.stdout if capture_output else "", stderr=result.stderr if capture_output else "")

        # Inline Linux execution if already inside docker
        full_cmd = f"cd {self.acezero_root} && {cmd_str}"
        logger.info(f"Docker Exec (Native Linux): {cmd_str[:150]}...")
        
        try:
            result = subprocess.run(
                ["bash", "-c", full_cmd],
                check=False,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                timeout=timeout,
                text=True if capture_output else False
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds.")
            if check: raise
            return subprocess.CompletedProcess(["bash"], 124, stdout="", stderr="")
            
        if result.returncode != 0 and check:
            logger.error(f"Command Failed (Return Code: {result.returncode}).")
            if capture_output:
                logger.error(f"Output: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd_str)
            
        return subprocess.CompletedProcess(result.args, result.returncode, stdout=result.stdout if capture_output else "", stderr=result.stderr if capture_output else "")

    def extract_frames(self, video_path: str, fps: float = 2.0) -> int:
        video_path = Path(video_path)
        if not video_path.exists(): raise FileNotFoundError(f"Video not found: {video_path}")
        
        
        # Check if images already exist (Resume mode)
        if self.images_dir.exists():
            existing_frames = sorted(self.images_dir.glob("frame_*.jpg"))
            if existing_frames and len(existing_frames) > 0:
                logger.info(f"Found {len(existing_frames)} existing frames at {self.images_dir}. Skipping extraction.")
                
                # Update image info from first frame
                import cv2
                sample = cv2.imread(str(existing_frames[0]))
                self.image_height, self.image_width = sample.shape[:2]
                self.focal_length = self.image_width / (2 * np.tan(np.radians(35)))
                
                return len(existing_frames)
            
            # If exists but empty or invalid, clean up
            for attempt in range(3):
                try:
                    shutil.rmtree(self.images_dir, onerror=self._remove_readonly)
                    break
                except Exception as e:
                     if attempt == 2:
                         logger.warning(f"Failed to remove images dir: {e}. Trying to proceed anyway...")
                     time.sleep(1.0)
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        output_pattern = str(self.images_dir / "frame_%05d.jpg")
        # Scale to max 640 width for optimization (ACE-Zero works well at low res)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vf", f"fps={fps},scale='min(640,iw)':-1", "-q:v", "2", output_pattern]
        
        logger.info(f"Extracting frames at {fps} FPS...")
        if subprocess.run(cmd).returncode != 0: raise RuntimeError("FFmpeg extraction failed")
        
        frame_files = sorted(self.images_dir.glob("*.jpg"))
        if frame_files:
            import cv2
            sample = cv2.imread(str(frame_files[0]))
            self.image_height, self.image_width = sample.shape[:2]
            self.focal_length = self.image_width / (2 * np.tan(np.radians(35))) # ~70 deg HFOV
            
        logger.info(f"Extracted {len(frame_files)} frames ({self.image_width}x{self.image_height})")
        return len(frame_files)
    
    def run_ace_zero(self, max_iterations: int = None) -> bool:
        """Run ACE-Zero with parameters calculated from frame count."""
        if not (self.acezero_root / "train_ace.py").exists():
            logger.error("ACE-Zero not found")
            return False
        
        # Calculate optimal parameters based on frame count
        num_frames = getattr(self, 'num_frames', 30)  # Default to 30 if not set
        
        # ===========================================
        # Individual Complexity-Based Scaling
        # Each parameter scales on its own curve
        # ===========================================
        
        # Seed Parallel Workers: Aggressively increased for 24-thread CPU
        # User confirmed wanting to maximize CPU usage for initialization speed.
        # Balancing against VRAM: 6GB VRAM can handle ~6-8 concurrent tiny networks
        self.seed_parallel_workers = 8

        # Seeds: With more parallel workers, we can afford to try more seeds
        # Complexity: O(seeds) but parallelized
        # Scale: sqrt(frames/5), clamped to [4, 12]
        self.try_seeds = max(4, min(12, int(4 + (num_frames / 30) ** 0.5)))
        
        # Iterations: More frames = more images to register per round
        # Complexity: O(iterations * frames) - linear scaling
        # Scale: 5 base + log2(frames), clamped to [5, 15]
        import math
        calculated_iters = max(5, min(15, int(5 + math.log2(max(1, num_frames)))))
        self.max_iterations = calculated_iters if calculated_iters else 10  # Ensure never None
        
        # Data Workers: CPU parallelism for data loading
        # User has 24 threads -> can safely increase cap
        # Complexity: Minor CPU overhead, scales memory slightly
        # Scale: 4 base + frames/50, clamped to [4, 16]
        self.num_data_workers = max(4, min(16, int(4 + num_frames / 50)))
        
        # Head Blocks: Network depth = map capacity
        # Complexity: O(blocks * params) - quadratic VRAM impact
        # Scale: 1 for most cases (6GB VRAM), 2 only for very large datasets
        self.num_head_blocks = 1 if num_frames < 150 else 2
        
        # Training Buffer CPU: Offload training buffer to RAM
        # Complexity: ~2x slower loading but saves ~1-2GB VRAM
        # Scale: Force GPU (False) per user request to maximize GPU usage
        self.training_buffer_cpu = False

        # [OPTIMIZATION] Fast / Streaming Mode Overrides
        if self.quality_mode == "fast":
            # Minimize Overhead
            self.try_seeds = 1             # Single shot mapping (trust first result)
            self.seed_parallel_workers = 1 # No need for parallelism
            self.training_buffer_cpu = True # Move buffer to RAM to free VRAM for 3DGS
            logger.info(f"[Optimization] Fast/Streaming Mode: Reduced overhead (Seeds={self.try_seeds}, CPU Buffer={self.training_buffer_cpu})")
            
        # Override with explicit max_iterations if provided
        if max_iterations is not None:
            self.max_iterations = max_iterations
            
        logger.info(f"ACE-Zero params [frames={num_frames}]: seeds={self.try_seeds}, "
                    f"iters={self.max_iterations}, seed_workers={self.seed_parallel_workers}, "
                    f"data_workers={self.num_data_workers}, heads={self.num_head_blocks}, "
                    f"cpu_buffer={self.training_buffer_cpu}, refinement={self.pose_refinement}")

        # [DEBUG] Verify Fast Mode Parameters
        logger.info(f"ACE-Quality Check: Mode={self.quality_mode}, LR_Max={getattr(self, 'learning_rate_max', 'N/A')}, SeedIters={getattr(self, 'seed_iterations', 'N/A')}")
            
        images_glob = str(self.images_dir.resolve() / "*.jpg")
        output_dir = str(self.acezero_output.resolve())
        
        # Select script based on hybrid mode
        if self.use_hybrid:
            acezero_script = str(self.acezero_root.resolve() / "ace_zero_hybrid.py")
            logger.info("[HYBRID MODE] Using single-shot pose initialization + single ACE pass")
        else:
            acezero_script = str(self.acezero_root.resolve() / "ace_zero.py")
        
        if sys.platform == 'win32':
            # Verify Docker/WSL GPU access before running
            try:
                env_type = "WSL" if getattr(self, "use_wsl", False) else "Docker"
                logger.info(f"Verifying {env_type} GPU access...")
                gpu_check = self._run_docker_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", 
                                                  check=False, timeout=10)
                if gpu_check.returncode == 0:
                    logger.info(f"{env_type} GPU access verified.")
                else:
                    logger.warning(f"{env_type} GPU check failed. ACE-Zero may fail or hang on GPU operations.")
            except Exception as e:
                logger.warning(f"Could not verify {env_type} GPU access: {e}. Continuing anyway...")
            
            return self._run_ace_zero_via_docker(images_glob, output_dir, acezero_script, max_iterations)
        else:
             return self._run_ace_zero_native(images_glob, output_dir, acezero_script, max_iterations)

    def _windows_to_docker_path(self, windows_path: str) -> str:
        p = Path(windows_path).resolve()
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        p_str = str(p)
        root_str = str(project_root)
        if p_str.lower().startswith(root_str.lower()):
            rel = p_str[len(root_str):].lstrip('\\/')
            rel_forward = rel.replace('\\', '/')
            return f"/workspace/{rel_forward}" if rel_forward else "/workspace"
        if self.is_docker:
            return p_str.replace('\\', '/')
        return p_str.replace('\\', '/')

    def _docker_to_windows_path(self, docker_path: str) -> str:
        p = str(docker_path).replace('\\', '/')
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        if p.startswith("/workspace"):
            rel = p[len("/workspace"):].lstrip('/')
            return str(project_root / rel)
        return str(Path(docker_path).resolve())
    
    def _run_ace_zero_via_docker(self, images_glob: str, output_dir: str, acezero_script: str, max_iterations: int) -> bool:
        docker_images_src = self._windows_to_docker_path(str(self.images_dir.resolve()))
        output_dir_abs = str(Path(output_dir).resolve())
        docker_final_output = self._windows_to_docker_path(output_dir_abs)
        docker_script = self._windows_to_docker_path(acezero_script)
        
        logger.debug(f"Docker paths: images_src={docker_images_src}, final_output={docker_final_output}")

        # Ensure output directory exists (in docker)
        setup_cmd = f"mkdir -p {docker_final_output}"
        
        # 2. Run ACE-Zero (uses Depth Anything V2 via modified dataset_io.py)
        logger.info(f"ACE-Zero will use Depth Anything V2 for depth estimation (~30ms/frame)")
        if self.use_hybrid:
            # Hybrid mode: single-pass command
            ace_cmd = (f'{self.docker_python} -u {docker_script} '
                       f'"{docker_images_src}/*.jpg" {docker_final_output} '
                       f'--hybrid_train_iterations {getattr(self, "hybrid_train_iterations", 15000)} '
                       f'--hybrid_pose_wait {getattr(self, "hybrid_pose_wait", 2000)} '
                       f'--learning_rate_max {getattr(self, "learning_rate_max", 0.003)} '
                       f'--training_buffer_cpu {self.training_buffer_cpu} '
                       f'--num_data_workers {self.num_data_workers} '
                       f'--num_head_blocks {self.num_head_blocks} '
                       f'--refinement {self.pose_refinement} '
                       f'--refinement_ortho {self.refinement_ortho} '
                       f'--pose_refinement_lr {self.pose_refinement_lr} '
                       f'--cooldown_iterations {self.cooldown_iterations} '
                       f'--cooldown_threshold {self.cooldown_threshold} '
                       f'--aug_rotation {self.aug_rotation} '
                       f'--image_resolution {getattr(self, "image_resolution", 480)} '
                       f'--registration_confidence {getattr(self, "registration_confidence", 500)}')
        else:
            ace_cmd = f'{self.docker_python} -u {docker_script} "{docker_images_src}/*.jpg" {docker_final_output} --iterations_max {getattr(self, "max_iterations", 10)} --seed_iterations {self.seed_iterations} --refit_iterations {self.refit_iterations} --learning_rate_max {getattr(self, "learning_rate_max", 0.003)} --try_seeds {self.try_seeds} --seed_parallel_workers {self.seed_parallel_workers} --training_buffer_cpu {self.training_buffer_cpu} --num_data_workers {self.num_data_workers} --num_head_blocks {self.num_head_blocks} --refinement {self.pose_refinement} --refinement_ortho {self.refinement_ortho} --pose_refinement_wait {self.pose_refinement_wait} --pose_refinement_lr {self.pose_refinement_lr} --cooldown_iterations {self.cooldown_iterations} --cooldown_threshold {self.cooldown_threshold} --aug_rotation {self.aug_rotation} --registration_threshold {self.registration_threshold}'
        
        full_cmd = f"{setup_cmd} && {ace_cmd}"
        
        try:
            self._run_docker_command(full_cmd, check=True, timeout=7200)
            logger.info("ACE-Zero (Docker) completed.")
            self.refine_poses_native() 
            return self._parse_acezero_output()
        except subprocess.CalledProcessError as e:
            logger.error(f"ACE-Zero (Docker) failed: {e}")
            return False
        except subprocess.TimeoutExpired as e:
            logger.error(f"ACE-Zero (Docker) timed out: {e}")
            return False
        except Exception as e:
            logger.error(f"ACE-Zero (Docker) failed: {e}")
            return False

    def _run_ace_zero_native(self, images_glob: str, output_dir: str, acezero_script: str, max_iterations: int) -> bool:
        """Run ACE-Zero natively on Linux (Docker or bare metal)."""
        # Ensure all parameters have valid defaults
        iters = max_iterations if max_iterations else getattr(self, 'max_iterations', 10)
        try_seeds = getattr(self, 'try_seeds', 3)
        seed_workers = getattr(self, 'seed_parallel_workers', 1)
        training_buffer = getattr(self, 'training_buffer_cpu', True)
        data_workers = getattr(self, 'num_data_workers', 2)
        head_blocks = getattr(self, 'num_head_blocks', 1)
        refinement = getattr(self, 'pose_refinement', 'mlp')
        refinement_ortho = getattr(self, 'refinement_ortho', 'gram-schmidt')
        refinement_wait = getattr(self, 'pose_refinement_wait', 5000)
        refinement_lr = getattr(self, 'pose_refinement_lr', 0.001)
        
        if self.use_hybrid:
            cmd = [
                sys.executable, "-u", acezero_script,
                images_glob, output_dir,
                "--hybrid_train_iterations", str(getattr(self, 'hybrid_train_iterations', 15000)),
                "--hybrid_pose_wait", str(getattr(self, 'hybrid_pose_wait', 2000)),
                "--learning_rate_max", str(getattr(self, 'learning_rate_max', 0.003)),
                "--training_buffer_cpu", str(training_buffer),
                "--num_data_workers", str(data_workers),
                "--num_head_blocks", str(head_blocks),
                "--refinement", str(refinement),
                "--refinement_ortho", str(refinement_ortho),
                "--pose_refinement_lr", str(refinement_lr),
                "--cooldown_iterations", str(getattr(self, 'cooldown_iterations', 5000)),
                "--cooldown_threshold", str(getattr(self, 'cooldown_threshold', 0.7)),
                "--aug_rotation", str(getattr(self, 'aug_rotation', 15)),
                "--image_resolution", str(getattr(self, 'image_resolution', 480)),
                "--registration_confidence", str(getattr(self, 'registration_confidence', 500)),
            ]
        else:
            cmd = [
                sys.executable, "-u", acezero_script,
                images_glob, output_dir,
                "--iterations_max", str(iters),
                "--try_seeds", str(try_seeds),
                "--seed_parallel_workers", str(seed_workers),
                "--training_buffer_cpu", str(training_buffer),
                "--num_data_workers", str(data_workers),
                "--num_head_blocks", str(head_blocks),
                "--refinement", str(refinement),
                "--refinement_ortho", str(refinement_ortho),
                "--pose_refinement_wait", str(refinement_wait),
                "--pose_refinement_lr", str(refinement_lr),
                "--cooldown_iterations", str(getattr(self, 'cooldown_iterations', 5000)),
                "--cooldown_threshold", str(getattr(self, 'cooldown_threshold', 0.7)),
                "--aug_rotation", str(getattr(self, 'aug_rotation', 15)),
                "--registration_threshold", str(getattr(self, 'registration_threshold', 0.99)),
            ]
        
        logger.info(f"[Native Linux] Running ACE-Zero: iters={iters}, seeds={try_seeds}, refinement={refinement}")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.acezero_root) + os.pathsep + env.get("PYTHONPATH", "")
        
        try:
            subprocess.run(cmd, cwd=str(self.acezero_root), env=env, check=True)
            self.refine_poses_native()
            return self._parse_acezero_output()
        except subprocess.CalledProcessError as e:
            logger.error(f"ACE-Zero native execution failed: {e}")
            return False
        except Exception as e:
            logger.error(f"ACE-Zero native unexpected error: {e}")
            return False

    def _parse_acezero_output(self) -> bool:
        from scipy.spatial.transform import Rotation
        # Try single file first
        fpath = self.acezero_output / "poses_final.txt"
        if fpath.exists():
            try:
                with open(fpath, 'r') as f:
                    for line in f:
                        tokens = line.split()
                        if len(tokens) == 10:
                            q = [float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[1])] # x,y,z,w
                            t = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
                            R = Rotation.from_quat(q).as_matrix()
                            w2c = np.eye(4); w2c[:3,:3] = R; w2c[:3,3] = t
                            self.poses[Path(tokens[0]).name] = np.linalg.inv(w2c)
                return len(self.poses) > 0
            except Exception as e: logger.error(f"Parse error: {e}")

        # Fallback to directory mode
        pose_files = sorted((self.acezero_output / "poses_ace_network").glob("*.txt")) or sorted(self.acezero_output.glob("*.txt"))
        pose_files = [f for f in pose_files if f.name != "poses_final.txt"]
        
        for pf, img in zip(pose_files, sorted(self.images_dir.glob("*.jpg"))):
            try:
                p = np.loadtxt(pf).reshape(4,4)
                self.poses[img.name] = p
            except: pass
            
        return len(self.poses) > 0

    def generate_initial_points(self) -> bool:
        """Generate initial 3D points from depth maps for better 3DGS convergence."""
        if not self.use_depth_init: return False
        if sys.platform == 'win32':
            return self._generate_points_via_docker()
        else:
            # Native Linux / Docker — run point generation directly
            return self._generate_points_native()

    def _generate_points_via_docker(self) -> bool:
        script = self._windows_to_docker_path(str(self.acezero_root / "generate_points_wsl.py"))
        poses = self._windows_to_docker_path(str(self.acezero_output / "poses_final.txt"))
        imgs = self._windows_to_docker_path(str(self.images_dir))
        out = self._windows_to_docker_path(str(self.sparse_dir / "points3D.bin"))
        
        cmd = f"{self.docker_python} -u {script} '{poses}' '{imgs}' '{out}'"
        try:
             res = self._run_docker_command(cmd, check=False, capture_output=True)
             if (self.sparse_dir / "points3D.bin").stat().st_size > 1000: return True
             logger.error(f"Docker Point Gen failed. Output: {res.stdout}")
        except Exception as e: logger.error(f"Docker Point Gen Error: {e}")
        return False

    def _generate_points_native(self) -> bool:
        """Generate initial 3D points on native Linux (Docker or bare metal).
        
        Same logic as the WSL variant, but calls the script directly — 
        no path translation needed since we're already on Linux.
        """
        script = self.acezero_root / "generate_points_wsl.py"
        if not script.exists():
            logger.warning(f"Point generation script not found: {script}")
            return False
        
        poses = str(self.acezero_output / "poses_final.txt")
        imgs = str(self.images_dir)
        out = str(self.sparse_dir / "points3D.bin")
        
        cmd = [sys.executable, "-u", str(script), poses, imgs, out]
        try:
            subprocess.run(cmd, check=True, cwd=str(self.acezero_root), timeout=600)
            if (self.sparse_dir / "points3D.bin").stat().st_size > 1000:
                return True
            logger.error("Native Point Gen: output too small")
        except Exception as e:
            logger.error(f"Native Point Gen Error: {e}")
        return False

    def create_downsampled_images(self, factors: List[int] = [2, 4]) -> bool:
        from PIL import Image
        images = sorted(self.images_dir.glob("*.jpg"))
        if not images: return False
        for factor in factors:
            out_dir = self.output_dir / f"images_{factor}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in images:
                try:
                    img = Image.open(p)
                    # Keep original extension (e.g. .jpg) to match COLMAP images.txt
                    out_path = out_dir / (p.stem + p.suffix)
                    img.resize((img.width//factor, img.height//factor), Image.BICUBIC).save(out_path)
                except Exception as e:
                    logger.warning(f"Failed to downsample {p}: {e}")
        logger.info(f"Downsampled images to 1/{factors} scale.")
        return True

    def write_colmap_format(self) -> bool:
        logger.info("Writing COLMAP output...")
        with open(self.sparse_dir / "cameras.bin", "wb") as f:
            f.write(struct.pack("<QIiQQdddd", 1, 1, 1, self.image_width, self.image_height, self.focal_length, self.focal_length, self.image_width/2, self.image_height/2))
        with open(self.sparse_dir / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", len(self.poses)))
            for i, (name, pose) in enumerate(sorted(self.poses.items()), 1):
                w2c = np.linalg.inv(pose); q = self._rotation_matrix_to_quaternion(w2c[:3,:3]); t = w2c[:3, 3]
                f.write(struct.pack("<IdddddddI", i, *q, *t, 1)); f.write(name.encode("utf-8") + b"\x00"); f.write(struct.pack("<Q", 0))
        if not (self.sparse_dir / "points3D.bin").exists():
            with open(self.sparse_dir / "points3D.bin", "wb") as f: f.write(struct.pack("<Q", 0)) 
        return True

    def _rotation_matrix_to_quaternion(self, R):
        try: from scipy.spatial.transform import Rotation; return Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        except: return np.array([1,0,0,0]) 

    def refine_poses_native(self) -> bool:
        return True # Placeholder for now as ace_zero.py does optimization

    def dump_config(self, output_path: str):
        """Dump the internal ACE-Zero hyperparameters to a YAML file."""
        import yaml
        
        # Gather all relevant parameters
        config = {
            "model_name": "ACE-Zero",
            "hyperparameters": {
                "max_iterations": getattr(self, "max_iterations", 1000), 
                "learning_rate_max": getattr(self, "learning_rate_max", 0.005),
                "num_head_blocks": getattr(self, "num_head_blocks", 1),
                "seed_iterations": getattr(self, "seed_iterations", 50),
                "try_seeds": getattr(self, "try_seeds", 3)
            },
            "refinement": {
                "strategy": self.pose_refinement,
                "orthonormalization": getattr(self, "refinement_ortho", "gram-schmidt"),
                "wait_steps": self.pose_refinement_wait,
                "learning_rate": self.pose_refinement_lr
            },
            "environment": {
                "quality_mode": self.quality_mode,
                "depth_model": getattr(self, "depth_model", "depth_anything")
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"[ACE-Zero] Config saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ACE-Zero Pose Estimator Wrapper for 3DGS Pipeline")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output_dir", default="./data/acezero", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate")
    
    # Pose refinement options
    parser.add_argument("--pose_refinement", type=str, default="mlp", 
                        choices=["mlp", "naive", "none"],
                        help="Pose refinement strategy: mlp (best), naive (direct backprop), none")
    parser.add_argument("--pose_refinement_wait", type=int, default=5000,
                        help="Freeze poses for first N iterations (stability)")
    parser.add_argument("--pose_refinement_lr", type=float, default=0.001,
                        help="Learning rate for pose refinement")
    parser.add_argument("--refinement_ortho", type=str, default="gram-schmidt",
                        choices=["gram-schmidt", "procrustes"],
                        help="How to orthonormalize rotations during refinement")
    
    # Adaptive ACE-Zero options
    parser.add_argument("--quality_mode", type=str, default="balanced",
                        choices=["fast", "balanced", "quality"],
                        help="Quality mode: fast (speed), balanced (default), quality (best)")
    parser.add_argument("--min_confidence", type=int, default=1000,
                        help="Minimum confidence for pose filtering (default: 1000)")
    
    args = parser.parse_args()
    
    estimator = ACEZeroPoseEstimator(
        args.output_dir,
        pose_refinement=args.pose_refinement,
        pose_refinement_wait=args.pose_refinement_wait,
        pose_refinement_lr=args.pose_refinement_lr,
        refinement_ortho=args.refinement_ortho,
        quality_mode=args.quality_mode,
        min_confidence=args.min_confidence
    )
    estimator.process_video(args.video_path, fps=args.fps)
