"""
ACE-Zero Pose Estimator Wrapper for 3DGS Pipeline (WSL Support)

This module replaces COLMAP with ACE-Zero (Niantic, ECCV 2024) for camera pose estimation.
It outputs data in COLMAP format compatible with gsplat's simple_trainer.py.

Key Features:
- Lightweight: Runs on 8-12GB VRAM GPUs
- Scalable: Handles thousands of images efficiently
- WSL Support: Uses 'ace0' conda environment in WSL

Requirements:
    - ACE-Zero repository cloned to ../acezero/
    - WSL installed with 'ace0' environment in ~/miniconda3/envs/ace0
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

# Path to ACE-Zero installation (relative to this file)
ACEZERO_ROOT = Path(__file__).parent.parent / "acezero"

class ACEZeroPoseEstimator:
    """Wrapper for ACE-Zero pose estimation."""
    
    # Supported depth models: 'depth_anything' (fast, ~30ms), 'zoedepth' (metric, ~200ms), 'midas' (lightweight)
    DEPTH_MODELS = ['depth_anything', 'zoedepth', 'midas']
    
    def __init__(self, output_dir: Path, acezero_root: Optional[Path] = None, use_depth_init: bool = True,
                 pose_refinement: str = "mlp", pose_refinement_wait: int = 5000, 
                 pose_refinement_lr: float = 0.001, refinement_ortho: str = "gram-schmidt",
                 quality_mode: str = "balanced", min_confidence: int = 1000,
                 depth_model: str = "depth_anything"):
        self.output_dir = Path(output_dir).resolve()  # Ensure absolute path
        self.acezero_root = acezero_root or ACEZERO_ROOT
        self.use_depth_init = use_depth_init
        
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
        
        # Depth model selection: 'depth_anything' (default, fast), 'zoedepth' (metric), 'midas' (lightweight)
        self.depth_model = depth_model if depth_model in self.DEPTH_MODELS else 'depth_anything'
        logger.info(f"Depth estimator: {self.depth_model}")
        
        self.images_dir = self.output_dir / "images"
        self.sparse_dir = self.output_dir / "sparse" / "0"
        self.acezero_output = self.output_dir / "acezero_output"
        self.streaming_status_file = self.output_dir / "streaming_status.json"
        
        # Debug logging
        try:
            import json
            from pathlib import Path as PathType
            debug_log_path = PathType(__file__).parent.parent.parent / ".cursor" / "debug.log"
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H5",
                "location": "ace_zero_wrapper.py:36",
                "message": "ACEZeroPoseEstimator __init__",
                "data": {
                    "output_dir_input": str(output_dir),
                    "output_dir_resolved": str(self.output_dir),
                    "acezero_output": str(self.acezero_output),
                    "is_absolute": self.output_dir.is_absolute()
                },
                "timestamp": int(__import__('time').time() * 1000)
            }
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass  # Silently fail if logging fails
        
        self.image_width = 0
        self.image_height = 0
        self.focal_length = 0.0
        
        self.poses: Dict[str, np.ndarray] = {}  # image_name -> 4x4 c2w matrix
        self.points: Optional[np.ndarray] = None
        
        # WSL Python Path
        self.wsl_python = "~/miniconda3/envs/ace0/bin/python"

    @staticmethod
    def _remove_readonly(func, path, _):
        """Clear the readonly bit and reattempt the removal"""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

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
        
        # Sample frames to estimate motion/blur
        motion_scores = []
        prev_gray = None
        sample_interval = max(1, frame_count // 30)  # Sample ~30 frames
        
        for i in range(0, min(frame_count, 900), sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
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
        if self.quality_mode == 'fast':
            base_iterations = 12500
            self.cooldown_threshold = 0.7
            self.registration_threshold = 0.95
        elif self.quality_mode == 'quality':
            base_iterations = 50000
            self.cooldown_threshold = 0.6
            self.registration_threshold = 0.99
        else:  # balanced
            base_iterations = 25000
            self.cooldown_threshold = 0.7
            self.registration_threshold = 0.99
        
        # Scale iterations by expected frames
        expected_frames = getattr(self, 'num_frames', frame_count // 10)
        self.adaptive_iterations = int(base_iterations * (1 + expected_frames / 100))
        self.adaptive_iterations = min(100000, self.adaptive_iterations)
        
        logger.info(f"Adaptive Params [mode={self.quality_mode}, motion={avg_motion:.1f}]: "
                    f"repro_clamp={self.repro_loss_soft_clamp}, reg_conf={self.registration_confidence}, "
                    f"seeds={self.adaptive_try_seeds}, iters={self.adaptive_iterations}")
    
    def _filter_poses(self) -> tuple:
        """Filter poses by confidence, return (good_count, bad_count, filtered_path)."""
        pose_file = self.acezero_output / "poses_final.txt"
        if not pose_file.exists():
            return (0, 0, None)
        
        good_poses = []
        bad_count = 0
        
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10:
                    conf = float(parts[-1])
                    if conf >= self.min_confidence:
                        good_poses.append(line)
                    else:
                        bad_count += 1
                else:
                    good_poses.append(line)
        
        if bad_count > 0:
            filtered_path = self.acezero_output / "poses_final_filtered.txt"
            with open(filtered_path, 'w') as f:
                f.writelines(good_poses)
            logger.info(f"Pose Filtering: {len(good_poses)} good, {bad_count} filtered (conf < {self.min_confidence})")
            return (len(good_poses), bad_count, str(filtered_path))
        
        return (len(good_poses), 0, None)
        
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
        if sys.platform != 'win32':
            # On Linux, just run the precompute script directly
            from precompute_depths import precompute_depths
            precompute_depths(self.images_dir)
            return
        
        # On Windows, run via WSL with ace0 conda env
        logger.info("Pre-computing depths via WSL (frees ~2GB VRAM)...")
        
        wsl_images_dir = self._windows_to_wsl_path(str(self.images_dir.resolve()))
        wsl_acezero = self._windows_to_wsl_path(str(self.acezero_root.resolve()))
        
        # Use ace0 conda env for depth computation - User-selected depth model
        # depth_model options: 'depth_anything' (fast, ~30ms), 'zoedepth' (metric, ~200ms), 'midas' (~100ms)
        depth_model_choice = self.depth_model
        precompute_cmd = f'''
cd {wsl_acezero} && {self.wsl_python} -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
from pathlib import Path
import torch

images_dir = Path('{wsl_images_dir}')
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
            self._run_wsl_command(precompute_cmd.strip(), timeout=600)
            logger.info("Depth pre-computation complete.")
        except Exception as e:
            logger.warning(f"Depth pre-computation failed: {e}. Will use inline ZoeDepth.")
            
    def _run_wsl_command(self, cmd_str: str, check: bool = True, capture_output: bool = False, timeout: int = 1800) -> subprocess.CompletedProcess:
        """Helper to run commands in WSL.
        
        Note: WSL commands run as the default user without password prompts.
        If permission errors occur, they will be logged and raised as exceptions.
        
        Args:
            timeout: Maximum time in seconds to wait (default 1800 = 30 minutes)
        """
        if sys.platform != 'win32':
             raise RuntimeError("WSL command called on non-Windows platform")
             
        acezero_wsl_path = self._windows_to_wsl_path(str(self.acezero_root))
        full_cmd = f"cd {acezero_wsl_path} && {cmd_str}"
        
        logger.info(f"WSL Exec: {cmd_str[:150]}...")
        
        # Stream directly to console (bypass pipes) to fix buffering issues
        # WSL runs as default user - no password required
        # Add timeout to prevent hanging (especially on GPU initialization issues)
        try:
            result = subprocess.run(
                ["wsl", "-e", "bash", "-c", full_cmd],
                check=False,
                stdout=None, # Inherit calling process stdout
                stderr=None,  # Inherit calling process stderr
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            logger.error(f"WSL Command timed out after {timeout} seconds. This may indicate:")
            logger.error("  - GPU initialization hang (check VRAM usage)")
            logger.error("  - WSL GPU passthrough issues (run: wsl -e bash -c 'nvidia-smi')")
            logger.error("  - Process deadlock or infinite loop")
            if check:
                raise subprocess.TimeoutExpired(cmd_str, timeout)
            return subprocess.CompletedProcess(["wsl"], 124, stdout="", stderr="")  # 124 = timeout exit code
        
        if result.returncode != 0 and check:
             logger.error(f"WSL Command Failed (Return Code: {result.returncode}). See console output above.")
             logger.error("Note: If you see permission errors, ensure WSL has access to the Windows file system.")
             logger.error("Note: If you see GPU errors, check VRAM usage and WSL GPU passthrough.")
             raise subprocess.CalledProcessError(result.returncode, cmd_str)
        
        # Mock CompletedProcess
        return subprocess.CompletedProcess(result.args, result.returncode, stdout="", stderr="")

    def extract_frames(self, video_path: str, fps: float = 2.0) -> int:
        video_path = Path(video_path)
        if not video_path.exists(): raise FileNotFoundError(f"Video not found: {video_path}")
        
        if self.images_dir.exists(): 
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
        
        # Seeds: More frames = more viewpoint diversity = need more seed candidates
        # Complexity: O(seeds * seed_eval_time) - linear impact on init time
        # Scale: sqrt(frames/10), clamped to [2, 6]
        self.try_seeds = max(2, min(6, int(2 + (num_frames / 30) ** 0.5)))
        
        # Iterations: More frames = more images to register per round
        # Complexity: O(iterations * frames) - linear scaling
        # Scale: 5 base + log2(frames), clamped to [5, 15]
        import math
        self.max_iterations = max(5, min(15, int(5 + math.log2(max(1, num_frames)))))
        
        # Seed Parallel Workers: DISABLED - causes race condition with seed_index
        # The parallel workers can conflict when accessing the same image list
        # Always use 1 worker for stability (still fast enough for our use case)
        self.seed_parallel_workers = 1  # Fixed: parallel workers cause IndexError
        
        # Data Workers: CPU parallelism for data loading
        # Complexity: Minor CPU overhead, scales memory slightly
        # Scale: 2 base + frames/50, clamped to [2, 8]
        self.num_data_workers = max(2, min(8, int(2 + num_frames / 50)))
        
        # Head Blocks: Network depth = map capacity
        # Complexity: O(blocks * params) - quadratic VRAM impact
        # Scale: 1 for most cases (6GB VRAM), 2 only for very large datasets
        self.num_head_blocks = 1 if num_frames < 150 else 2
        
        # Training Buffer CPU: Offload training buffer to RAM
        # Complexity: ~2x slower loading but saves ~1-2GB VRAM
        # Scale: True (save VRAM) unless >100 frames and confident about VRAM
        self.training_buffer_cpu = num_frames < 100 or self.num_head_blocks > 1
        
        # Override with explicit max_iterations if provided
        if max_iterations is not None:
            self.max_iterations = max_iterations
            
        logger.info(f"ACE-Zero params [frames={num_frames}]: seeds={self.try_seeds}, "
                    f"iters={self.max_iterations}, seed_workers={self.seed_parallel_workers}, "
                    f"data_workers={self.num_data_workers}, heads={self.num_head_blocks}, "
                    f"cpu_buffer={self.training_buffer_cpu}, refinement={self.pose_refinement}")
            
        images_glob = str(self.images_dir.resolve() / "*.jpg")
        output_dir = str(self.acezero_output.resolve())
        acezero_script = str(self.acezero_root.resolve() / "ace_zero.py")
        
        if sys.platform == 'win32':
            # Verify WSL GPU access before running (helps catch GPU passthrough issues early)
            try:
                logger.info("Verifying WSL GPU access...")
                gpu_check = self._run_wsl_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", 
                                                  check=False, timeout=10)
                # #region agent log
                try:
                    import json
                    from pathlib import Path as PathType
                    import time
                    debug_log_path = PathType(__file__).parent.parent.parent / ".cursor" / "debug.log"
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H5",
                        "location": "ace_zero_wrapper.py:195",
                        "message": "WSL GPU check result",
                        "data": {
                            "returncode": gpu_check.returncode,
                            "gpu_accessible": gpu_check.returncode == 0
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception:
                    pass
                # #endregion
                if gpu_check.returncode == 0:
                    logger.info("WSL GPU access verified.")
                else:
                    logger.warning("WSL GPU check failed. ACE-Zero may fail or hang on GPU operations.")
            except Exception as e:
                # #region agent log
                try:
                    import json
                    from pathlib import Path as PathType
                    import time
                    debug_log_path = PathType(__file__).parent.parent.parent / ".cursor" / "debug.log"
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H5",
                        "location": "ace_zero_wrapper.py:210",
                        "message": "WSL GPU check exception",
                        "data": {
                            "exception_type": type(e).__name__,
                            "exception_message": str(e)
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception:
                    pass
                # #endregion
                logger.warning(f"Could not verify WSL GPU access: {e}. Continuing anyway...")
            
            return self._run_ace_zero_via_wsl(images_glob, output_dir, acezero_script, max_iterations)
        else:
             return self._run_ace_zero_native(images_glob, output_dir, acezero_script, max_iterations)

    def _windows_to_wsl_path(self, windows_path: str) -> str:
        p = str(Path(windows_path).resolve())
        suffix = p[2:].replace('\\', '/')
        if len(p) >= 2 and p[1] == ':':
            return f"/mnt/{p[0].lower()}{suffix}"
        return p.replace('\\', '/')
    
    def _run_ace_zero_via_wsl(self, images_glob: str, output_dir: str, acezero_script: str, max_iterations: int) -> bool:
        # Generate temporary WSL paths
        run_id = uuid.uuid4().hex[:8]
        # We assume WSL has write access to c-drive for temp speed, OR we copy to /tmp? 
        # ace0 example copied to /tmp. Let's stick to /tmp for speed/compat.
        wsl_temp_images = f"/tmp/acezero_images_{run_id}"
        wsl_temp_output = f"/tmp/acezero_output_{run_id}"
        
        wsl_images_src = self._windows_to_wsl_path(str(self.images_dir.resolve()))
        # Ensure output_dir is absolute before conversion
        output_dir_abs = str(Path(output_dir).resolve())
        wsl_final_output = self._windows_to_wsl_path(output_dir_abs)
        wsl_script = self._windows_to_wsl_path(acezero_script)
        
        # Debug logging
        try:
            import json
            from pathlib import Path as PathType
            debug_log_path = PathType(__file__).parent.parent.parent / ".cursor" / "debug.log"
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "H5",
                "location": "ace_zero_wrapper.py:169",
                "message": "WSL path conversion",
                "data": {
                    "output_dir_input": output_dir,
                    "output_dir_abs": output_dir_abs,
                    "wsl_final_output": wsl_final_output,
                    "acezero_output_path": str(self.acezero_output.resolve())
                },
                "timestamp": int(__import__('time').time() * 1000)
            }
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass  # Silently fail if logging fails


        # Commands
        # Note: Our modified dataset_io.py (with Depth Anything V2) is already in gsplat/scripts/acezero/
        # It will be used automatically via WSL path mounting
        
        # 1. Setup - copy images
        setup_cmd = f"mkdir -p {wsl_temp_images} {wsl_temp_output} && cp {wsl_images_src}/*.jpg {wsl_temp_images}/"
        
        # 2. Run ACE-Zero (will now use Depth Anything V2 via our modified dataset_io.py)
        logger.info(f"ACE-Zero will use Depth Anything V2 for depth estimation (~30ms/frame)")
        ace_cmd = f'{self.wsl_python} -u {wsl_script} "{wsl_temp_images}/*.jpg" {wsl_temp_output} --iterations_max {self.max_iterations} --try_seeds {self.try_seeds} --seed_parallel_workers {self.seed_parallel_workers} --training_buffer_cpu {self.training_buffer_cpu} --num_data_workers {self.num_data_workers} --num_head_blocks {self.num_head_blocks} --refinement {self.pose_refinement} --refinement_ortho {self.refinement_ortho} --pose_refinement_wait {self.pose_refinement_wait} --pose_refinement_lr {self.pose_refinement_lr}'
        
        cleanup_cmd = f"rm -rf {wsl_temp_images} {wsl_temp_output}"
        
        # 3. Copy back
        copy_back_cmd = f"mkdir -p {wsl_final_output} && cp -r {wsl_temp_output}/* {wsl_final_output}/"
        
        # Chain it: setup -> run ACE-Zero -> copy back -> cleanup
        full_cmd = f"{setup_cmd} && {ace_cmd} && {copy_back_cmd} && {cleanup_cmd}"
        
        try:
            # #region agent log
            try:
                import json
                from pathlib import Path as PathType
                import time
                debug_log_path = PathType(__file__).parent.parent.parent / ".cursor" / "debug.log"
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H5",
                    "location": "ace_zero_wrapper.py:250",
                    "message": "About to run ACE-Zero WSL command",
                    "data": {
                        "full_cmd_preview": full_cmd[:200],
                        "wsl_temp_images": wsl_temp_images,
                        "wsl_temp_output": wsl_temp_output,
                        "wsl_final_output": wsl_final_output
                    },
                    "timestamp": int(time.time() * 1000)
                }
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            
            self._run_wsl_command(full_cmd, check=True)
            logger.info("ACE-Zero (WSL) completed.")
            self.refine_poses_native(via_wsl=True) # Check logic for this
            return self._parse_acezero_output()
        except subprocess.CalledProcessError as e:
            # #region agent log
            try:
                import json
                from pathlib import Path as PathType
                import time
                debug_log_path = PathType(__file__).parent.parent.parent / ".cursor" / "debug.log"
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H5",
                    "location": "ace_zero_wrapper.py:255",
                    "message": "ACE-Zero WSL command failed",
                    "data": {
                        "returncode": e.returncode,
                        "cmd_preview": str(e.cmd)[:200] if hasattr(e, 'cmd') else "unknown"
                    },
                    "timestamp": int(time.time() * 1000)
                }
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            logger.error(f"ACE-Zero (WSL) failed: {e}")
            return False
        except subprocess.TimeoutExpired as e:
            # #region agent log
            try:
                import json
                from pathlib import Path as PathType
                import time
                debug_log_path = PathType(__file__).parent.parent.parent / ".cursor" / "debug.log"
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H5",
                    "location": "ace_zero_wrapper.py:275",
                    "message": "ACE-Zero WSL command timed out",
                    "data": {
                        "timeout": e.timeout if hasattr(e, 'timeout') else "unknown"
                    },
                    "timestamp": int(time.time() * 1000)
                }
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            logger.error(f"ACE-Zero (WSL) timed out: {e}")
            return False
        except Exception as e:
            # #region agent log
            try:
                import json
                from pathlib import Path as PathType
                import time
                debug_log_path = PathType(__file__).parent.parent.parent / ".cursor" / "debug.log"
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "H5",
                    "location": "ace_zero_wrapper.py:290",
                    "message": "ACE-Zero WSL unexpected exception",
                    "data": {
                        "exception_type": type(e).__name__,
                        "exception_message": str(e)
                    },
                    "timestamp": int(time.time() * 1000)
                }
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
            logger.error(f"ACE-Zero (WSL) failed: {e}")
            return False

    def _run_ace_zero_native(self, images_glob: str, output_dir: str, acezero_script: str, max_iterations: int) -> bool:
        cmd = [sys.executable, "-u", acezero_script, images_glob, output_dir, "--iterations_max", str(max_iterations), "--seed_parallel_workers", "1", "--refinement", self.pose_refinement, "--refinement_ortho", self.refinement_ortho, "--pose_refinement_wait", str(self.pose_refinement_wait), "--pose_refinement_lr", str(self.pose_refinement_lr)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.acezero_root) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            subprocess.run(cmd, cwd=str(self.acezero_root), env=env, check=True)
            self.refine_poses_native()
            return self._parse_acezero_output()
        except: return False

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
        if not self.use_depth_init: return False
        if sys.platform == 'win32': return self._generate_points_via_wsl()
        return False

    def _generate_points_via_wsl(self) -> bool:
        script = self._windows_to_wsl_path(str(self.acezero_root / "generate_points_wsl.py"))
        poses = self._windows_to_wsl_path(str(self.acezero_output / "poses_final.txt"))
        imgs = self._windows_to_wsl_path(str(self.images_dir))
        out = self._windows_to_wsl_path(str(self.sparse_dir / "points3D.bin"))
        
        cmd = f"{self.wsl_python} -u {script} '{poses}' '{imgs}' '{out}'"
        try:
             res = self._run_wsl_command(cmd, check=False, capture_output=True)
             if (self.sparse_dir / "points3D.bin").stat().st_size > 1000: return True
             logger.error(f"WSL Point Gen failed. Output: {res.stdout}")
        except Exception as e: logger.error(f"WSL Point Gen Error: {e}")
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

    def refine_poses_native(self, via_wsl: bool = False) -> bool:
        return True # Placeholder for now as ace_zero.py does optimization

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
