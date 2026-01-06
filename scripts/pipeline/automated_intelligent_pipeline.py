#!/usr/bin/env python3
"""
Automated 3DGS Pipeline with ACE-Zero Pose Estimation
Condensed & Optimized Version

Flow: Video -> ACE-Zero (AI Poses) -> 3DGS Training (MCMC)
"""

import sys
import subprocess
import time
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug logging helper
DEBUG_LOG_PATH = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
def debug_log(location: str, message: str, data: dict, hypothesis_id: str = None):
    """Write debug log entry in NDJSON format"""
    try:
        import time
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        # Log to stderr so we can see if debug logging fails
        import sys
        print(f"DEBUG LOG ERROR: {e}", file=sys.stderr)

def resolve_path_relative_to_project(path_str: str, must_exist: bool = False, must_be_file: bool = False) -> Path:
    """Resolve a path relative to project root, handling 'gsplat/' prefix"""
    path_obj = Path(path_str)
    project_root = Path(__file__).resolve().parent.parent.parent
    
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path not found: {resolved}")
        if must_be_file and not resolved.is_file():
            raise ValueError(f"Path is not a file: {resolved}")
        return resolved
    
    # Try multiple resolution strategies
    candidates = []
    
    # Strategy 1: If path starts with "gsplat/", remove that prefix and try relative to project root
    if str(path_str).startswith("gsplat/"):
        path_without_prefix = str(path_str)[7:]  # Remove "gsplat/" prefix
        candidates.append(project_root / path_without_prefix)
    
    # Strategy 2: Relative to project root as-is
    candidates.append(project_root / path_str)
    
    # Strategy 3: Relative to current working directory
    candidates.append(path_obj.resolve())
    
    # Find the first candidate that exists (if must_exist is True) or use the first one
    if must_exist or must_be_file:
        for candidate in candidates:
            if candidate.exists():
                if not must_be_file or candidate.is_file():
                    return candidate.resolve()
        # If we need it to exist but none found, raise error with tried paths
        raise FileNotFoundError(f"Path not found: {path_str}. Tried: {[str(c) for c in candidates]}")
    
    # Return the first candidate (prefer path without gsplat/ prefix)
    return candidates[0].resolve()

class PipelineConfig:
    """Central configuration for the pipeline"""
    
    def __init__(self, args):
        self.OUTPUT_BASE = resolve_path_relative_to_project(args.output_dir)
        # Ensure it's an absolute path
        self.OUTPUT_BASE = self.OUTPUT_BASE.resolve()
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:88", "OUTPUT_BASE resolved", {"original": args.output_dir, "resolved": str(self.OUTPUT_BASE), "exists": self.OUTPUT_BASE.exists(), "is_absolute": self.OUTPUT_BASE.is_absolute()}, "H5")
        # #endregion
        self.FPS = args.fps
        self.DATA_FACTOR = args.data_factor
        
        # Training Settings
        self.TRAINING_MAX_STEPS = args.max_steps
        self.TRAINING_EVAL_STEPS = [3000, 7000]
        self.TRAINING_SAVE_STEPS = [3000, 7000]
        
        # Quality/Tuning
        self.INIT_SCALE = args.init_scale
        self.OPACITY_REG = args.opacity_reg
        self.SCALE_REG = args.scale_reg
        self.USE_APP_OPT = args.app_opt
        
        # Advanced Modules
        self.WITH_UT = args.with_ut
        self.WITH_EVAL3D = args.with_eval3d
        
        # Cloud Storage
        self.CLOUD_SYNC = getattr(args, 'cloud_sync', False)
        self.SCENE_NAME = getattr(args, 'scene_name', None)
        
        # Streaming Mode
        self.STREAMING = getattr(args, 'streaming', False)
        
        # Floater Pruning
        self.PRUNE = getattr(args, 'prune', True)  # Default ON
        self.PRUNE_MODEL = getattr(args, 'prune_model', None)  # Optional DQN model path
        
        # GeNVS-Lite Novel View Generation
        self.GENVS_ENABLED = getattr(args, 'genvs', False)  # Augment with synthetic views
        self.GENVS_NUM_VIEWS = getattr(args, 'genvs_views', 20)  # Number of novel views

        # JOGS-style Joint Pose-3DGS Optimization (default ON)
        self.POSE_OPT = getattr(args, 'pose_opt', True)
        self.POSE_OPT_WARMUP = getattr(args, 'pose_opt_warmup', 1000)  # Steps before pose opt starts

        # [Autoregressive GeNVS] The "Dream" Loop
        self.GENVS_AUTOREGRESSIVE = getattr(args, 'genvs_autoregressive', False)

        # [ROBUSTNESS] Flag Validation
        if self.WITH_UT and self.POSE_OPT:
            logger.warning("Unscented Transform (+UT) + Pose Opt is currently incompatible in gsplat.")
            logger.warning("Disabling UT/Eval3D to prioritize Pose Optimization.")
            self.WITH_UT = False
            self.WITH_EVAL3D = False

        # Override Input
        self.COLMAP_INPUT = getattr(args, 'colmap_input', None)
        
        # === NEW: Skip ACE detection (consistent with pipeline_wrapper.py) ===
        self.SKIP_ACE = getattr(args, 'skip_ace', False)
        
        # === NEW: ACE-Zero Quality Mode (from adaptive_ace_zero.py) ===
        self.QUALITY_MODE = getattr(args, 'quality_mode', 'balanced')  # fast | balanced | quality
        
        # === NEW: Pose Confidence Filtering ===
        self.MIN_CONFIDENCE = getattr(args, 'min_confidence', 1000)  # Filter poses below this
        
        # === NEW: Early Stopping (from pipeline_wrapper.py) ===
        self.EARLY_STOPPING = getattr(args, 'early_stopping', True)
        self.EARLY_STOP_PATIENCE = getattr(args, 'early_stop_patience', 500)
        self.EARLY_STOP_MIN_DELTA = getattr(args, 'early_stop_min_delta', 0.001)
        self.EARLY_STOP_MIN_STEPS = getattr(args, 'early_stop_min_steps', 2000)
        
        # === NEW: Depth Model Selection ===
        # Options: 'depth_anything' (fast, ~30ms), 'zoedepth' (metric, ~200ms), 'midas' (~100ms)
        self.DEPTH_MODEL = getattr(args, 'depth_model', 'depth_anything')

    def create_directories(self):
        """Ensure output directories exist"""
        dirs = [self.OUTPUT_BASE, self.OUTPUT_BASE / "results"]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

class IntelligentPipeline:
    """Streamlined Pipeline Orchestrator"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, video_path: str) -> Dict:
        """Execute the optimized ACE-Zero -> 3DGS pipeline"""
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:56", "run() entry", {"video_path": str(video_path), "output_dir": str(self.config.OUTPUT_BASE), "cwd": str(Path.cwd())}, "H4")
        # #endregion
        
        # Resolve video path using helper function (must exist and be a file)
        video_path = resolve_path_relative_to_project(video_path, must_exist=True, must_be_file=True)
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:107", "video_path resolved", {"resolved_path": str(video_path), "exists": video_path.exists(), "is_file": video_path.is_file()}, "H4")
        # #endregion
        
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("INTELLIGENT PIPELINE: ACE-Zero + 3DGS")
        logger.info("="*70)
        logger.info(f"Input: {video_path}")
        logger.info(f"Output: {self.config.OUTPUT_BASE}")

        self.config.create_directories()
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:67", "directories created", {"output_base": str(self.config.OUTPUT_BASE), "output_base_exists": self.config.OUTPUT_BASE.exists()}, "H6")
        # #endregion

        try:
            # 1. Pose Estimation (or Load Existing)
            if self.config.COLMAP_INPUT:
                ace_output = resolve_path_relative_to_project(self.config.COLMAP_INPUT, must_exist=True)
                logger.info(f"Using provided COLMAP input: {ace_output}")
            elif self.config.SKIP_ACE or self._detect_existing_ace_output():
                logger.info("Skipping ACE-Zero (using existing output).")
                ace_output = self.config.OUTPUT_BASE / "acezero_output"
                if not ace_output.exists():
                    raise FileNotFoundError(f"Cannot skip ACE-Zero: {ace_output} does not exist.")
            else:
                ace_output = self._run_pose_estimation(video_path)
                # === NEW: Filter low-confidence poses ===
                ace_output = self._filter_poses_by_confidence(ace_output)
            # #region agent log
            debug_log("automated_intelligent_pipeline.py:72", "pose estimation completed", {"ace_output": str(ace_output), "ace_output_exists": ace_output.exists()}, "H5")
            # #endregion
            
            # 1.5. GeNVS Novel View Generation (optional)
            if self.config.GENVS_ENABLED:
                ace_output = self._run_novel_view_generation(ace_output)
            
            # 2. Training
            result_dir = self._run_training(ace_output)
            # #region agent log
            debug_log("automated_intelligent_pipeline.py:75", "training completed", {"result_dir": str(result_dir)}, "H3")
            # #endregion
            
            # 2.2 Autoregressive Refinement (The "Dream" Loop)
            if self.config.GENVS_AUTOREGRESSIVE:
                result_dir = self._run_autoregressive_refinement(ace_output, result_dir)
            
            # 2.5. Floater Pruning (optional post-processing)
            if self.config.PRUNE:
                self._run_floater_pruning(result_dir)
            
            # 3. Cloud Sync (optional)
            cloud_result = None
            if self.config.CLOUD_SYNC:
                cloud_result = self._upload_to_cloud(result_dir)
            
            elapsed = time.time() - start_time
            self._print_success(elapsed, result_dir)
            
            return {
                "status": "success",
                "elapsed": elapsed,
                "output": str(result_dir),
                "cloud_sync": cloud_result
            }

        except Exception as e:
            # #region agent log
            debug_log("automated_intelligent_pipeline.py:85", "pipeline exception", {"exception_type": type(e).__name__, "exception_message": str(e)}, "H3")
            # #endregion
            logger.error(f"Pipeline Failed: {e}")
            raise

    def _detect_existing_ace_output(self) -> bool:
        """Check if valid ACE-Zero output already exists (consistent with pipeline_wrapper.py)."""
        expected_output = self.config.OUTPUT_BASE / "acezero_output"
        sparse_dir = expected_output / "sparse" / "0"
        # Check for valid COLMAP output (this is the definitive sign ACE-Zero finished)
        if (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists():
            logger.info(f"Detected existing ACE-Zero output at {expected_output}")
            return True
        return False

    def _filter_poses_by_confidence(self, ace_output: Path) -> Path:
        """Filter poses by confidence score to remove bad frames."""
        pose_file = ace_output / "poses_final.txt"
        if not pose_file.exists():
            return ace_output
        
        min_conf = self.config.MIN_CONFIDENCE
        logger.info(f"Filtering poses with confidence < {min_conf}...")
        
        good_poses = []
        bad_count = 0
        
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10:  # Has confidence field
                    conf = float(parts[-1])
                    if conf >= min_conf:
                        good_poses.append(line)
                    else:
                        bad_count += 1
                        logger.debug(f"Filtered low-confidence frame: {parts[0]} (conf={conf})")
                else:
                    good_poses.append(line)  # Keep lines without confidence
        
        if bad_count > 0:
            # Write filtered poses
            filtered_path = ace_output / "poses_final_filtered.txt"
            with open(filtered_path, 'w') as f:
                f.writelines(good_poses)
            logger.info(f"Filtered {bad_count} low-confidence poses. {len(good_poses)} remaining.")
            logger.info(f"Filtered poses saved to: {filtered_path}")
        
        return ace_output

    def _run_pose_estimation(self, video_path: Path) -> Path:
        """Step 1: AI-Powered Pose Estimation"""
        logger.info("\nSTEP 1: POSES (ACE-Zero)")
        logger.info(f"Quality Mode: {self.config.QUALITY_MODE}")
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:89", "_run_pose_estimation entry", {"video_path": str(video_path), "__file__": str(__file__), "parent": str(Path(__file__).parent)}, "H1")
        # #endregion
        
        # Lazy import to keep startup fast
        try:
            from .ace_zero_wrapper import ACEZeroPoseEstimator
            # #region agent log
            debug_log("automated_intelligent_pipeline.py:95", "import success (relative)", {"method": "relative_import"}, "H1")
            # #endregion
        except ImportError as e1:
            # #region agent log
            debug_log("automated_intelligent_pipeline.py:97", "relative import failed", {"error": str(e1)}, "H1")
            # #endregion
            try:
                from ace_zero_wrapper import ACEZeroPoseEstimator
                # #region agent log
                debug_log("automated_intelligent_pipeline.py:98", "import success (absolute)", {"method": "absolute_import"}, "H1")
                # #endregion
            except ImportError as e2:
                # #region agent log
                debug_log("automated_intelligent_pipeline.py:100", "absolute import failed, trying fallback", {"error": str(e2), "sys_path_before": str(sys.path)}, "H1")
                # #endregion
                # Fallback for when running from root
                sys.path.append(str(Path(__file__).parent))
                # #region agent log
                debug_log("automated_intelligent_pipeline.py:102", "sys.path modified", {"added_path": str(Path(__file__).parent), "sys_path_after": str(sys.path)}, "H1")
                # #endregion
                from ace_zero_wrapper import ACEZeroPoseEstimator
                # #region agent log
                debug_log("automated_intelligent_pipeline.py:103", "import success (fallback)", {"method": "fallback_import"}, "H1")
                # #endregion

        output_dir = self.config.OUTPUT_BASE / "acezero_output"
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:104", "output_dir set", {"output_dir": str(output_dir), "output_dir_absolute": str(output_dir.resolve()), "output_base": str(self.config.OUTPUT_BASE), "output_base_absolute": str(self.config.OUTPUT_BASE.resolve())}, "H5")
        # #endregion
        logger.info(f"Depth Model: {self.config.DEPTH_MODEL}")
        estimator = ACEZeroPoseEstimator(
            output_dir=output_dir,
            quality_mode=self.config.QUALITY_MODE,
            min_confidence=self.config.MIN_CONFIDENCE,
            depth_model=self.config.DEPTH_MODEL
        )
        
        success = estimator.process_video(str(video_path), fps=self.config.FPS)
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:107", "process_video completed", {"success": success, "fps": self.config.FPS}, "H5")
        # #endregion
        
        if not success:
            raise RuntimeError("ACE-Zero Pose Estimation failed.")

        # Validation
        points_bin = output_dir / "sparse" / "0" / "points3D.bin"
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:113", "points_bin validation", {"points_bin": str(points_bin), "exists": points_bin.exists(), "size": points_bin.stat().st_size if points_bin.exists() else 0}, "H5")
        # #endregion
        if not points_bin.exists() or points_bin.stat().st_size < 100:
            raise RuntimeError(f"Invalid point cloud generated at {points_bin}")

        logger.info("Pose estimation valid.")
        return output_dir

    def _run_novel_view_generation(self, colmap_dir: Path) -> Path:
        """Step 1.5: GeNVS-Lite Novel View Generation to Augment Dataset"""
        logger.info("\nSTEP 1.5: GENVS-LITE (Novel View Synthesis)")
        logger.info(f"Generating {self.config.GENVS_NUM_VIEWS} synthetic views...")
        
        # Import GeNVS components
        import sys
        genvs_path = Path(__file__).parent.parent / "genvs"
        sys.path.insert(0, str(genvs_path))
        
        try:
            from run_completion import GeNVSLite
            from inject_back_views import inject_back_views
            
            # Step A: Generate novel views from input images
            input_images_dir = colmap_dir / "images"
            novel_output_dir = self.config.OUTPUT_BASE / "novel_views"
            novel_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  Input: {input_images_dir}")
            logger.info(f"  Output: {novel_output_dir}")
            
            # Initialize GeNVS-Lite (uses ZoeDepth by default)
            genvs = GeNVSLite(use_zoedepth=True)
            
            # Process sample frames for novel view generation
            import cv2
            import numpy as np
            
            image_files = sorted(input_images_dir.glob("*.jpg")) + sorted(input_images_dir.glob("*.png"))
            if not image_files:
                logger.warning("  No images found, skipping GeNVS")
                return colmap_dir
            
            # Use middle frame as reference
            ref_image_path = image_files[len(image_files) // 2]
            ref_image = cv2.imread(str(ref_image_path))
            ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"  Reference frame: {ref_image_path.name}")
            
            # Generate novel views
            novel_views = genvs.generate_novel_views(
                ref_image_rgb, 
                num_views=self.config.GENVS_NUM_VIEWS
            )
            
            # Save novel views
            for i, view in enumerate(novel_views):
                out_path = novel_output_dir / f"novel_{i:04d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
            
            logger.info(f"  Generated {len(novel_views)} novel views")
            
            # Step B: Inject novel views into COLMAP dataset
            augmented_dir = self.config.OUTPUT_BASE / "augmented_colmap"
            
            inject_back_views(
                colmap_dir=colmap_dir,
                novel_views_dir=novel_output_dir,
                output_dir=augmented_dir
            )
            
            logger.info(f"  Augmented dataset: {augmented_dir}")
            logger.info("GeNVS augmentation complete.")
            
            return augmented_dir
            
        except Exception as e:
            logger.warning(f"GeNVS failed: {e}. Falling back to original dataset.")
            return colmap_dir

    def _run_training(self, data_dir: Path) -> Path:
        """Step 2: MCMC 3DGS Training"""
        logger.info("\nSTEP 2: TRAINING (3DGS MCMC)")
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:120", "_run_training entry", {"data_dir": str(data_dir), "data_dir_exists": data_dir.exists(), "__file__": str(__file__)}, "H2")
        # #endregion
        
        result_dir = self.config.OUTPUT_BASE / "results" / "acezero_3dgs"
        trainer_script = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_trainer.py"
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:125", "trainer_script path calculated", {"trainer_script": str(trainer_script), "exists": trainer_script.exists(), "parent_paths": str(Path(__file__).resolve().parent.parent.parent)}, "H2")
        # #endregion
        
        if not trainer_script.exists():
             # Fallback location check
             trainer_script = Path("examples/simple_trainer.py")
             # #region agent log
             debug_log("automated_intelligent_pipeline.py:129", "trainer_script fallback", {"trainer_script": str(trainer_script), "exists": trainer_script.exists(), "is_absolute": trainer_script.is_absolute()}, "H2")
             # #endregion

        cmd = [
            sys.executable, "-u", str(trainer_script), "mcmc",
            "--data_dir", str(data_dir),
            "--result_dir", str(result_dir),
            "--max_steps", str(self.config.TRAINING_MAX_STEPS),
            "--data_factor", str(self.config.DATA_FACTOR),
            "--disable_viewer",
            "--save_ply",
            
            # Robustness / Quality settings
            "--init_scale", str(self.config.INIT_SCALE),
            "--opacity_reg", str(self.config.OPACITY_REG),
            "--scale_reg", str(self.config.SCALE_REG),
        ]

        if self.config.USE_APP_OPT: cmd.append("--app_opt")
        if self.config.WITH_UT: cmd.append("--with_ut")
        if self.config.WITH_EVAL3D: cmd.append("--with_eval3d")
        if self.config.STREAMING: cmd.append("--streaming")
        
        # JOGS-style joint pose-3DGS optimization
        if self.config.POSE_OPT:
            cmd.extend(["--pose_opt", "--pose_warmup_steps", str(self.config.POSE_OPT_WARMUP)])

        logger.info(f"Command: {' '.join(cmd)}")
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:150", "command constructed", {"cmd": cmd, "sys_executable": sys.executable, "trainer_script": str(trainer_script)}, "H2,H3")
        # #endregion
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:152", "subprocess started", {"pid": process.pid}, "H3")
        # #endregion

        # Stream output
        line_count = 0
        for line in process.stdout:
            print(f"[Train] {line.strip()}")
            line_count += 1
            if line_count <= 5:  # Log first 5 lines
                # #region agent log
                debug_log("automated_intelligent_pipeline.py:157", "subprocess output", {"line": line.strip()[:200]}, "H3")
                # #endregion
            
        exit_code = process.wait()
        # #region agent log
        debug_log("automated_intelligent_pipeline.py:160", "subprocess completed", {"exit_code": exit_code, "line_count": line_count}, "H3")
        # #endregion
        if exit_code != 0:
            raise RuntimeError(f"Training process exited with error code {exit_code}.")

        return result_dir

    def _run_autoregressive_refinement(self, ace_output: Path, initial_result_dir: Path) -> Path:
        """Step 3: Autoregressive Refinement (The 'Dream' Loop)"""
        logger.info("\nSTEP 3: AUTOREGRESSIVE REFINEMENT")
        logger.info("Sampling 3D Space -> Refining with GeNVS -> Retraining")
        
        # Setup Imports
        import sys
        import json
        import shutil
        import cv2
        import numpy as np
        
        genvs_path = Path(__file__).parent.parent / "genvs"
        sys.path.insert(0, str(genvs_path))
        from run_completion import GeNVSLite
        from inject_back_views import read_model, mirror_camera_pose, compute_scene_center, qvec2rotmat
        
        # 1. Setup Iteration Paths
        iter_dir = self.config.OUTPUT_BASE / "autoregressive_iter1"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Compute "Back" Pose from COLMAP data
        sparse_in = ace_output / "sparse" / "0"
        cameras, images,Points3D = read_model(str(sparse_in))
        
        # Pick reference (middle image)
        ref_image_id = list(images.keys())[len(images) // 2]
        ref_image = images[ref_image_id]
        scene_center = compute_scene_center(images)
        
        # Calculate Mirror Pose
        qvec, tvec = mirror_camera_pose(ref_image.qvec, ref_image.tvec, scene_center)
        
        # Convert to c2w for Renderer
        R_w2c = qvec2rotmat(qvec)
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ tvec
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w
        
        # Get Instrinsics
        cam = cameras[ref_image.camera_id]
        # output width/height? Assume matches ref image or default
        # SimpleTrainer defaults to loaded dataset resolution.
        # We need to explicitly set W/H for render.
        # COLMAP Camera model: assumes PINHOLE [fx, fy, cx, cy]
        if cam.model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE"]:
             fx, fy, cx, cy = cam.params[:4]
             K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
             w, h = cam.width, cam.height
        elif cam.model == "SIMPLE_PINHOLE":
             f, cx, cy = cam.params[:3]
             K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
             w, h = cam.width, cam.height
        else:
             logger.warning(f"Unknown camera model {cam.model}, guessing K")
             w, h = cam.width, cam.height
             f = max(w, h)
             K = [[f, 0, w/2], [0, f, h/2], [0, 0, 1]]

        # Write Pose File
        pose_data = {
            "c2w": c2w.tolist(),
            "K": K,
            "width": w,
            "height": h
        }
        pose_file = iter_dir / "target_pose.json"
        with open(pose_file, 'w') as f:
            json.dump(pose_data, f)
            
        # 3. Render 'Draft' View (Sample 3D Space)
        # We call simple_trainer with --render_pose_file using the INITIAL model
        logger.info("  Sampling 3D latent space (Rendering)...")
        # Reuse _run_training logic but with extra flags
        trainer_script = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_trainer.py"
        ckpt_dir = initial_result_dir / "ckpts"
        # Find latest ckpt
        ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: int(p.stem.split('_')[1]))
        latest_ckpt = ckpts[-1]
        
        cmd_render = [
            sys.executable, str(trainer_script), "mcmc",
            "--result_dir", str(initial_result_dir),
            "--data_dir", str(ace_output),  # <--- REQUIRED for Runner init
            "--data_factor", "1", # <--- Prevent looking for images_4
            "--ckpt", str(latest_ckpt),
            "--render_pose_file", str(pose_file),
            "--disable_viewer"
        ]
        subprocess.run(cmd_render, check=True)
        
        rendered_img_path = initial_result_dir / "renders" / "custom_render.png"
        if not rendered_img_path.exists():
            raise RuntimeError("Rendering failed to produce output")
            
        # 4. GeNVS Refinement
        logger.info("  Refining sample with GeNVS...")
        genvs = GeNVSLite(use_zoedepth=True)
        
        # Load Reference (Original Front)
        front_img = cv2.cvtColor(cv2.imread(str(ace_output / "images" / ref_image.name)), cv2.COLOR_BGR2RGB)
        # Load Init (Rendered Back)
        init_img = cv2.cvtColor(cv2.imread(str(rendered_img_path)), cv2.COLOR_BGR2RGB)
        
        # Generate
        novel_views = genvs.generate_novel_views(
            front_image=front_img,
            num_views=1, # Just 1 for now to prove concept
            init_images=[init_img] # <--- THE KEY: Seeding with 3DGS render
        )
        
        # Save Refined Image
        refined_path = iter_dir / "refined_back_0000.png"
        cv2.imwrite(str(refined_path), cv2.cvtColor(novel_views[0], cv2.COLOR_RGB2BGR))
        
        # 5. Inject into Dataset
        logger.info("  Injecting refined view...")
        augmented_dir = self.config.OUTPUT_BASE / "acezero_output_autoregressive"
        
        # Use inject_back_views but point to our NEW single image dir
        # We need a temp dir just for this image so injector finds it
        temp_novel_dir = iter_dir / "novel_temp"
        temp_novel_dir.mkdir(exist_ok=True)
        shutil.copy2(refined_path, temp_novel_dir / "novel_0.png")
        
        # This will create a dataset with Original + This New View
        # Note: We should probably accummulate if we did multiple loops, but for V1 we restart from clean
        from inject_back_views import inject_back_views
        inject_back_views(
            colmap_dir=ace_output,
            novel_views_dir=temp_novel_dir,
            output_dir=augmented_dir
        )
        
        # 6. Re-Train
        logger.info("  Retraining on Refined Reality...")
        return self._run_training(augmented_dir)

    def _run_floater_pruning(self, result_dir: Path):
        """Step 2.5: Prune floaters from 3DGS model (optional post-processing)"""
        logger.info("\nSTEP 2.5: FLOATER PRUNING")
        
        # Find the PLY file
        ply_dir = result_dir / "ply"
        if not ply_dir.exists():
            logger.warning(f"No PLY directory found at {ply_dir}. Skipping pruning.")
            return
        
        # Get the latest PLY file
        ply_files = list(ply_dir.glob("*.ply"))
        if not ply_files:
            logger.warning("No PLY files found. Skipping pruning.")
            return
        
        # Use the most recent PLY
        input_ply = max(ply_files, key=lambda p: p.stat().st_mtime)
        output_ply = result_dir / "ply" / f"{input_ply.stem}_cleaned.ply"
        
        logger.info(f"Input: {input_ply}")
        logger.info(f"Output: {output_ply}")
        
        # Import and run pruner
        try:
            pruner_script = Path(__file__).resolve().parent.parent / "post" / "dqn_pruner" / "prune.py"
            sys.path.insert(0, str(pruner_script.parent))
            
            from prune import prune_ply
            
            success = prune_ply(
                str(input_ply),
                str(output_ply),
                self.config.PRUNE_MODEL  # None = use heuristic
            )
            
            if success:
                logger.info(f"Floater pruning complete: {output_ply}")
            else:
                logger.warning("Floater pruning failed.")
                
        except ImportError as e:
            logger.error(f"Could not import DQN pruner: {e}")
            logger.info("Skipping pruning step.")
        except Exception as e:
            logger.error(f"Pruning error: {e}")
            logger.info("Skipping pruning step.")

    def _upload_to_cloud(self, result_dir: Path) -> dict:
        """Step 3: Upload results to Cloudflare R2 (optional)"""
        logger.info("\nSTEP 3: CLOUD SYNC (R2)")
        
        try:
            from .r2_uploader import R2Uploader
        except ImportError:
            try:
                from r2_uploader import R2Uploader
            except ImportError:
                sys.path.append(str(Path(__file__).parent))
                from r2_uploader import R2Uploader
        
        uploader = R2Uploader()
        
        if not uploader.is_configured:
            logger.warning("R2 not configured. Skipping cloud sync.")
            logger.info("Run 'python scripts/pipeline/r2_uploader.py setup' to configure.")
            return {"status": "skipped", "reason": "not_configured"}
        
        # Generate scene name from video or use provided
        scene_name = self.config.SCENE_NAME or f"scene_{int(time.time())}"
        
        result = uploader.upload_gsplat_results(
            result_dir,
            scene_name,
            include_checkpoints=False  # Save space, upload only .ply
        )
        
        if result.get("skipped"):
            return {"status": "skipped", "reason": "not_configured"}
        
        logger.info(f"Cloud sync complete: {result['uploaded']} files uploaded")
        return {"status": "success", "uploaded": result["uploaded"], "scene": scene_name}
    
    def _print_success(self, elapsed: float, result_dir: Path):
        print("\n" + "="*70)
        print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        print(f"Artifacts: {result_dir}")
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Condensed Intelligent 3DPipeline")
    
    # Core
    parser.add_argument("video_path", help="Input video file")
    parser.add_argument("--output_dir", default="data/output_3dgs", help="Output root")
    parser.add_argument("--fps", type=float, default=10.0, help="Extraction FPS")
    parser.add_argument("--data_factor", type=int, default=2, help="Downsample factor")
    
    # Training
    parser.add_argument("--max_steps", type=int, default=7000, help="Training steps")
    parser.add_argument("--with_ut", action="store_true", help="Uncertainty Training")
    parser.add_argument("--with_eval3d", action="store_true", help="3D Evaluation")
    
    # Tuning
    parser.add_argument("--init_scale", type=float, default=2.5)
    parser.add_argument("--opacity_reg", type=float, default=0.05)
    parser.add_argument("--scale_reg", type=float, default=0.05)
    parser.add_argument("--app_opt", action="store_true")
    
    # Cloud Storage
    parser.add_argument("--cloud-sync", dest="cloud_sync", action="store_true", 
                        help="Upload results to Cloudflare R2")
    parser.add_argument("--scene-name", dest="scene_name", type=str, default=None,
                        help="Scene name for cloud upload (auto-generated if not set)")
    
    # Streaming Mode
    parser.add_argument("--streaming", action="store_true",
                        help="Enable streaming mode: start 3DGS training while ACE-Zero is running")
    
    # Floater Pruning
    parser.add_argument("--prune", action="store_true", default=True,
                        help="Run floater pruning post-processing (default: True)")
    parser.add_argument("--no-prune", dest="prune", action="store_false",
                        help="Disable floater pruning")
    parser.add_argument("--prune-model", dest="prune_model", type=str, default=None,
                        help="Path to trained DQN pruner model (uses heuristic if not set)")
    
    # GeNVS-Lite Novel View Augmentation
    parser.add_argument("--genvs", action="store_true",
                        help="Enable GeNVS-Lite to generate synthetic novel views for data augmentation")
    parser.add_argument("--genvs-views", dest="genvs_views", type=int, default=20,
                        help="Number of novel views to generate (default: 20)")
    parser.add_argument("--skip-ace", action="store_true", help="Skip ACE-Zero pose estimation (use existing)")
    
    # JOGS-style Pose Optimization
    parser.add_argument("--pose-opt", dest="pose_opt", action="store_true", default=True,
                        help="Enable JOGS-style joint pose-3DGS optimization (default: True)")
    parser.add_argument("--no-pose-opt", dest="pose_opt", action="store_false",
                        help="Disable pose optimization (use fixed ACE-Zero poses)")
    parser.add_argument("--pose-opt-warmup", dest="pose_opt_warmup", type=int, default=1000,
                        help="Steps before pose optimization starts (default: 1000)")

    # Autoregressive GeNVS
    parser.add_argument("--genvs-autoregressive", dest="genvs_autoregressive", action="store_true",
                        help="Enable 'Dream Loop': Train -> Render Back -> Refine(GeNVS) -> Retrain")

    # Debug/Overrides
    parser.add_argument("--colmap-input", dest="colmap_input", type=str, default=None,
                        help="Override input COLMAP directory (e.g. for testing existing datasets)")
    
    # === NEW: ACE-Zero Quality Mode ===
    parser.add_argument("--quality-mode", dest="quality_mode", type=str, default="balanced",
                        choices=["fast", "balanced", "quality"],
                        help="ACE-Zero quality mode: fast (speed), balanced (default), quality (max accuracy)")
    
    # === NEW: Pose Confidence Filtering ===
    parser.add_argument("--min-confidence", dest="min_confidence", type=int, default=1000,
                        help="Minimum pose confidence threshold (poses below this are filtered)")
    
    # === NEW: Early Stopping ===
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true", default=True,
                        help="Enable early stopping based on loss convergence (default: True)")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false",
                        help="Disable early stopping")
    parser.add_argument("--early-stop-patience", dest="early_stop_patience", type=int, default=500,
                        help="Steps to wait for loss improvement before stopping (default: 500)")
    parser.add_argument("--early-stop-min-delta", dest="early_stop_min_delta", type=float, default=0.001,
                        help="Minimum loss improvement to reset patience (default: 0.001)")
    parser.add_argument("--early-stop-min-steps", dest="early_stop_min_steps", type=int, default=2000,
                        help="Minimum steps before early stopping can trigger (default: 2000)")
    
    # === NEW: Depth Model Selection ===
    parser.add_argument("--depth-model", dest="depth_model", type=str, default="depth_anything",
                        choices=["depth_anything", "zoedepth", "midas"],
                        help="Depth estimator: depth_anything (~30ms, fast), zoedepth (~200ms, metric), midas (~100ms)")

    args = parser.parse_args()
    
    try:
        config = PipelineConfig(args)
        pipeline = IntelligentPipeline(config)
        pipeline.run(args.video_path)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
