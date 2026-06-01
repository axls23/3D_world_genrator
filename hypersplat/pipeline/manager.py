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
import shutil
import numpy as np
import cv2
import torch

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



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
        self.FPS = args.fps
        self.DATA_FACTOR = args.data_factor
        
        # Training Settings
        self.TRAINING_MAX_STEPS = args.max_steps
        self.TRAINING_EVAL_STEPS = getattr(args, 'eval_steps', [3000, 7000])
        self.TRAINING_SAVE_STEPS = getattr(args, 'save_steps', [3000, 7000])
        
        # Quality/Tuning
        self.INIT_SCALE = args.init_scale
        self.OPACITY_REG = args.opacity_reg
        self.SCALE_REG = args.scale_reg
        self.USE_APP_OPT = args.app_opt
        
        # Advanced Hyperparameters
        self.SH_DEGREE = args.sh_degree
        self.MEANS_LR = args.means_lr
        self.SSIM_LAMBDA = args.ssim_lambda
        self.RANDOM_BKGD = args.random_bkgd
        
        # Advanced Modules
        self.WITH_UT = args.with_ut
        self.WITH_EVAL3D = args.with_eval3d
        
        # [ACE-Zero Options]
        self.SKIP_ACE = getattr(args, 'skip_ace', False)
        self.COLMAP_INPUT = getattr(args, 'colmap_input', None)
        self.QUALITY_MODE = getattr(args, 'quality_mode', 'balanced')
        # Pose filtering threshold — passed to ACEZeroPoseEstimator._filter_poses()
        self.MIN_REGISTRATION_CONFIDENCE = getattr(args, 'min_registration_confidence', 1000) # ACE-Zero threshold (int)
        self.DEPTH_MODEL = getattr(args, 'depth_model', 'depth_anything') # Default depth model

        # Cloud Storage
        self.CLOUD_SYNC = getattr(args, 'cloud_sync', False)
        self.SCENE_NAME = getattr(args, 'scene_name', None)
        
        # Streaming Mode
        self.STREAMING = getattr(args, 'streaming', False)
        
        # Floater Pruning
        self.PRUNE = getattr(args, 'prune', True)  # Default ON
        
        # [Unified Stream override]
        if getattr(args, 'unified_stream', False):
            self.STREAMING = True
            logger.info("[Config] Unified Stream enabled -> Forcing STREAMING=True")
            # If explicit quality mode wasn't provided (still default 'balanced'), force 'fast'
            if self.QUALITY_MODE == 'balanced':
                 self.QUALITY_MODE = 'fast'
                 logger.info("[Config] Unified Stream enabled -> Forcing QUALITY_MODE='fast' for speed")

        # Early Stopping
        self.EARLY_STOPPING = getattr(args, 'early_stopping', True)
        self.EARLY_STOP_PATIENCE = getattr(args, 'early_stop_patience', 500)
        self.EARLY_STOP_MIN_DELTA = getattr(args, 'early_stop_min_delta', 0.001)
        self.EARLY_STOP_MIN_STEPS = getattr(args, 'early_stop_min_steps', 2000)

        self.PRUNE_MODEL = getattr(args, 'prune_model', None)  # Optional DQN model path
        
        # GeNVS-Lite Novel View Generation
        self.GENVS_ENABLED = getattr(args, 'genvs', False)  # Augment with synthetic views
        self.GENVS_NUM_VIEWS = getattr(args, 'genvs_views', 20)  # Number of novel views
        self.GENVS_CKPT = getattr(args, 'genvs_ckpt', None)

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


        # Streaming mode: dynamically accept new poses during training
        # Removed redundant STREAMING reset that ignored unified_stream

        # How often to check for new poses (in training steps)
        self.STREAMING_INTERVAL = getattr(args, 'streaming_check_interval', 1000)

        # GeNVS Online Integration
        self.GENVS_INTERVAL = getattr(args, 'genvs_interval', 1000)

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
        
        # Resolve video path using helper function (must exist and be a file)
        video_path = resolve_path_relative_to_project(video_path, must_exist=True, must_be_file=True)
        
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("INTELLIGENT PIPELINE: ACE-Zero + 3DGS")
        logger.info("="*70)
        logger.info(f"Input: {video_path}")
        logger.info(f"Output: {self.config.OUTPUT_BASE}")

        self.config.create_directories()

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
                if self.config.STREAMING:
                    logger.info("STREAMING MODE: Launching ACE-Zero pose estimation in a background thread...")
                    import threading
                    ace_output = self.config.OUTPUT_BASE / "acezero_output"
                    
                    # Clear any stale sparse folder to prevent resuming on bad data
                    sparse_dir = ace_output / "sparse" / "0"
                    if sparse_dir.exists():
                        shutil.rmtree(sparse_dir, ignore_errors=True)
                        
                    pose_thread = threading.Thread(
                        target=self._run_pose_estimation, 
                        args=(video_path,),
                        daemon=True
                    )
                    pose_thread.start()
                    
                    # Wait for frame extraction and the first seed poses file
                    logger.info("Waiting for frame extraction and initial seed pose...")
                    start_wait = time.time()
                    acezero_result_dir = ace_output / "acezero_output"
                    
                    found_seed = False
                    while not found_seed:
                        if acezero_result_dir.exists():
                            pose_files = list(acezero_result_dir.glob("poses_*.txt"))
                            if pose_files:
                                found_seed = True
                                break
                        time.sleep(2.0)
                        if time.time() - start_wait > 900: # 15 minutes timeout
                            raise RuntimeError("Timeout waiting for ACE-Zero to register the seed frame.")
                            
                    logger.info("Seed poses detected! Bootstrapping initial COLMAP binaries...")
                    # Perform initial conversion using the watcher
                    import sys
                    from pathlib import Path
                    scripts_pipeline = Path(__file__).resolve().parent.parent.parent / "scripts" / "pipeline"
                    if str(scripts_pipeline) not in sys.path:
                        sys.path.insert(0, str(scripts_pipeline))
                    from pose_watcher import PoseWatcher
                    
                    watcher = PoseWatcher(ace_output)
                    watcher.check_for_updates()
                    
                    # Create downsampled images since 3DGS trainer needs resized versions if data_factor > 1
                    from hypersplat.pipeline.wrappers.perception import ACEZeroPoseEstimator
                    temp_estimator = ACEZeroPoseEstimator(output_dir=ace_output)
                    if hasattr(self, 'use_wsl'):
                        temp_estimator.use_wsl = self.use_wsl
                    temp_estimator.create_downsampled_images(factors=[2, 4])
                    
                    logger.info("Initial COLMAP structure ready. Proceeding to 3DGS training.")
                else:
                    ace_output = self._run_pose_estimation(video_path)
            
            # 1.5. GeNVS Novel View Generation (optional)
            if self.config.GENVS_ENABLED:
                ace_output = self._run_novel_view_generation(ace_output)
            
            # 2. Training
            result_dir = self._run_training(ace_output)
            
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


    def _clear_gpu_memory(self):
        """Aggressively collect garbage and empty PyTorch CUDA cache."""
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)
            torch.cuda.empty_cache()
            allocated_after = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)
            logger.info(
                f"[VRAM Telemetry] Freed VRAM: "
                f"{allocated_before:.2f}MB -> {allocated_after:.2f}MB allocated, "
                f"{reserved_before:.2f}MB -> {reserved_after:.2f}MB reserved"
            )

    def _run_pose_estimation(self, video_path: Path) -> Path:
        """Step 1: AI-Powered Pose Estimation"""
        logger.info("\nSTEP 1: POSES (ACE-Zero)")
        logger.info(f"Quality Mode: {self.config.QUALITY_MODE}")
        
        # Lazy import to keep startup fast
        try:
            from hypersplat.pipeline.wrappers.perception import ACEZeroPoseEstimator
        except ImportError as e1:
            logger.error(f"Failed to import from hypersplat: {e1}")
            try:
                # Fallback for relative import if running as module
                from .wrappers.perception import ACEZeroPoseEstimator
            except ImportError as e2:
                 raise ImportError(f"Could not load ACEZeroPoseEstimator: {e2}")

        output_dir = self.config.OUTPUT_BASE / "acezero_output"
        logger.info(f"Depth Model: {self.config.DEPTH_MODEL}")
        estimator = ACEZeroPoseEstimator(
            output_dir=output_dir,
            quality_mode=self.config.QUALITY_MODE,
            min_confidence=self.config.MIN_REGISTRATION_CONFIDENCE,
            depth_model=self.config.DEPTH_MODEL
        )
        
        success = estimator.process_video(str(video_path), fps=self.config.FPS)
        
        if not success:
            raise RuntimeError("ACE-Zero Pose Estimation failed.")

        # Validation
        points_bin = output_dir / "sparse" / "0" / "points3D.bin"
        if not points_bin.exists() or points_bin.stat().st_size < 100:
            raise RuntimeError(f"Invalid point cloud generated at {points_bin}")

        logger.info("Pose estimation valid.")
        return output_dir

    def _run_novel_view_generation(self, colmap_dir: Path) -> Path:
        """Step 1.5: GeNVS-Lite Novel View Generation to Augment Dataset"""
        logger.info("\nSTEP 1.5: GENVS-LITE (Novel View Synthesis)")
        logger.info(f"Generating {self.config.GENVS_NUM_VIEWS} synthetic views...")
        
        # Import GeNVS components
        pipeline = None
        genvs = None
        ref_tensor = None
        generated_batch = None
        novel_views = None
        
        try:
            from hypersplat.beta.genvs.run_completion import GeNVSLite, inject_back_views
            
            # Step A: Choose implementation based on Quality Mode
            input_images_dir = colmap_dir / "images"
            novel_output_dir = self.config.OUTPUT_BASE / "novel_views"
            novel_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  Input: {input_images_dir}")
            logger.info(f"  Output: {novel_output_dir}")
            
            # Check for True GeNVS Core usage
            use_core_genvs = (self.config.QUALITY_MODE == 'quality') or getattr(self.config, 'USE_TRUE_GENVS', False)
            
            if use_core_genvs:
                logger.info("  [GeNVS] Using GeNVS-Core (True 3D-Aware Diffusion)")
                try:
                    # Import our new core pipeline
                    from hypersplat.beta.genvs.pipeline import GeNVSPipeline
                    
                    # Initialize with model_channels=64 to match trained checkpoints
                    pipeline = GeNVSPipeline(device='cuda', model_channels=64)
                    
                    # === CHECKPOINT LOADING ===
                    # Priority: explicit CLI path > auto-detect latest
                    ckpt_path = None
                    if self.config.GENVS_CKPT and Path(self.config.GENVS_CKPT).exists():
                        ckpt_path = self.config.GENVS_CKPT
                    else:
                        # Auto-detect latest checkpoint
                        ckpt_dir = Path("results/genvs_train/checkpoints")
                        if ckpt_dir.exists():
                            ckpts = sorted(ckpt_dir.glob("step_*.pt"), 
                                         key=lambda p: int(p.stem.split("_")[1]))
                            if ckpts:
                                ckpt_path = str(ckpts[-1])
                    
                    if ckpt_path:
                        step = pipeline.load_checkpoint(ckpt_path)
                        logger.info(f"  [GeNVS-Core] Loaded trained checkpoint: {ckpt_path} (step {step})")
                    else:
                        logger.error("  [GeNVS-Core] NO CHECKPOINT FOUND! Model has random weights.")
                        logger.error("  [GeNVS-Core] Train first: python hypersplat/beta/genvs/train.py --data_dir <path>")
                        logger.error("  [GeNVS-Core] Or specify: --genvs_ckpt <path/to/step_XXXX.pt>")
                        raise FileNotFoundError("GeNVS checkpoint required but not found. Train the model first.")
                    
                    # === REAL POSE INTEGRATION ===
                    from hypersplat.beta.genvs.dataset import GenVSDataset
                    
                    # 1. Load the real dataset from the ACE output
                    # This handles pose loading, focal lengths, and scene normalization
                    # Determine poses_final.txt location (at root or in acezero_output/)
                    pose_file = "poses_final.txt"
                    if not (colmap_dir / pose_file).exists() and (colmap_dir / "acezero_output" / pose_file).exists():
                        pose_file = "acezero_output/poses_final.txt"
                    
                    dataset = GenVSDataset(root_dir=colmap_dir, pose_file=pose_file, image_dir="images", image_size=self.config.GENVS_IMAGE_SIZE if hasattr(self.config, 'GENVS_IMAGE_SIZE') else 128)
                    
                    # 2. Select Reference View
                    ref_idx = len(dataset) // 2
                    ref_item = dataset.load_view(ref_idx)
                    
                    ref_tensor = ref_item['image'].unsqueeze(0).to('cuda')
                    source_pose = ref_item['pose'].unsqueeze(0).to('cuda')
                    source_K = ref_item['K'].unsqueeze(0).to('cuda')
                    
                    # Use real dimensions from the dataset loading
                    H, W = ref_item['image'].shape[1:]
                    
                    logger.info(f"  [GeNVS] Reference pose loaded (Focal: {ref_item['K'][0,0]:.2f})")
                    
                    # Generate Target Poses (Orbit)
                    num_views = self.config.GENVS_NUM_VIEWS
                    target_poses = []
                    target_Ks = []
                    
                    for i in range(num_views):
                        # Simple rotation around Y axis in the local camera frame
                        angle = (i / num_views) * 2 * np.pi 
                        
                        # Orbit radius: small perturbation for better coverage
                        c, s = np.cos(angle), np.sin(angle)
                        R_y = torch.tensor([
                            [c, 0, s, 0],
                            [0, 1, 0, 0],
                            [-s, 0, c, 0],
                            [0, 0, 0, 1]
                        ]).float().to('cuda')
                        
                        target_poses.append(source_pose.squeeze(0) @ R_y)
                        target_Ks.append(source_K.squeeze(0))
                        
                    target_poses_stack = torch.stack(target_poses) # [N, 4, 4]
                    target_Ks_stack = torch.stack(target_Ks)       # [N, 3, 3]
                    
                    # Check batch size VRAM capability. 4 is safe for 12GB.
                    batch_size = 4
                    
                    logger.info(f"  [GeNVS-Core] Running batched generation (Batch Size: {batch_size})...")
                    
                    generated_batch = pipeline.sample_batch(
                        ref_tensor, source_pose, source_K, 
                        target_poses_stack, target_Ks_stack, 
                        batch_size=batch_size
                    ) # [N, 3, H, W]
                    
                    # Save results
                    for i in range(num_views):
                        img_tensor = generated_batch[i]
                        # [-1, 1] -> [0, 255]
                        img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
                        
                        out_path = novel_output_dir / f"novel_{i:04d}.png"
                        cv2.imwrite(str(out_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                        
                    logger.info(f"  [GeNVS-Core] Generated {num_views} high-fidelity views.")
                    
                except ImportError as e:
                    logger.warning(f"  [GeNVS] Failed to load Core: {e}. Falling back to Lite.")
                    use_core_genvs = False
                except Exception as e:
                    logger.error(f"  [GeNVS-Core] Execution failed: {e}")
                    # Don't fallback, just fail the novel view part but continue pipeline?
                    # Or fallback to Lite? Let's fallback.
                    use_core_genvs = False

            if not use_core_genvs:
                # Initialize GeNVS-Lite (uses ZoeDepth by default)
                genvs = GeNVSLite(use_zoedepth=True)
                
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
        finally:
            logger.info("Cleaning up GeNVS model resources...")
            del pipeline
            del genvs
            del ref_tensor
            del generated_batch
            del novel_views
            self._clear_gpu_memory()

    def _run_training(self, data_dir: Path, resume_ckpt: Optional[Path] = None) -> Path:
        """Step 2: MCMC 3DGS Training"""
        logger.info("\nSTEP 2: TRAINING (3DGS MCMC)")
        
        result_dir = self.config.OUTPUT_BASE / "results" / "acezero_3dgs"
        trainer_script = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_trainer.py"
        
        if not trainer_script.exists():
             # Fallback location check
             trainer_script = Path("examples/simple_trainer.py")

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
            
            # New Hyperparameters
            "--sh_degree", str(self.config.SH_DEGREE),
            "--means_lr", str(self.config.MEANS_LR),
            "--ssim_lambda", str(self.config.SSIM_LAMBDA),
        ]

        if self.config.TRAINING_SAVE_STEPS:
            cmd.append("--save_steps")
            cmd.extend([str(s) for s in self.config.TRAINING_SAVE_STEPS])
        
        if self.config.TRAINING_EVAL_STEPS:
            cmd.append("--eval_steps")
            cmd.extend([str(s) for s in self.config.TRAINING_EVAL_STEPS])

        if self.config.RANDOM_BKGD: cmd.append("--random_bkgd")
        if self.config.USE_APP_OPT: cmd.append("--app_opt")
        if self.config.WITH_UT: cmd.append("--with_ut")
        if self.config.WITH_EVAL3D: cmd.append("--with_eval3d")
        if self.config.STREAMING: 
             cmd.append("--streaming")
             cmd.extend(["--streaming_check_interval", str(self.config.STREAMING_INTERVAL)])
        
        # JOGS-style joint pose-3DGS optimization
        if self.config.POSE_OPT:
            cmd.extend(["--pose_opt", "--pose_warmup_steps", str(self.config.POSE_OPT_WARMUP)])

        if self.config.GENVS_CKPT:
             cmd.extend(["--genvs_ckpt", str(self.config.GENVS_CKPT)])
             cmd.extend(["--genvs_interval", str(self.config.GENVS_INTERVAL)])

        if resume_ckpt and resume_ckpt.exists():
            cmd.extend(["--ckpt", str(resume_ckpt)])
            logger.info(f"  [Resume] Continuing from checkpoint: {resume_ckpt.name}")

        logger.info(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        # Stream output
        line_count = 0
        for line in process.stdout:
            print(f"[Train] {line.strip()}")
            line_count += 1
            if line_count <= 5:  # Log first 5 lines to logger
                logger.debug(f"[Train] {line.strip()}")
            
        exit_code = process.wait()
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
        import yaml
        
        from hypersplat.beta.genvs.run_completion import GeNVSLite, read_model, mirror_camera_pose, compute_scene_center, qvec2rotmat
        
        # [Governor Loop]
        max_loops = 3
        current_data_dir = ace_output
        current_result_dir = initial_result_dir
        
        for i in range(max_loops):
            logger.info(f"\n--- Autoregressive Loop {i+1}/{max_loops} ---")
            
            # 1. Ask Governor for marching orders
            action, overrides = self._run_governor(current_data_dir, current_result_dir)
            
            if action == "STOP":
                logger.info("Governor decided to STOP refinement.")
                break
                
            logger.info(f"Governor Action: {action} (Overrides: {overrides})")
            
            # 2. Execute Refinement Steps (Sampling -> GeNVS -> Injection)
            # For V1, we simply perform the standard refinement but with Governor's tuning
            # (In future, 'action' could trigger completely different functions)
            
            # [Logic from previous static implementation, now parameterized]
            iter_dir = self.config.OUTPUT_BASE / f"autoregressive_iter{i+1}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            
            augmented_dir = self._perform_genvs_refinement(
                current_data_dir, 
                current_result_dir, 
                iter_dir,
                overrides # Pass governor config to GeNVS?
            )
            
            # 3. Re-Train with Governor's Overrides
            # We need to pass overrides to _run_training. 
            # Currently _run_training reads from self.config.
            # We temporarily swap self.config with a patched copy
            import copy
            original_config = self.config
            self.config = self._patch_config(original_config, overrides)
            
            try:
                current_result_dir = self._run_training(augmented_dir)
                current_data_dir = augmented_dir
            finally:
                self.config = original_config
                
        return current_result_dir

    def _run_governor(self, data_dir: Path, result_dir: Path):
        """Invoke the DQN Governor to decide next steps."""
        governor_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "post" / "dqn_pruner" / "governor.py"
        
        # We pass the directory containing the *current state* (results of last run)
        # The Governor expects to find 'acezero_output' and 'results' parent structure?
        # Actually governor expects inputs. Our structure is:
        # OUTPUT_BASE/
        #   acezero_output/
        #   results/acezero_3dgs/
        
        # Let's pass the RESULT dir, governor can navigate up.
        cmd = [sys.executable, str(governor_script), str(result_dir)]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Read output
            cfg_path = result_dir / "cfg_governor.yml"
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get("action", "STEADY"), data.get("overrides", {})
        except Exception as e:
            logger.warning(f"Governor execution failed: {e}. Defaulting to STEADY.")
            
        return "STEADY", {}

    def _patch_config(self, base_config, overrides: Dict):
        """Create a patched copy of config without mutating the original."""
        import copy
        new_config = copy.copy(base_config)
        for k, v in overrides.items():
            # Map snake_case overrides to UPPER_CASE config keys if needed
            key = k.upper()
            if hasattr(new_config, key):
                logger.info(f"  [Config Override] {key}: {getattr(new_config, key)} -> {v}")
                setattr(new_config, key, v)
        return new_config

    def _perform_genvs_refinement(self, ace_output, initial_result_dir, iter_dir, overrides) -> Path:
        import torch
        import cv2
        import numpy as np

        if overrides.get("skip_genvs", False):
             logger.info("  Skipping GeNVS refinement as requested (Wait/Fine-tune mode).")
             return ace_output

        try:
             from hypersplat.beta.genvs.run_completion import (
                 GeNVSLite, read_model, mirror_camera_pose, 
                 compute_scene_center, qvec2rotmat, inject_autoregressive_back_views
             )
        except ImportError as e:
             logger.error(f"Failed to import GeNVS utilities from beta/genvs: {e}")
             raise e
             
        # Imports handled above
        
        # 1. Compute Pose (Mirror of center view)
        sparse_in = ace_output / "sparse" / "0"
        cameras, images,Points3D = read_model(str(sparse_in))
        
        ref_image_id = list(images.keys())[len(images) // 2]
        ref_image = images[ref_image_id]
        scene_center = compute_scene_center(images)
        
        qvec, tvec = mirror_camera_pose(ref_image.qvec, ref_image.tvec, scene_center)
        R_w2c = qvec2rotmat(qvec)
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ tvec
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w
        
        cam = cameras[ref_image.camera_id]
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

        pose_data = {
            "c2w": c2w.tolist(),
            "K": K,
            "width": w,
            "height": h
        }
        pose_file = iter_dir / "target_pose.json"
        with open(pose_file, 'w') as f:
            json.dump(pose_data, f)
            
        # 2. Render 'Draft' View
        logger.info("  Sampling 3D latent space (Rendering)...")
        trainer_script = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_trainer.py"
        ckpt_dir = initial_result_dir / "ckpts"
        ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: int(p.stem.split('_')[1]))
        latest_ckpt = ckpts[-1]
        
        cmd_render = [
            sys.executable, str(trainer_script), "mcmc",
            "--result_dir", str(initial_result_dir),
            "--data_dir", str(ace_output),
            "--data_factor", "1",
            "--ckpt", str(latest_ckpt),
            "--render_pose_file", str(pose_file),
            "--disable_viewer"
        ]
        subprocess.run(cmd_render, check=True)
        
        rendered_img_path = initial_result_dir / "renders" / "custom_render.png"
        
        # 3. GeNVS Refinement
        logger.info("  Refining sample with GeNVS...")
        
        genvs = None
        src_img_t = None
        src_pose_t = None
        novel_view_np = None
        
        try:
            # Use the already imported GeNVSLite
            genvs = GeNVSLite() # No use_zoedepth in our current implementation
            
            front_img_bgr = cv2.imread(str(ace_output / "images" / ref_image.name))
            front_img = cv2.cvtColor(front_img_bgr, cv2.COLOR_BGR2RGB)
            
            # Use our generate_augmented_view method
            # We need source_img (tensor), source_pose, source_K, target_pose
            # For simplicity in this pipeline wrapper, we can just use the provided method
            # if we wrap it correctly or update run_completion to have the expected API.
            
            # Let's adjust run_completion.py to have generate_novel_views for compatibility 
            # OR just use generate_augmented_view here.
            
            # Actually, let's keep manager.py clean and fix run_completion.py to expose what it needs.
            # But for now, I'll match what I wrote in run_completion.
            
            source_item = {
                "image": torch.from_numpy(front_img).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0,
                "pose": torch.from_numpy(ref_image.qvec2rotmat()).float(), # placeholder, needs 4x4
                "K": torch.tensor(K).float()
            }
            # Source Pose (The original frame)
            R_src_w2c = qvec2rotmat(ref_image.qvec)
            R_src_c2w = R_src_w2c.T
            t_src_c2w = -R_src_c2w @ ref_image.tvec
            src_c2w = np.eye(4)
            src_c2w[:3, :3] = R_src_c2w
            src_c2w[:3, 3] = t_src_c2w
            
            src_img_t = torch.from_numpy(front_img).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
            src_pose_t = torch.from_numpy(src_c2w).float()
            
            novel_view_np = genvs.generate_augmented_view(
                source_img=src_img_t,
                source_pose=src_pose_t,
                source_K=torch.tensor(K).float(),
                target_pose=torch.from_numpy(c2w).float() # Using the mirrored c2w we computed
            )
            
            refined_path = iter_dir / "refined_back_0000.png"
            cv2.imwrite(str(refined_path), cv2.cvtColor(novel_view_np, cv2.COLOR_RGB2BGR))
            
            # 4. Inject into Dataset
            logger.info("  Injecting refined view...")
            augmented_dir = self.config.OUTPUT_BASE / f"acezero_output_autoregressive_{iter_dir.name}"
            
            temp_novel_dir = iter_dir / "novel_temp"
            temp_novel_dir.mkdir(exist_ok=True)
            shutil.copy2(refined_path, temp_novel_dir / "novel_0.png")
            
            # Use the already imported inject_autoregressive_back_views
            inject_autoregressive_back_views(
                ace_output=ace_output,
                initial_result_dir=initial_result_dir,
                iter_dir=iter_dir,
                overrides=overrides
            )
            return augmented_dir
        finally:
            logger.info("Cleaning up GeNVS model resources in autoregressive loop...")
            del genvs
            del src_img_t
            del src_pose_t
            del novel_view_np
            self._clear_gpu_memory()

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
            pruner_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "post" / "dqn_pruner" / "prune.py"
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
            from hypersplat.services.storage.r2 import R2Uploader
        except ImportError:
            try:
                 # Fallback: Check if r2_uploader is still in scripts/pipeline (for backward compat or mixed state)
                 sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "scripts" / "pipeline"))
                 from r2_uploader import R2Uploader
            except ImportError:
                 raise ImportError("Could not find R2Uploader in hypersplat.services.storage.r2 or scripts/pipeline")
        
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

    # Advanced Hyperparameters
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")
    parser.add_argument("--means_lr", type=float, default=0.00016, help="Primary learning rate")
    parser.add_argument("--ssim_lambda", type=float, default=0.2, help="SSIM loss weight")
    parser.add_argument("--random_bkgd", action="store_true", help="Use random background during training")
    
    # Cloud Storage
    parser.add_argument("--cloud-sync", dest="cloud_sync", action="store_true", 
                        help="Upload results to Cloudflare R2")
    parser.add_argument("--scene-name", dest="scene_name", type=str, default=None,
                        help="Scene name for cloud upload (auto-generated if not set)")
    
    parser.add_argument("--streaming", action="store_true",
                        help="Enable streaming mode: start 3DGS training while ACE-Zero is running")
    parser.add_argument("--unified_stream", action="store_true",
                        help="Alias for --streaming --quality-mode fast")
    parser.add_argument("--streaming_check_interval", type=int, default=1000,
                        help="Interval to check for new poses in streaming mode (default: 1000)")

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
    parser.add_argument("--genvs_ckpt", type=str, default=None, help="Path to GeNVS checkpoint for Autoregressive Injection")
    parser.add_argument("--genvs_interval", type=int, default=1000, help="Interval for GeNVS injection (default: 1000)")
    parser.add_argument("--skip-ace", action="store_true", help="Skip ACE-Zero pose estimation (use existing)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

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
    parser.add_argument("--min-registration-confidence", dest="min_registration_confidence", type=int, default=1000,
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
    
    # Global Seed Initialization
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Set global random seed to {args.seed}")

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
