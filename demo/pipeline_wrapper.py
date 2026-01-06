"""
Pipeline Wrapper for Demo Server
Level 2: ACE-Zero Direct + 3DGS MCMC Training

Pipeline Modes:
  - Level 1: Simple Training (COLMAP-based, requires pre-processed data)
  - Level 2: ACE-Zero Direct (AI Poses → 3DGS MCMC) [DEFAULT]
  - Level 3: Full Chunked Pipeline (Video → Chunks → COLMAP → 3DGS per chunk)
"""

import os
import subprocess
import threading
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any


class PipelineMode:
    """Pipeline execution modes"""
    LEVEL_1_SIMPLE = "simple"           # Direct training on existing data
    LEVEL_2_ACE_ZERO = "ace_zero"       # ACE-Zero poses → 3DGS (DEFAULT)
    LEVEL_3_CHUNKED = "chunked"         # Full chunked pipeline with COLMAP


class TrainingManager:
    """
    Manages video-to-3DGS training pipeline with multiple modes.
    
    Level 2 (Default): ACE-Zero Direct
    - Uses ACE-Zero for COLMAP-free pose estimation
    - AI depth initialization with Depth Anything V2 (6x faster than ZoeDepth)
    - MCMC-based 3DGS training with 3DGUT support
    """
    
    def __init__(self, root_dir: Path, output_dir: Path):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.scripts_dir = self.root_dir / "scripts" / "pipeline"
        
        # State
        self.current_video_path = None
        self.process = None
        self.logs = []
        self.status = "idle"  # idle, training_running, training_complete, error
        self.progress = 0
        self.training_viewer_url = None
        self.log_file_path = self.root_dir / "pipeline_log.txt"
        self.current_mode = PipelineMode.LEVEL_2_ACE_ZERO
        
        # Training config
        self.training_config = {
            "max_steps": 7000,
            "data_factor": 2,
            "fps": 2,  # Frames per second for video extraction
            "with_ut": False,
            "with_eval3d": False,
            "save_ply": True,
            "disable_viewer": False,
            "eval_steps": [3000, 7000],
            # Early stopping config
            "early_stopping": True,
            "loss_patience": 500,  # Steps to wait for improvement
            "loss_threshold": 0.001,  # Min improvement to reset patience
            "min_steps": 2000,  # Minimum steps before early stopping
            # Zero123 mode for GeNVS
            "use_zero123": False,
        }
        
        # Loss tracking for early stopping
        self.loss_history = []
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        
    def start_training(
        self, 
        video_path: str, 
        mode: str = PipelineMode.LEVEL_2_ACE_ZERO,
        config_override: Optional[Dict[str, Any]] = None
    ):
        """
        Start training pipeline.
        
        Args:
            video_path: Path to input video
            mode: Pipeline mode (simple, ace_zero, chunked)
            config_override: Optional config overrides
        """
        if self.status == "training_running":
            raise RuntimeError("Training already in progress")
        
        self.current_video_path = video_path
        self.current_mode = mode
        self.status = "training_running"
        self.logs = []
        self.progress = 0
        self.training_viewer_url = None
        
        # Apply config overrides
        if config_override:
            self.training_config.update(config_override)
        
        # Reset early stopping state
        self.loss_history = []
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        
        # Validate and log fps
        self.validate_fps_config(video_path)
        
        # Initialize log file
        try:
            with open(self.log_file_path, "w", encoding='utf-8') as f:
                f.write(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mode: {mode}\n")
                f.write(f"Video: {video_path}\n")
                f.write(f"Config: {self.training_config}\n")
        except Exception as e:
            print(f"Failed to create log file: {e}")
        
        # Select pipeline based on mode
        if mode == PipelineMode.LEVEL_1_SIMPLE:
            return self._run_simple_training(video_path)
        elif mode == PipelineMode.LEVEL_2_ACE_ZERO:
            return self._run_ace_zero_pipeline(video_path)
        else:  # LEVEL_3_CHUNKED
            return self._run_chunked_pipeline(video_path)
    
    def _run_ace_zero_pipeline(self, video_path: str):
        """
        Level 2: ACE-Zero Direct Pipeline
        
        Step 1: ACE-Zero Pose Estimation
        Step 2: 3DGS MCMC Training
        """
        self.logs.append("=== LEVEL 2: ACE-ZERO PIPELINE ===")
        
        # Setup output directories (uses unified output structure)
        acezero_output = self.output_dir / "acezero"
        result_dir = self.output_dir / "results"
        
        acezero_output.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the combined pipeline command
        pipeline_script = self.scripts_dir / "automated_intelligent_pipeline.py"
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        cmd = [
            sys.executable, "-u", str(pipeline_script),
            video_path,
            "--output_dir", str(self.output_dir),
            "--fps", str(self.training_config["fps"]),
            "--max_steps", str(self.training_config["max_steps"]),
            "--data_factor", str(self.training_config["data_factor"]),
        ]
        
        # 3DGUT options
        if self.training_config.get("with_ut"):
            cmd.append("--with_ut")
        if self.training_config.get("with_eval3d"):
            cmd.append("--with_eval3d")
        
        # GeNVS novel view synthesis
        if self.training_config.get("use_zero123") or self.training_config.get("genvs"):
            cmd.append("--genvs")
            num_views = self.training_config.get("genvs_views", 20)
            cmd.extend(["--genvs-views", str(num_views)])
        
        # Pruning control
        if not self.training_config.get("prune", True):
            cmd.append("--no-prune")
        
        # Streaming mode
        if self.training_config.get("streaming"):
            cmd.append("--streaming")
        
        # === NEW: Pass quality mode, confidence filter, early stopping ===
        quality_mode = self.training_config.get("quality_mode", "balanced")
        cmd.extend(["--quality-mode", quality_mode])
        
        min_conf = self.training_config.get("min_confidence", 1000)
        cmd.extend(["--min-confidence", str(min_conf)])
        
        # Depth model selection: depth_anything (default), zoedepth, midas
        depth_model = self.training_config.get("depth_model", "depth_anything")
        cmd.extend(["--depth-model", depth_model])
        
        if self.training_config.get("early_stopping", True):
            cmd.append("--early-stopping")
            cmd.extend(["--early-stop-patience", str(self.training_config.get("loss_patience", 500))])
            cmd.extend(["--early-stop-min-delta", str(self.training_config.get("loss_threshold", 0.001))])
            cmd.extend(["--early-stop-min-steps", str(self.training_config.get("min_steps", 2000))])
        else:
            cmd.append("--no-early-stopping")

        # Smart Resume: Check if ACE-Zero output exists
        # ACE-Zero creates 'acezero_output' inside the output_dir
        expected_ace_output = self.output_dir / "acezero_output"
        # Check for valid COLMAP output (images.bin or images.txt)
        # This is the definitive sign that ACE-Zero finished successfully
        sparse_dir = expected_ace_output / "sparse" / "0"
        if (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists():
            logger.info("Found existing ACE-Zero COLMAP output. Auto-enabling --skip-ace to resume.")
            cmd.append("--skip-ace")
        
        self.logs.append(f"Command: {' '.join(cmd)}")
        print(f"Starting ACE-Zero pipeline: {' '.join(cmd)}")
        
        return self._start_subprocess(cmd, env)
    
    def _run_simple_training(self, video_path: str):
        """
        Level 1: Simple direct training (assumes data is pre-processed)
        """
        self.logs.append("=== LEVEL 1: SIMPLE TRAINING ===")
        
        # For simple mode, video_path is actually data_dir
        data_dir = Path(video_path)
        result_dir = self.output_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        trainer_script = self.root_dir / "examples" / "simple_trainer.py"
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        cmd = [
            sys.executable, "-u", str(trainer_script),
            "mcmc",
            "--data_dir", str(data_dir),
            "--result_dir", str(result_dir),
            "--max_steps", str(self.training_config["max_steps"]),
            "--data_factor", str(self.training_config["data_factor"]),
            "--eval_steps", *[str(s) for s in self.training_config["eval_steps"]],
        ]
        
        if self.training_config.get("save_ply"):
            cmd.append("--save_ply")
        if self.training_config.get("disable_viewer"):
            cmd.append("--disable_viewer")
        if self.training_config.get("with_ut"):
            cmd.append("--with_ut")
        if self.training_config.get("with_eval3d"):
            cmd.append("--with_eval3d")
        
        self.logs.append(f"Command: {' '.join(cmd)}")
        print(f"Starting simple training: {' '.join(cmd)}")
        
        return self._start_subprocess(cmd, env)
    
    def _run_chunked_pipeline(self, video_path: str):
        """
        Level 3: Full chunked pipeline with COLMAP
        """
        self.logs.append("=== LEVEL 3: CHUNKED PIPELINE ===")
        
        pipeline_script = self.scripts_dir / "automated_intelligent_pipeline.py"
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        cmd = [
            sys.executable, "-u", str(pipeline_script),
            video_path,
            "--output_dir", str(self.output_dir),
            "--use_colmap",  # Force COLMAP/chunked mode
            "--fps", str(self.training_config["fps"]),
            "--max_steps", str(self.training_config["max_steps"]),
            "--min_chunk_duration", "5",
        ]
        
        self.logs.append(f"Command: {' '.join(cmd)}")
        print(f"Starting chunked pipeline: {' '.join(cmd)}")
        
        return self._start_subprocess(cmd, env)
    
    def _start_subprocess(self, cmd: list, env: dict):
        """Start subprocess and monitor"""
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.root_dir,
                bufsize=1,
                env=env,
                text=True,
                encoding='utf-8'
            )
            
            # Start monitoring thread
            thread = threading.Thread(target=self._monitor_process)
            thread.daemon = True
            thread.start()
            
            return True
        except Exception as e:
            self.status = "error"
            self.logs.append(f"Failed to start process: {e}")
            raise e

    def stop_training(self):
        """Stop current training"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status = "idle"
            self.logs.append("Training stopped by user.")

    def get_status(self):
        """Get current pipeline status"""
        return {
            "status": self.status,
            "progress": self.progress,
            "mode": self.current_mode,
            "logs": self.logs[-50:],  # Return last 50 logs
            "training_viewer_url": self.training_viewer_url
        }

    def _monitor_process(self):
        """Monitor subprocess output"""
        try:
            for line in iter(self.process.stdout.readline, ''):
                decoded_line = line.strip()
                if decoded_line:
                    self.logs.append(decoded_line)
                    
                    # Write to file
                    try:
                        with open(self.log_file_path, "a", encoding='utf-8') as f:
                            f.write(decoded_line + "\n")
                    except:
                        pass

                    if len(self.logs) > 1000:
                        self.logs.pop(0)
                        
                    self._parse_log_line(decoded_line)
                    
        except Exception as e:
            self.logs.append(f"Error reading logs: {e}")
        finally:
            self.process.stdout.close()
            return_code = self.process.wait()
            
            if return_code == 0:
                self.status = "training_complete"
                self.progress = 100
                self.logs.append("Training completed successfully.")
            else:
                self.status = "error"
                self.logs.append(f"Process failed with return code {return_code}")

    def _parse_log_line(self, line: str):
        """Parse log line for progress and events"""
        # Capture Viewer URL
        if "TRAINING_VIEWER_URL:" in line:
            parts = line.split("TRAINING_VIEWER_URL:")
            if len(parts) > 1:
                self.training_viewer_url = parts[1].strip()
                self.logs.append(f"Training Viewer Detected: {self.training_viewer_url}")

        # ACE-Zero specific progress
        if "ACE-ZERO POSE ESTIMATION" in line:
            self.progress = 15
        if "Extracting frames" in line:
            self.progress = 20
        if "Extracted" in line and "frames" in line:
            self.progress = 30
        if "3DGS TRAINING" in line:
            self.progress = 40
            
        # Legacy progress indicators
        if "STEP 1:" in line: 
            self.progress = 10
        if "STEP 2:" in line: 
            self.progress = 40
        if "STEP 3:" in line: 
            self.progress = 70
        
        # Training step progress (from tqdm output)
        if "it/s]" in line and "/" in line:
            try:
                # Parse format like: 375/500 [00:38<00:12, 10.25it/s]
                parts = line.split("|")
                if len(parts) >= 2:
                    step_part = parts[-1].strip()
                    if "/" in step_part:
                        current, total = step_part.split("/")[0], step_part.split("/")[1].split()[0]
                        current, total = int(current.strip()), int(total.strip())
                        # Map 40-95% to training progress
                        train_progress = (current / total) * 55
                        self.progress = 40 + int(train_progress)
            except:
                pass

        # Chunk progress (for Level 3)
        if "[Progress]" in line and "chunks completed" in line:
            try:
                parts = line.split("]")[1].strip().split("chunks")[0].strip()
                current, total = map(int, parts.split("/"))
                chunk_progress = (current / total) * 40
                self.progress = 50 + int(chunk_progress)
            except:
                pass

        if "PIPELINE COMPLETE" in line or "PIPELINE SUMMARY" in line: 
            self.progress = 100
            self.status = "training_complete"
        
        # Parse loss for early stopping
        if "loss:" in line.lower() or "Loss:" in line:
            self._check_early_stopping(line)
    
    def _check_early_stopping(self, line: str):
        """Check if training should stop early based on loss convergence."""
        if not self.training_config.get("early_stopping", False):
            return
        
        try:
            # Parse loss value from line (formats: "loss: 0.0123" or "Loss=0.0123")
            import re
            match = re.search(r'[Ll]oss[=:]\s*([0-9.]+)', line)
            if not match:
                return
            
            loss = float(match.group(1))
            self.loss_history.append(loss)
            
            # Check for improvement
            if loss < self.best_loss - self.training_config["loss_threshold"]:
                self.best_loss = loss
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1
            
            # Check early stopping condition
            current_step = len(self.loss_history)
            min_steps = self.training_config.get("min_steps", 2000)
            patience = self.training_config.get("loss_patience", 500)
            
            if (current_step >= min_steps and 
                self.steps_without_improvement >= patience):
                self.logs.append(f"[EARLY STOP] Loss converged at {loss:.6f} after {current_step} steps")
                self.logs.append(f"[EARLY STOP] No improvement for {patience} steps, stopping...")
                # Signal to stop the process
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                    self.status = "training_complete"
                    self.progress = 100
        except Exception as e:
            pass  # Ignore parsing errors
    
    def get_video_fps(self, video_path: str) -> float:
        """Get actual FPS from video file."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else 30.0
        except:
            return 30.0  # Default fallback
    
    def validate_fps_config(self, video_path: str):
        """Validate and adjust fps config based on video."""
        video_fps = self.get_video_fps(video_path)
        config_fps = self.training_config.get("fps", 1.5)
        
        # Ensure extraction fps doesn't exceed video fps
        if config_fps > video_fps:
            self.logs.append(f"[FPS] Capping extraction fps from {config_fps} to {video_fps}")
            self.training_config["fps"] = video_fps
        
        # Log effective frame extraction rate
        estimated_frames = video_fps / self.training_config["fps"]
        self.logs.append(f"[FPS] Video: {video_fps:.2f} fps, Extracting: 1 frame per {1/self.training_config['fps']:.2f}s")
        return self.training_config["fps"]
