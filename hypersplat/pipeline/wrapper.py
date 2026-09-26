"""
Pipeline Wrapper for Demo Server
Level 2: ACE-Zero Direct + 3DGS MCMC Training

Pipeline Modes:
  - Level 1: Simple Training (COLMAP-based, requires pre-processed data)
  - Level 2: ACE-Zero Direct (AI Poses → 3DGS MCMC) [DEFAULT]
  - Level 3: Full Chunked Pipeline (Video → Chunks → COLMAP → 3DGS per chunk)
"""

import os
import re
import subprocess
import threading
import time
import sys
import collections
from pathlib import Path
from typing import Optional, Dict, Any, List

from hypersplat.pipeline import log_parsing


# =============================================================================
# Pipeline command building -- single source of truth for wrapper.py and
# services/api/server.py.
#
# The manager (hypersplat.pipeline.manager) treats any param that is NOT passed
# on the command line as "auto": a dynamic strategy derives it from measured
# signals (video, poses, GPU memory), with today's constants only as fallback.
# An explicitly passed flag is "user-set" and is never overridden. So entry
# points must only emit a flag when the caller actually provided a value;
# hard-coding and always passing a default silently disables the strategy.
# =============================================================================

# config key -> manager CLI flag, emitted only when the value is not None.
_VALUE_FLAGS = (
    ("fps", "--fps"),
    ("max_steps", "--max_steps"),
    ("data_factor", "--data_factor"),
    ("init_scale", "--init_scale"),
    ("cap_max", "--cap-max"),
    ("opacity_reg", "--opacity_reg"),
    ("scale_reg", "--scale_reg"),
    ("sh_degree", "--sh_degree"),
    ("means_lr", "--means_lr"),
    ("ssim_lambda", "--ssim_lambda"),
    ("quality_mode", "--quality-mode"),
    ("min_registration_confidence", "--min-registration-confidence"),
    ("depth_model", "--depth-model"),
    ("pose_opt_warmup", "--pose-opt-warmup"),
    ("refine_loops", "--refine-loops"),
    ("difix_views", "--difix-views"),
    ("difix_max_angle", "--difix-max-angle"),
    ("difix_pseudo_weight", "--difix-pseudo-weight"),
    ("colmap_input", "--colmap-input"),
    ("seed", "--seed"),
)

# config key -> store_true flag, emitted only when the value is truthy.
_BOOL_FLAGS = (
    ("with_ut", "--with_ut"),
    ("with_eval3d", "--with_eval3d"),
    ("random_bkgd", "--random_bkgd"),
    ("app_opt", "--app_opt"),
    ("difix", "--difix"),
)

# Early-stop values: (accepted config keys, flag). wrapper.py historically used
# loss_patience/loss_threshold/min_steps, server.py early_stop_*; both work.
_EARLY_STOP_FLAGS = (
    (("early_stop_patience", "loss_patience"), "--early-stop-patience"),
    (("early_stop_min_delta", "loss_threshold"), "--early-stop-min-delta"),
    (("early_stop_min_steps", "min_steps"), "--early-stop-min-steps"),
)

# Sanity bounds for caller-provided values (validated, never injected).
_POSITIVE_KEYS = ("fps", "max_steps", "data_factor", "genvs_views", "cap_max", "init_scale")


def _first_set(config: Dict[str, Any], keys) -> Any:
    for key in keys:
        if config.get(key) is not None:
            return config[key]
    return None


def validate_pipeline_config(config: Dict[str, Any]) -> None:
    """Raise ValueError for caller-provided values that are out of range.

    Unset (None) values are fine: the manager resolves them automatically.
    """
    for key in _POSITIVE_KEYS:
        value = config.get(key)
        if value is None:
            continue
        try:
            ok = float(value) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"{key} must be a positive number, got {value!r}")


def build_pipeline_args(config: Dict[str, Any]) -> List[str]:
    """Build the manager CLI flags for ``config``.

    Only values the caller set (not None) are emitted, so every unset param
    reaches the manager as "auto".
    """
    args: List[str] = []
    for key, flag in _VALUE_FLAGS:
        value = config.get(key)
        if value is not None:
            args.extend([flag, str(value)])

    for key, flag in _BOOL_FLAGS:
        if config.get(key):
            args.append(flag)

    # GeNVS novel view synthesis
    if config.get("use_zero123") or config.get("genvs"):
        args.append("--genvs")
        if config.get("genvs_views") is not None:
            args.extend(["--genvs-views", str(config["genvs_views"])])

    # Pruning control (manager default: prune off unless asked)
    if config.get("prune") is not None and not config["prune"]:
        args.append("--no-prune")

    if config.get("unified_stream") or config.get("streaming"):
        args.append("--streaming")

    pose_opt = config.get("pose_opt")
    if pose_opt is not None:
        args.append("--pose-opt" if pose_opt else "--no-pose-opt")

    early_stopping = config.get("early_stopping")
    if early_stopping is not None and not early_stopping:
        args.append("--no-early-stopping")
    else:
        if early_stopping:
            args.append("--early-stopping")
        for keys, flag in _EARLY_STOP_FLAGS:
            value = _first_set(config, keys)
            if value is not None:
                args.extend([flag, str(value)])

    return args


# tqdm progress: "loss=0.123| sh degree=3| :  12%|#   | 850/7000 [00:30<03:40, 27.9it/s]"
_TQDM_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
# Explicit step logs: "Step 850", "step: 850", "Train Step 850/7000", "Iteration 850"
_STEP_RE = re.compile(r"\b(?:[Ss]tep|[Ii]teration)[\s:=]+(\d+)")
_LOSS_RE = re.compile(r"[Ll]oss[=:]\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")


def parse_training_step(line: str) -> Optional[int]:
    """Extract the training step number from a tqdm/log line, or None."""
    match = _TQDM_STEP_RE.search(line) or _STEP_RE.search(line)
    return int(match.group(1)) if match else None


def parse_training_loss(line: str) -> Optional[float]:
    """Extract a loss value ("loss=0.0123" / "Loss: 0.0123") from a log line, or None."""
    match = _LOSS_RE.search(line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


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
        self.logs = collections.deque(maxlen=1000)
        self.status = "idle"  # idle, training_running, training_complete, error
        self.progress = 0
        self.training_viewer_url = None
        self.log_file_path = self.root_dir / "pipeline_log.txt"
        self.current_mode = PipelineMode.LEVEL_2_ACE_ZERO
        
        # Training config. Numeric params (fps, max_steps, data_factor,
        # eval/save steps, early-stop thresholds, regs, ...) are deliberately
        # absent: unset means "auto" and the manager derives them per scene.
        # Set them via config_override to pin a value.
        self.default_training_config = {
            "with_ut": False,
            "with_eval3d": False,
            "save_ply": True,
            "disable_viewer": False,
            # Early stopping: None = manager default; True/False = explicit
            "early_stopping": None,
            # Zero123 mode for GeNVS
            "use_zero123": False,
        }
        self.training_config = dict(self.default_training_config)

        # Loss tracking for the log-regex early-stopping fallback (simple mode
        # only, when the trainer has no native early stopping)
        self.loss_history = []
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.best_loss_step = 0
        self.regex_early_stop = False
        self.early_stopped = False
        
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
        self.logs.clear()
        self.progress = 0
        self.training_viewer_url = None
        
        # Apply config overrides on top of fresh defaults, so values from a
        # previous run (or a previous fps cap) don't leak in as "user-set"
        self.training_config = dict(self.default_training_config)
        if config_override:
            self.training_config.update(config_override)
        try:
            validate_pipeline_config(self.training_config)
        except ValueError:
            self.status = "idle"
            raise

        # Reset early stopping state
        self.loss_history = []
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.best_loss_step = 0
        self.regex_early_stop = False
        self.early_stopped = False
        
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
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        cmd = self.build_ace_zero_command(video_path)

        # Smart Resume: Check if ACE-Zero output exists
        # ACE-Zero creates 'acezero_output' inside the output_dir
        expected_ace_output = self.output_dir / "acezero_output"
        # Check for valid COLMAP output (images.bin or images.txt)
        # This is the definitive sign that ACE-Zero finished successfully
        sparse_dir = expected_ace_output / "sparse" / "0"
        if (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists():
            msg = "Found existing ACE-Zero COLMAP output. Auto-enabling --skip-ace to resume."
            print(msg)
            self.logs.append(msg)
            cmd.append("--skip-ace")
        
        self.logs.append(f"Command: {' '.join(cmd)}")
        print(f"Starting ACE-Zero pipeline: {' '.join(cmd)}")
        
        return self._start_subprocess(cmd, env)
    
    def build_ace_zero_command(self, video_path: str) -> List[str]:
        """Pure: the Level 2 manager command for the current training_config."""
        pipeline_script = self.scripts_dir / "automated_intelligent_pipeline.py"
        return [
            sys.executable, "-u", str(pipeline_script),
            video_path,
            "--output_dir", str(self.output_dir),
            *build_pipeline_args(self.training_config),
        ]

    def _trainer_has_native_early_stop(self, trainer_script: Path) -> bool:
        """Whether simple_trainer.py implements early stopping itself."""
        try:
            return "early_stop" in trainer_script.read_text(encoding="utf-8")
        except OSError:
            return False

    def build_simple_command(self, data_dir: str) -> List[str]:
        """Pure: the Level 1 simple_trainer command for the current training_config.

        Unset max_steps/data_factor/eval_steps/save_steps are left to the
        trainer's own defaults.
        """
        trainer_script = self.root_dir / "examples" / "simple_trainer.py"
        result_dir = self.output_dir / "results"
        config = self.training_config
        cmd = [
            sys.executable, "-u", str(trainer_script),
            "mcmc",
            "--data_dir", str(data_dir),
            "--result_dir", str(result_dir),
        ]
        for key in ("max_steps", "data_factor"):
            if config.get(key) is not None:
                cmd.extend([f"--{key}", str(config[key])])
        # The trainer's default eval/save steps (7k/30k) never fire for a
        # shorter custom run, so derive them from max_steps when unset.
        for key in ("eval_steps", "save_steps"):
            steps = config.get(key)
            if steps is None and config.get("max_steps") is not None:
                steps = [config["max_steps"]]
            if steps is not None:
                cmd.extend([f"--{key}", *[str(s) for s in steps]])

        if config.get("save_ply"):
            cmd.append("--save_ply")
        if config.get("disable_viewer"):
            cmd.append("--disable_viewer")
        if config.get("with_ut"):
            cmd.append("--with_ut")
        if config.get("with_eval3d"):
            cmd.append("--with_eval3d")
        return cmd

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
        
        cmd = self.build_simple_command(str(data_dir))

        # Log-regex early stopping is only a fallback for trainers without
        # native early stopping, and only when the user asked for it.
        self.regex_early_stop = (
            self.training_config.get("early_stopping") is True
            and not self._trainer_has_native_early_stop(trainer_script)
        )
        if self.regex_early_stop:
            self.logs.append("[EARLY STOP] Trainer has no native early stopping; using log-based fallback.")
        
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
            *build_pipeline_args(self.training_config),
        ]
        if self.training_config.get("min_chunk_duration") is not None:
            cmd.extend(["--min_chunk_duration", str(self.training_config["min_chunk_duration"])])
        
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
            "logs": list(self.logs)[-50:],  # Return last 50 logs
            "training_viewer_url": self.training_viewer_url
        }

    def _monitor_process(self):
        """Monitor subprocess output"""
        try:
            for line in iter(self.process.stdout.readline, ''):
                decoded_line = line.strip()
                if decoded_line:
                    # OPTIMIZATION: Collapse repetitive logs (tqdm, iterations)
                    is_repetitive = ("it/s]" in decoded_line and "%" in decoded_line) or \
                                    ("Iteration" in decoded_line and "Loss" in decoded_line) or \
                                    ("Train Step" in decoded_line)

                    if is_repetitive and self.logs and self._is_repetitive_match(self.logs[-1], decoded_line):
                        self.logs[-1] = decoded_line
                    else:
                        self.logs.append(decoded_line)
                    
                    # Write to file (always write all logs to file for debugging)
                    try:
                        with open(self.log_file_path, "a", encoding='utf-8') as f:
                            f.write(decoded_line + "\n")
                    except:
                        pass
                        
                    self._parse_log_line(decoded_line)
                    if self.regex_early_stop:
                        self._check_early_stopping(decoded_line)
                    
        except Exception as e:
            self.logs.append(f"Error reading logs: {e}")
        finally:
            self.process.stdout.close()
            return_code = self.process.wait()
            
            if return_code == 0 or self.early_stopped:
                self.status = "training_complete"
                self.progress = 100
                self.logs.append("Training completed successfully.")
            else:
                self.status = "error"
                self.logs.append(f"Process failed with return code {return_code}")

    def _is_repetitive_match(self, last_line: str, new_line: str) -> bool:
        """Check if new line is a progress update of the same type as last line"""
        return log_parsing.is_repetitive_match(last_line, new_line)

    def _parse_log_line(self, line: str):
        """Parse log line for progress and events"""
        result = log_parsing.parse_marker_progress(line)

        if result.viewer_url is not None:
            self.training_viewer_url = result.viewer_url
            self.logs.append(f"Training Viewer Detected: {self.training_viewer_url}")

        if result.progress is not None:
            self.progress = result.progress

        # Chunk progress (for Level 3) -- not part of the shared marker set,
        # since server.py's AppState.parse_progress has no equivalent.
        if "[Progress]" in line and "chunks completed" in line:
            try:
                parts = line.split("]")[1].strip().split("chunks")[0].strip()
                current, total = map(int, parts.split("/"))
                chunk_progress = (current / total) * 40
                self.progress = 50 + int(chunk_progress)
            except:
                pass

        if result.is_complete:
            self.status = "training_complete"

    # Fallback thresholds for the log-regex early stop, used only inside the
    # wrapper (never passed on the CLI) when the user didn't set them.
    _FALLBACK_EARLY_STOP = {"patience": 500, "min_delta": 0.001, "min_steps": 2000}

    def _check_early_stopping(self, line: str):
        """Log-regex early stopping fallback (see _run_simple_training).

        Steps are the trainer's actual step numbers parsed from the tqdm/log
        line, not a count of log lines.
        """
        if not self.regex_early_stop or self.early_stopped:
            return

        loss = parse_training_loss(line)
        step = parse_training_step(line)
        if loss is None or step is None:
            return
        self.loss_history.append((step, loss))

        config = self.training_config
        fallback = self._FALLBACK_EARLY_STOP
        patience = _first_set(config, ("early_stop_patience", "loss_patience"))
        min_delta = _first_set(config, ("early_stop_min_delta", "loss_threshold"))
        min_steps = _first_set(config, ("early_stop_min_steps", "min_steps"))
        patience = fallback["patience"] if patience is None else patience
        min_delta = fallback["min_delta"] if min_delta is None else min_delta
        if min_steps is None:
            min_steps = fallback["min_steps"]
            if config.get("max_steps") is not None:
                # Short runs: don't wait longer than a third of the budget
                min_steps = min(min_steps, int(config["max_steps"]) // 3)

        if loss < self.best_loss - min_delta:
            self.best_loss = loss
            self.best_loss_step = step
        self.steps_without_improvement = step - self.best_loss_step

        if step >= min_steps and self.steps_without_improvement >= patience:
            self.logs.append(f"[EARLY STOP] Loss converged at {loss:.6f} after {step} steps")
            self.logs.append(f"[EARLY STOP] No improvement for {self.steps_without_improvement} steps, stopping...")
            if self.process and self.process.poll() is None:
                self.early_stopped = True
                self.process.terminate()
                self.status = "training_complete"
                self.progress = 100

    def get_video_fps(self, video_path: str) -> Optional[float]:
        """Get actual FPS from video file, or None if it can't be read."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else None
        except Exception:
            return None

    def validate_fps_config(self, video_path: str) -> Optional[float]:
        """Sanity-check a user-set extraction fps against the video.

        Unset fps stays unset (the manager picks it per scene); a user-set fps
        above the video's own rate is capped to it.
        """
        config_fps = self.training_config.get("fps")
        video_fps = self.get_video_fps(video_path)
        if video_fps is None:
            return config_fps
        if config_fps is None:
            self.logs.append(f"[FPS] Video: {video_fps:.2f} fps, extraction fps: auto")
            return None

        # Ensure extraction fps doesn't exceed video fps
        if config_fps > video_fps:
            self.logs.append(f"[FPS] Capping extraction fps from {config_fps} to {video_fps}")
            self.training_config["fps"] = video_fps

        fps = self.training_config["fps"]
        self.logs.append(f"[FPS] Video: {video_fps:.2f} fps, Extracting: 1 frame per {1/fps:.2f}s")
        return fps
