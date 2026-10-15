import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Hugging Face models (Difix, ~5 GB) go to the data drive's shared cache, not ~/.cache on the
# root disk. Must precede any import that pulls in huggingface_hub.
_SHARED_HF_CACHE = Path(__file__).resolve().parents[2] / "hf_cache"
if _SHARED_HF_CACHE.is_dir():
    os.environ.setdefault("HF_HOME", str(_SHARED_HF_CACHE))
# Growable CUDA segments: online Difix and densification allocate/free large blocks; without
# this, fragmentation left >1 GB reserved-but-unusable on a 6 GB GPU. Before torch loads CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import itertools
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from hypersplat.stream.latent import LatentEncoder, latent_cosine_loss, latent_lambda, patch_error_weight
from hypersplat.stream.vjepa_latent import LatentPacer
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from trainer_schedule import (
    PlateauDetector,
    StreamHorizon,
    WorldConvergence,
    early_stop_defaults,
    floater_prune_mask,
    resolve_trainer_schedule,
)

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

# Streaming support
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.genvs_core.pipeline import GeNVSPipeline
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "pipeline"))
    from pose_watcher import StreamingDataManager
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Checkpoint to resume from
    resume: Optional[str] = None
    # [AI Acceleration] Path to AI-generated dense initialization PLY
    init_ply_path: Optional[str] = None
    
    # GeNVS Online Integration
    genvs_ckpt: Optional[str] = None
    genvs_interval: int = 200 # Frequency of adding views
    genvs_depth_only: bool = True  # If True, inject depth maps (fast, geometric). If False, inject RGB (slow, diffusion).
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"
    # [Autoregressive GeNVS] Render specific pose from file (skips training)
    render_pose_file: Optional[str] = None
    
    # Latent Injection / Master Tuner
    lambda_sds: float = 0.05  # Weight for SDS Guidance

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8081
    # Viewer walk mode (stream runs; needs scripts/viewer/build_walk_client.sh): keep the
    # viewer camera at capture height inside the region the training cameras covered
    viewer_walk: bool = True
    # Walkable radius around each camera / walk speed, in median camera spacings
    walk_radius_mult: float = 3.0
    walk_speed_mult: float = 4.0
    # Max look up/down from horizontal in walk mode (degrees)
    walk_pitch_deg: float = 35.0
    # Viewer-only splat cleanup ("Clean view"): hide splats that only show up as streaks and
    # floaters off the training views. Training is unaffected.
    viewer_min_opacity: float = 0.02
    viewer_max_aniso: float = 30.0
    viewer_max_scale_voxels: float = 8.0

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # Data Loader Workers
    data_loader_workers: int = 4
    # Random seed (torch / numpy / random; per rank)
    seed: int = 42
    # Reproducible runs: deterministic torch/cuDNN/cuBLAS kernels, the stream sampler draws
    # inline instead of on a prefetch thread, and a file source is fully localized before
    # step 0, so seeds, depth maps and new views arrive at the same steps in every run (live
    # sources cannot be pinned); ACE maps (bootstrap, refits) train with deterministic kernels
    # too. Remaining spread: float atomics in the rasterizer backward and the ACE sidecar's
    # RANSAC.
    deterministic: bool = False
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    
    # Masking / Object Extraction (Ported from Dynamic)
    use_mask: bool = False    # Use masks to ignore background
    mask_weight: float = 1.0  # Weight for masked pixels
    bg_reg: float = 0.0       # Background opacity regularization (clean artifacts)
    save_object_ply: bool = False # Save only the masked object
    run_pruner: bool = False  # Run DQN post-process pruner on the object PLY
    dqn_pruning: bool = True  # Enable online DQN-style floater elimination during training
    # Floater pruning starts after / runs every N steps (None: schedule-relative, 2000 / 400 @ 30k)
    floater_prune_start: Optional[int] = None
    floater_prune_every: Optional[int] = None
    # Max fraction of Gaussians killed per floater-pruning pass
    floater_max_frac: float = 0.02
    mask_dir: str = "masks"   # Directory name for masks

    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps (None: schedule-relative, 1000 @ 30k steps)
    sh_degree_interval: Optional[int] = None
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Training SSIM: "torchmetrics" (stateful metric module, generic depthwise convs, host syncs
    # in its input checks) or "fused" (fused_ssim CUDA kernel, valid padding = torchmetrics'
    # cropped border). Evaluation always uses torchmetrics
    ssim_impl: Literal["torchmetrics", "fused"] = "torchmetrics"

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # L-BFGS (quasi-Newton, torch.optim.LBFGS + strong-Wolfe line search) polish of the Gaussians
    # once densification has stopped (parameter count fixed). Each round runs on a FIXED batch of
    # views so the objective is deterministic, and is reverted if it does not lower that loss
    # (no loss spikes). Params are rescaled by their Adam lr so one curvature model fits all groups.
    lbfgs_refine: bool = False
    lbfgs_every: int = 50          # steps between rounds (Adam keeps running in between)
    lbfgs_views: int = 4           # views in the fixed batch
    lbfgs_max_iter: int = 8        # L-BFGS iterations per round
    lbfgs_history: int = 5         # history size (memory: 2 * history * N * params-per-splat floats)
    lbfgs_lr: float = 5.0          # first step moves the largest element by this many Adam-lr units

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # JOGS-style decoupled pose optimization (faster convergence)
    # When enabled, alternates between Gaussian-only and Pose phases
    pose_opt_jogs: bool = True
    # Warmup: delay pose optimization until Gaussians stabilize
    pose_warmup_steps: int = 1000
    # Interval: how often to switch to pose phase (steps)
    pose_opt_interval: int = 100
    # Duration: how many steps per pose phase
    pose_phase_steps: int = 20

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2
    # Warm-up: no depth loss before this step (None: schedule-relative, 100 @ 30k steps)
    depth_warmup_steps: Optional[int] = None
    # Ramp: linearly increase depth_lambda over this many steps (None: schedule-relative, 500 @ 30k)
    depth_ramp_steps: Optional[int] = None

    # Auxiliary latent loss: 1 - cos between frozen-ViT patch features of render and target
    latent_loss: bool = False
    # Weight for the latent loss
    latent_lambda: float = 0.1
    # Warm-up: no latent loss before this step
    latent_warmup_steps: int = 0
    # Linearly anneal latent_lambda to 0 at this step (0: hold constant)
    latent_anneal_end: int = 0
    # Feature backend: "depth_anything" (cached, depth-tuned) or "dinov2"
    latent_backend: Literal["depth_anything", "dinov2"] = "depth_anything"
    # ViT layer (-1: last; depth_anything accepts 3, 6, 9, 12)
    latent_layer: int = -1
    # Long side (px) of the images fed to the ViT
    latent_side: int = 224
    # Weight each patch by its render-vs-photo error (pooled to the ViT grid), so mismatched regions pull harder
    latent_patch_weight: bool = False
    # Added to the mean-1 error weight: floor for already-matched patches
    latent_weight_floor: float = 0.1
    # "vjepa" (default): V-JEPA 2 is the main model; the depth/DINO stream is supporting context hooked
    # into its predictor. "feature": plain cosine on frozen DINO / Depth Anything features (baseline).
    latent_model: Literal["feature", "vjepa"] = "vjepa"
    # V-JEPA 2 ViT-L checkpoint (vitl.pt: target_encoder + predictor), required for latent_model="vjepa"
    vjepa_checkpoint: Optional[str] = "/mnt/Shared/3D_WORLD_GENRATION/hf_cache/vjepa2/vitl.pt"
    # Long side (px) of the photo / render fed to V-JEPA; multiple of 16
    vjepa_side: int = 256
    # Fraction of V-JEPA patches predicted (the rest are the visible context)
    vjepa_target_frac: float = 0.5
    # Learning rate of the secondary-token projection inside the predictor hook
    vjepa_hook_lr: float = 1e-3
    # V-JEPA steps may add this fraction of the step time without it (whole step, backward included); it
    # runs every k-th step to stay within it, with the term scaled by the actual gap. 0 runs it every step.
    vjepa_pace_budget: float = 0.5
    # Feed the render's alpha (3DGUT coverage, pooled per patch) to the predictor with the secondary stream
    vjepa_alpha_condition: bool = True
    # Weight of the alpha hole term: raise alpha where V-JEPA's latent error is high and coverage low.
    # 0 disables it. Patches without stream depth (e.g. sky) are skipped when depth is available
    vjepa_alpha_lambda: float = 0.2
    # Patches with mean alpha below this count as holes for the alpha term
    vjepa_alpha_hole_max: float = 0.5

    # Early stopping on a plateau of the (EMA-smoothed) training loss
    early_stopping: bool = False
    # Stop after this many steps without relative improvement (None: ~7% of max_steps)
    early_stop_patience: Optional[int] = None
    # Minimum relative EMA-loss improvement that resets patience
    early_stop_min_delta: float = 1e-3
    # Never stop before this step (None: ~30% of max_steps)
    early_stop_min_steps: Optional[int] = None

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # CUDA timeline: torch.profiler over profile_cuda_steps steps from profile_cuda_start
    # (0 = off). Writes {result_dir}/profile/: Chrome trace, op table, sync/copy summary.
    profile_cuda_steps: int = 0
    profile_cuda_start: int = 500
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False
    # Unscented transform sigma points (with_ut; gsplat UnscentedTransformParameters). With
    # ut_require_all_valid a Gaussian is culled when any sigma point falls outside the image
    # plus ut_margin (fraction of its size): large / near Gaussians at the border pop in and
    # out. False keeps it if any point is valid, but then invalid (even behind-camera)
    # projections still enter its covariance
    ut_alpha: float = 0.1
    ut_beta: float = 2.0
    ut_kappa: float = 0.0
    ut_margin: float = 0.1
    ut_require_all_valid: bool = True

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    # Streaming mode: dynamically accept new poses during training
    streaming: bool = False
    # How often to check for new poses (in training steps)
    streaming_check_interval: int = 500

    # Streaming pipeline (hypersplat/stream): frames -> keyframes -> ACE sidecar -> this trainer,
    # all in memory. Replaces --data_dir. Video file (just-in-time), camera index / /dev/videoN,
    # rtsp/http URL (live) or flight://host:port (external camera pushing over Arrow Flight).
    stream_source: Optional[str] = None
    # auto: files as fast as training consumes them, live sources in real time. "realtime" on
    # a file rehearses the live path (frames are skipped when training is busy).
    stream_pace: Literal["auto", "realtime", "fast"] = "auto"
    # Train-resolution long side of streamed frames
    stream_max_side: int = 960
    # Memory budget for encoded keyframes + ACE copies + pseudo views, and keyframe cap
    stream_budget_mb: float = 384.0
    stream_max_keyframes: int = 1000
    # Frame rate asked of a local camera (phones over USB take it in the android:// spec)
    stream_capture_fps: Optional[float] = 60.0
    # Keyframes collected before the first ACE map is built
    stream_bootstrap_frames: int = 24
    # Cap on the first ACE map's LR cooldown (None: the ACE preset); shorter = map sooner,
    # background refits recover the accuracy
    stream_bootstrap_cooldown: Optional[int] = None
    # File sources: cache the first ACE map in this directory and reuse it on later runs of
    # the same file (StreamConfig.map_cache)
    stream_map_cache: Optional[str] = None
    # Keyframe gate novelty threshold (mean abs gray difference, 0-255)
    stream_min_motion: float = 8.0
    # Steps between pulls of new seed points (new keyframes/poses are picked up every step)
    stream_sync_every: int = 50
    # ACE refits: "legacy" train_ace defaults or "ace0" (ACE-Zero paper mapping iteration), and
    # Sim(3) gauge lock of every refit to the bootstrap frame (hypersplat/stream/gauge.py)
    stream_refit_recipe: Literal["legacy", "ace0"] = "legacy"
    stream_gauge_lock: bool = False
    # "continuous": after the first map the ACE head keeps training in step with 3DGS (one tick
    # per `stream_ace_tick_every` steps) and keyframes are relocalized round-robin, instead of
    # whole refits (hypersplat/stream/session.py, scripts/acezero/stream_mapper.py)
    stream_ace_mode: Literal["refit", "continuous"] = "refit"
    stream_ace_tick_every: int = 1
    # Cold start (implies continuous mode): no bootstrap wait. The first keyframe's Depth
    # Anything depth seeds the ACE map in seconds, training starts on it, and the mapper
    # registers the following keyframes as they arrive (StreamConfig.cold_start). Focal: ACE's
    # heuristic unless stream_cold_focal (px of the ACE-res frames)
    stream_cold_start: bool = False
    stream_cold_focal: Optional[float] = None
    # Depth Anything keeps guiding the ACE mapper loss after the seed (needs stream_depth and
    # continuous/cold-start mapping): aligned depth -> scene-coordinate targets under each frame's pose
    stream_mapper_depth: bool = False
    stream_mapper_depth_frac: float = 1.0
    stream_mapper_depth_weight: float = 50.0
    # Continuous mode: ACE pose / reloc rows through shared-memory rings instead of the pipe
    stream_ring: bool = False
    # Record the ACE events this run applied to {result_dir}/stream_events.arrows, or replay a
    # recording instead of running ACE (identical pose stream across A/B arms)
    stream_record: bool = False
    # Device pose table (hypersplat/stream/cuda): ACE pose rows applied by a CUDA kernel on the
    # training stream; real views render with the table's pose. stream_pose_alpha < 1 filters
    # relocalization jitter (SO(3) x R^3 EMA, corrections pass whole); stream_tiers weights
    # sampling by pose stability (provisional views x stream_tier_provisional_weight)
    stream_device_poses: bool = False
    stream_pose_alpha: float = 1.0
    stream_tiers: bool = False
    stream_tier_provisional_weight: float = 0.3
    stream_replay: Optional[str] = None
    # Per-source keyframe spacing (off by default; see hypersplat/stream/spacing.py)
    stream_adaptive_spacing: bool = False
    stream_spacing_min_inliers: int = 30
    # Debug: save every offered frame and its gate decision here (offers.jsonl + jpgs)
    stream_debug_offered_dir: Optional[str] = None
    # File sources: steps to keep training once every keyframe is localized (None: to the
    # dynamic end below)
    stream_tail_steps: Optional[int] = None
    # Stream runs have no fixed length (a live source's frame count is unbounded): max_steps
    # is only the initial budget, and every newly registered view moves the end to at least
    # step + stream_extend_steps (None: max_steps // 3), with LR decay and densification
    # following it. A live run ends that many steps after its last new view, or on STOP.
    stream_extend_steps: Optional[int] = None
    # Optional hard ceiling on the dynamic length (None: uncapped)
    stream_max_steps_cap: Optional[int] = None
    # Stop recording (camera released, training continues on the keyframes collected) after
    # this many seconds / keyframes; also on <result_dir>/STOP_RECORDING or the dashboard button
    stream_record_seconds: Optional[float] = None
    stream_record_keyframes: Optional[int] = None
    # Loss-aware view sampling: scale each real view's draw weight by (1 - mix) + mix *
    # (its photometric loss EMA / the median) ** power, clipped to [1/4, 4] (StreamTrainSet).
    # 0 disables it (novelty-only sampling)
    stream_hard_view_mix: float = 0.5
    stream_hard_view_power: float = 1.0
    # Stream early stop (with early_stopping): end training once the world is adequately
    # reconstructed instead of running to max_steps - coverage settled (no new views for
    # quiet_steps, ACE map idle), held-out PSNR plateaued (or >= target_psnr and gaining
    # little) and the Gaussian count stable, confirmed on consecutive probes. See
    # trainer_schedule.WorldConvergence.
    stream_converge_every: int = 500
    stream_converge_min_steps: int = 3000
    stream_converge_quiet_steps: int = 1500
    stream_converge_target_psnr: float = 22.0
    stream_converge_min_gain: float = 0.15
    # Probe views per held-out PSNR probe (full resolution, a few ms each): 6 views jittered
    # ~0.5 dB between probes on a plateau, more than min_gain
    stream_converge_probe_views: int = 12
    # The early stop also waits until the settle anneal below is this far along: stopping on
    # a plateau before the LR anneals keeps the noisy, unconverged model
    stream_converge_min_anneal: float = 0.75
    # Anneal on settle: once coverage settles (file source fully localized, or no new views
    # for settle_quiet_steps with the ACE map idle) the end is pulled in to step +
    # settle_tail_steps and the LR anneals from where it is (trainer_schedule.StreamHorizon.
    # settle). New views on a live source resume the moving schedule. None: schedule-relative.
    stream_settle: bool = True
    stream_settle_quiet_steps: Optional[int] = None
    stream_settle_tail_steps: Optional[int] = None
    stream_settle_min_steps: Optional[int] = None
    # Dense depth for streams (hypersplat/stream/depth.py): Depth Anything V2 on each
    # registered keyframe, aligned to its ACE points, used to seed Gaussians densely and as a
    # log-depth loss. Keyframes per sync step, seed pixel stride, loss weight.
    stream_depth: bool = True   # --no-stream-depth turns it off
    stream_depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    stream_depth_per_sync: int = 3
    stream_depth_seed_stride: int = 8
    # Dense seeding stops at this fraction of an MCMC cap (depth then only feeds the loss):
    # at the cap every seed would evict a live Gaussian, and MCMC fills the rest where the
    # images need detail
    stream_depth_seed_cap_frac: float = 0.8
    stream_depth_lambda: float = 0.05
    # Depth guard: Depth Anything supervises when the render drifts from it. The EMA of the
    # log-depth error is compared to its running best; while it exceeds best x (1 + tol) the
    # depth weight is boosted (x1.02 per step, up to guard_max), then decays back to 1.
    stream_depth_guard: bool = True   # --no-stream-depth-guard turns it off
    stream_depth_guard_tol: float = 0.25
    stream_depth_guard_max: float = 8.0
    # Closed depth loop (with_eval3d: the rasterizer renders ray-hit depth and its variance).
    # A rendered pixel is "confident" when its depth std is below stream_depth_conf_tau x its
    # depth. conf_weight: weight the depth loss per pixel by 1 / (1 + (std / (tau z))^2)
    # (floored at 0.1). render_align: keyframes per cycle whose monocular depth is re-fitted
    # to confident render pixels as well as ACE's points (round-robin; 0 = off)
    stream_depth_conf_tau: float = 0.05
    stream_depth_conf_weight: bool = False
    stream_depth_render_align: int = 0
    # Render feedback: every epoch, run Depth Anything on the 3DGS render of this many train
    # keyframes (round-robin) next to their camera frame, and down-weight the depth loss where
    # the two disagree (render_consistency). Runs with the ACE refit / 3DGUT training; 0 = off
    stream_depth_feedback: int = 0
    # Per-keyframe corrections for streams (hypersplat/stream/training.StreamViewAdjust):
    # pose deltas (pose_opt_lr / pose_opt_reg / pose_warmup_steps / JOGS phases apply) and an
    # exposure gain + bias per keyframe (camera auto-exposure). Slots follow keyframe ids, so
    # they survive the stream's evictions and reset when an ACE refit moves a keyframe.
    stream_pose_opt: bool = False
    stream_exposure_opt: bool = False
    stream_exposure_lr: float = 1e-3
    # Weight of the penalty on the mean exposure over keyframes (keeps renders neutral)
    stream_exposure_reg: float = 1e-2
    # Also serve live views/stats over Arrow Flight on this port (flight:// sources always do)
    stream_flight_port: Optional[int] = None
    # Live dashboard (viser viewer left, camera feed right) on this port; host 0.0.0.0 exposes
    # the camera feed on the network
    stream_dashboard_port: Optional[int] = None
    stream_dashboard_host: str = "127.0.0.1"

    # Online Difix: repair pseudo views from the live model during training (instead of the
    # post-training render -> Difix -> retrain loop). Works for datasets and streams.
    difix_online: bool = False
    # Steps of the rounds (None: 40/60/80% of max_steps for datasets; every
    # difix_online_every steps from difix_online_start for streams)
    difix_online_steps: Optional[List[int]] = None
    difix_online_start: int = 1500
    difix_online_every: int = 2000
    difix_model_id: str = "nvidia/difix_ref"
    # Streams: a round is spread over chunks (render + Difix for up to difix_chunk_s seconds,
    # one chunk every difix_chunk_every steps) so training and the viewer keep moving, with at
    # most difix_online_views pseudo views per round. With the world early stop, streams run
    # one early round (difix_online_start) and a final polish round once the world has
    # converged, then train difix_final_tail_steps on it before ending.
    difix_online_views: int = 16
    # Stream cycle: after every epoch (one pass over the current train keyframes, at least
    # stream_epoch_min_steps steps) run [cheap Difix: difix_epoch_views views at
    # difix_epoch_scale into a rolling pool of difix_online_views] -> [depth: re-align maps of
    # keyframes ACE re-localized] -> [ACE: background refit if anything changed]. Replaces the
    # step-scheduled Difix rounds and the final polish round.
    stream_cycle: bool = True
    stream_epoch_min_steps: int = 100
    difix_epoch_views: int = 2
    difix_epoch_scale: float = 0.5
    # Orbit angle cap for the cycle's small rounds (each faithful round widens the angle)
    difix_epoch_max_angle: float = 8.0
    # Cycle Difix starts once step >= this fraction of the (moving) training end: renders of
    # the early, blob-like model are what Difix hallucinates on, and a rolling pool keeps
    # those views for difix_online_views / difix_epoch_views epochs. 0 = from the first epoch
    difix_cycle_start_frac: float = 0.0
    # Streams: at most this share of training draws goes to pseudo views (None: uncapped -
    # fresh pseudo views win the sampler's novelty weight, ~40% of a 2k drone run's steps)
    difix_pseudo_share: Optional[float] = None
    # ACE step of the cycle: refit once this many keyframes changed (a refit takes minutes and
    # holds the GPU lease, so refitting after every new keyframe would starve Difix)
    stream_cycle_refit_min: int = 4
    difix_chunk_s: float = 3.0
    difix_chunk_every: int = 50
    difix_final_tail_steps: int = 750
    # Start Difix at scale 0.75 when less than this much VRAM is free (MB; OOM steps down anyway)
    difix_low_vram_mb: float = 3500.0
    # Reproducible Difix (hypersplat/stream/difix_online.OnlineDifix): per-view private seed
    # (None: torch's global RNG, shared with MCMC), VAE posterior mean instead of a sample, and
    # hallucination stop / orbit angle / pseudo weight judged on >= difix_pool_views pooled
    # views (0: on each round alone - one 2-view cycle round could stop Difix for the run)
    difix_seed: Optional[int] = 0
    difix_deterministic: bool = True
    difix_pool_views: int = 8

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        if self.sh_degree_interval is not None:
            self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

    def resolve_schedule(self, refine_stop_explicit: bool = False, growth_explicit: bool = False):
        """Fill schedule-relative step constants from max_steps (explicit values win)."""
        sched = resolve_trainer_schedule(self.max_steps)
        if self.sh_degree_interval is None:
            self.sh_degree_interval = sched.sh_degree_interval
        if self.depth_warmup_steps is None:
            self.depth_warmup_steps = sched.depth_warmup_steps
        if self.depth_ramp_steps is None:
            self.depth_ramp_steps = sched.depth_ramp_steps
        if self.floater_prune_start is None:
            self.floater_prune_start = sched.floater_start
        if self.floater_prune_every is None:
            self.floater_prune_every = sched.floater_every
        if self.stream_settle_quiet_steps is None:
            self.stream_settle_quiet_steps = sched.settle_quiet_steps
        if self.stream_settle_tail_steps is None:
            self.stream_settle_tail_steps = sched.settle_tail_steps
        if self.stream_settle_min_steps is None:
            self.stream_settle_min_steps = sched.settle_min_steps
        if isinstance(self.strategy, MCMCStrategy) and not growth_explicit:
            # Reach cap_max within the refine window whatever the schedule length (never slower
            # than the paper's 5% per refine, so 30k runs are unchanged)
            self.strategy.growth = "schedule"
        if isinstance(self.strategy, MCMCStrategy) and not refine_stop_explicit:
            # MCMC's 25k default outlives short pipeline runs (relocation until the last step)
            self.strategy.refine_stop_iter = sched.refine_stop_iter
        self.early_stop_patience, self.early_stop_min_steps = early_stop_defaults(
            self.max_steps, self.early_stop_patience, self.early_stop_min_steps
        )
        print(
            f"[Schedule] max_steps={self.max_steps} sh_degree_interval={self.sh_degree_interval} "
            f"depth_warmup/ramp={self.depth_warmup_steps}/{self.depth_ramp_steps} "
            f"floater_prune start/every={self.floater_prune_start}/{self.floater_prune_every} "
            f"refine_stop_iter={getattr(self.strategy, 'refine_stop_iter', None)}"
        )
        print(
            f"[Schedule] early_stopping={self.early_stopping} patience={self.early_stop_patience} "
            f"min_delta={self.early_stop_min_delta} min_steps={self.early_stop_min_steps}"
        )


def enable_determinism():
    """Deterministic torch kernels where they exist (warn, not fail, where they do not)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _flag_given(argv: List[str], *names: str) -> bool:
    """True if any of the CLI flags ``names`` appears in argv (``--x v`` or ``--x=v``)."""
    return any(a == n or a.startswith(n + "=") for a in argv for n in names)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    init_ply_path: Optional[str] = None,
    max_init_pts: Optional[int] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    elif init_type == "external":
        import open3d as o3d
        print(f"[AI Acceleration] Loading dense initialization from {init_ply_path}")
        pcd = o3d.io.read_point_cloud(init_ply_path)
        points = torch.tensor(np.asarray(pcd.points)).float()
        rgbs = torch.tensor(np.asarray(pcd.colors)).float()
        # Ensure we don't have too many points if memory is tight; with MCMC the
        # (smaller) strategy cap applies so the cloud never starts above cap_max.
        ext_cap = min(500_000, max_init_pts) if max_init_pts is not None else 500_000
        if points.shape[0] > ext_cap:
             # Random subsample
             indices = torch.randperm(points.shape[0])[:ext_cap]
             points = points[indices]
             rgbs = rgbs[indices]
    else:
        raise ValueError("Please specify a correct init_type: sfm, random, or external")

    # MCMC never removes Gaussians, so an initial cloud above cap_max would stay above it
    # (dense depth-unprojected clouds easily exceed 1M points and OOM small GPUs)
    if max_init_pts is not None and points.shape[0] > max_init_pts:
        print(f"Subsampling initial points {points.shape[0]} -> {max_init_pts} (strategy cap)")
        indices = torch.randperm(points.shape[0])[:max_init_pts]
        points = points[indices]
        rgbs = rgbs[indices]

    # Initialize the GS size to be the average dist of the nearest neighbors
    # Use min(4, num_points-1) to handle small datasets
    k_neighbors = min(4, points.shape[0] - 1)
    if k_neighbors < 1:
        k_neighbors = 1
    dist2_avg = (knn(points, k_neighbors)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    # Duplicate points (common in ACE-Zero clouds) give zero distance -> log(0) = -inf
    dist_avg = torch.sqrt(dist2_avg).clamp_min(1e-7)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


@dataclass
class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mask_dir_name="masks"):
        self.dataset = dataset
        self.mask_dir_name = mask_dir_name
        self.image_paths = dataset.parser.image_paths
        self.root_dir = Path(dataset.parser.data_dir)
        self.mask_dir = self.root_dir / mask_dir_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img_path = Path(self.image_paths[index])
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        
        mask = None
        if mask_path.exists():
             import cv2
             m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
             if m is not None:
                 H, W = data["image"].shape[0], data["image"].shape[1]
                 if m.shape != (H, W):
                     m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                 mask = torch.from_numpy(m).float() / 255.0
                 mask = mask.unsqueeze(0) # [1, H, W]
        
        if mask is None:
             mask = torch.ones((1, data["image"].shape[0], data["image"].shape[1]), dtype=torch.float32)
             
        data["mask"] = mask
        return data

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(cfg.seed + local_rank)
        if cfg.deterministic:
            enable_determinism()
        from gsplat.cuda._wrapper import UnscentedTransformParameters
        self.ut_params = UnscentedTransformParameters(
            alpha=cfg.ut_alpha, beta=cfg.ut_beta, kappa=cfg.ut_kappa,
            in_image_margin_factor=cfg.ut_margin, require_all_sigma_points_valid=cfg.ut_require_all_valid,
        )

        self.cfg = cfg
        if cfg.sh_degree_interval is None:  # not resolved by the CLI entry point
            cfg.resolve_schedule()
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb", flush_secs=2)

        self.stream = None
        if cfg.stream_source:
            self._start_stream()
        else:
            # Load data: Training data should contain initial points and colors.
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss or cfg.genvs_ckpt is not None,
            )
            if cfg.use_mask:
                print("[SimpleTrainer] Using masks for training.")
                self.trainset = MaskedDataset(self.trainset, cfg.mask_dir)

            self.valset = Dataset(self.parser, split="val")
            if cfg.use_mask:
                self.valset = MaskedDataset(self.valset, cfg.mask_dir)
        self.base_trainset = self.trainset
        self.difix = None
        if cfg.difix_online:
            from hypersplat.pipeline.params import SceneProfile
            from hypersplat.stream.difix_online import OnlineDifix
            self.difix = OnlineDifix(cfg.difix_model_id, self.device,
                                     profile=SceneProfile.from_env() or SceneProfile(),
                                     debug_dir=f"{cfg.result_dir}/difix_online",
                                     seed=cfg.difix_seed, deterministic=cfg.difix_deterministic,
                                     pool_views=cfg.difix_pool_views)
            self.difix.preload()   # pinned host copy now; the first round only copies to the GPU
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            init_ply_path=cfg.init_ply_path,
            max_init_pts=cfg.strategy.cap_max if isinstance(cfg.strategy, MCMCStrategy) else None,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        self.view_adjust = None
        if self.stream is not None and (cfg.stream_pose_opt or cfg.stream_exposure_opt):
            from hypersplat.stream.training import StreamViewAdjust
            capacity = cfg.stream_max_keyframes + 32
            pose_module = None
            if cfg.stream_pose_opt:
                pose_module = CameraOptModule(capacity)
                pose_module.zero_init()
            self.view_adjust = StreamViewAdjust(capacity, pose_module, exposure=cfg.stream_exposure_opt).to(self.device)
            if pose_module is not None:
                opt = torch.optim.Adam(pose_module.parameters(), lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                                       weight_decay=cfg.pose_opt_reg)
                self.view_adjust.pose_optimizers = [opt]
                self.pose_optimizers.append(opt)   # warmup / JOGS phases / checkpoint as usual
            if self.view_adjust.exposure is not None:
                opt = torch.optim.Adam(self.view_adjust.exposure.parameters(), lr=cfg.stream_exposure_lr)
                self.view_adjust.exposure_optimizers = [opt]
                self.stream_exposure_optimizers = [opt]
            print(f"[Stream] Per-keyframe corrections: pose={cfg.stream_pose_opt} "
                  f"exposure={cfg.stream_exposure_opt} ({capacity} slots)")

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        self._latent_enc = None  # LatentEncoder, built on first use (shares the depth model)
        self._vjepa = None  # VJEPALatentLoss, built on first use when latent_model == "vjepa"
        self._pending_latent_ckpt = None  # saved hook/optimizer state, applied when _vjepa is built
        self._pacer = LatentPacer(cfg.vjepa_pace_budget)  # paces the V-JEPA term against the step rate
        self.lpips = None
        try:
            if cfg.lpips_net == "alex":
                self.lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True
                ).to(self.device)
            elif cfg.lpips_net == "vgg":
                # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
                self.lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="vgg", normalize=False
                ).to(self.device)
            else:
                raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")
        except Exception as e:
            print(f"\n[Warning] Failed to initialize LPIPS (likely offline / network error): {e}")
            print("Using a dummy LPIPS metric to prevent pipeline crash.\n")
            class DummyLPIPS:
                def to(self, device):
                    return self
                def __call__(self, x, y):
                    return torch.tensor(0.0, device=x.device)
            self.lpips = DummyLPIPS()

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )
            if getattr(self, "dashboard", None) is not None:
                self.dashboard.viewer_port = cfg.port
            self.walk = None
            if self.stream is not None and cfg.viewer_walk:
                from hypersplat.stream.walk_bounds import WalkBoundsPublisher, WalkConfig
                self.walk = WalkBoundsPublisher(self.server, WalkConfig(
                    radius_mult=cfg.walk_radius_mult, speed_mult=cfg.walk_speed_mult,
                    pitch_deg=cfg.walk_pitch_deg))
                self._walk_update(force=True)

        # Streaming mode initialization
        self.streaming_manager = None
        if self.cfg.streaming and STREAMING_AVAILABLE:
            print("[Streaming] Initializing streaming data manager...")
            self.streaming_manager = StreamingDataManager(Path(cfg.data_dir))
            self.streaming_manager.start_watching()

        # GeNVS Initialization
        if self.cfg.genvs_ckpt:
            print(f"[GeNVS] Loading pipeline from {self.cfg.genvs_ckpt}...")
            self.genvs_pipeline = GeNVSPipeline(device=self.device, model_channels=64)
            ckpt = torch.load(self.cfg.genvs_ckpt, map_location=self.device)
            self.genvs_pipeline.encoder.load_state_dict(ckpt['encoder'])
            self.genvs_pipeline.renderer.load_state_dict(ckpt['renderer'])
            self.genvs_pipeline.unet.load_state_dict(ckpt['unet'])
            self.genvs_pipeline.volume_struct.load_state_dict(ckpt['volume'])
            print("[GeNVS] Pipeline Loaded. Online Generation Enabled.")
            
            # Export Config
            self.genvs_pipeline.save_config(f"{self.cfg.result_dir}/cfg_genvs.yml")
            
            # Setup output dir for online images
            self.genvs_out_dir = Path(self.cfg.data_dir) / "genvs_online"
            self.genvs_out_dir.mkdir(parents=True, exist_ok=True)
            self.genvs_count = 0
            
            # [Adaptive Interpolation] Loss Tracking
            self.loss_ema = 1.0 # Initialize high
            self.loss_history = []
            self.adaptive_mode = "deterministic"
            print("[GeNVS] Adaptive Interpolation Active (Mode: Deterministic)")

    # ------------------------------------------------------------------ streaming
    def _start_stream(self):
        """Stream source instead of a COLMAP dataset (hypersplat/stream)."""
        import logging
        from hypersplat.pipeline.params import SceneProfile
        from hypersplat.stream.session import StreamConfig, StreamSession
        from hypersplat.stream.training import (StopFile, StreamParser, StreamTrainSet,
                                                StreamValSet, seed_voxel_size)

        cfg = self.cfg
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        # Per-image modules are sized once at start; a stream's image set keeps changing
        per_image = [k for k in ("pose_opt", "app_opt", "use_bilateral_grid", "use_mask", "depth_loss")
                     if getattr(cfg, k)]
        if per_image:
            print(f"[Stream] Disabled for streams (per-image state of a changing image set): {per_image}")
            for k in per_image:
                setattr(cfg, k, False)
        scfg = StreamConfig(
            source=cfg.stream_source, pace=cfg.stream_pace, max_side=cfg.stream_max_side,
            budget_mb=cfg.stream_budget_mb, max_keyframes=cfg.stream_max_keyframes,
            capture_fps=cfg.stream_capture_fps,
            test_every=cfg.test_every, min_motion=cfg.stream_min_motion,
            bootstrap_frames=cfg.stream_bootstrap_frames, flight_port=cfg.stream_flight_port,
            bootstrap_cooldown=cfg.stream_bootstrap_cooldown,
            map_cache=cfg.stream_map_cache,
            record_stop_file=os.path.join(cfg.result_dir, "STOP_RECORDING"),
            record_seconds=cfg.stream_record_seconds, record_keyframes=cfg.stream_record_keyframes,
            depth=cfg.stream_depth, ace_deterministic=cfg.deterministic,
            refit_recipe=cfg.stream_refit_recipe, gauge_lock=cfg.stream_gauge_lock,
            ace_mode="continuous" if cfg.stream_cold_start else cfg.stream_ace_mode, ring=cfg.stream_ring,
            cold_start=cfg.stream_cold_start, cold_focal=cfg.stream_cold_focal,
            adaptive_spacing=cfg.stream_adaptive_spacing,
            spacing_min_inliers=cfg.stream_spacing_min_inliers,
            debug_offered_dir=cfg.stream_debug_offered_dir,
            mapper_depth=cfg.stream_mapper_depth, mapper_depth_frac=cfg.stream_mapper_depth_frac,
            mapper_depth_weight=cfg.stream_mapper_depth_weight,
            record=f"{cfg.result_dir}/stream_events.arrows" if cfg.stream_record else None,
            replay=cfg.stream_replay,
        )
        self.depth_estimator = None
        if cfg.stream_depth:
            from hypersplat.stream.depth import DepthEstimator
            self.depth_estimator = DepthEstimator(cfg.stream_depth_model, device=str(self.device)).load()
        self.stream = StreamSession(scfg, profile=SceneProfile.from_env())
        self.stream.depth_estimator = self.depth_estimator   # cold start seeds with it
        self.stream_stop_file = StopFile(cfg.result_dir)
        self.stream.start()
        self.dashboard = None
        if cfg.stream_dashboard_port:
            from hypersplat.stream.dashboard import Dashboard
            self.dashboard = Dashboard(self.stream, cfg.stream_dashboard_port, cfg.stream_dashboard_host,
                                  tb_dir=f"{cfg.result_dir}/tb").start()
            print(f"[Stream] Dashboard (3D viewer | camera): {self.dashboard.url}", flush=True)
        print("[Stream] Waiting for the first ACE map...")
        preview = None
        if self.depth_estimator is not None and self.dashboard is not None:
            from hypersplat.stream.depth import DepthPreview
            dash = self.dashboard
            preview = DepthPreview(self.depth_estimator, self.stream.latest_frame,
                                   lambda jpg: setattr(dash, "depth_preview", jpg)).start()
        try:
            map_ok = self.stream.wait_for_map(should_stop=self.stream_stop_file.requested)
        finally:
            if preview is not None:
                preview.stop()   # the estimator belongs to the depth worker from here on
                self.dashboard.depth_preview = None
                print(f"[Depth] Preview showed {preview.frames} frames while the map was built")
        if not map_ok:
            reason = self.stream.failed or ("STOP requested" if self.stream_stop_file.requested() else "stopped")
            self.stream.stop()
            if self.dashboard is not None:
                self.dashboard.stop()
            raise RuntimeError(f"[Stream] No ACE map: {reason}")
        if cfg.deterministic:
            self._wait_stream_settled()
        self.parser = StreamParser(self.stream, cfg.result_dir)
        self.trainset = StreamTrainSet(self.stream, seed=cfg.seed, hard_mix=cfg.stream_hard_view_mix,
                                       hard_power=cfg.stream_hard_view_power,
                                       pseudo_share=cfg.difix_pseudo_share,
                                       prefetch=0 if cfg.deterministic else 2)
        self.valset = StreamValSet(self.stream)
        self.device_poses = None
        if cfg.stream_device_poses or cfg.stream_tiers or cfg.stream_pose_alpha < 1.0:
            from hypersplat.stream.training import DevicePoses
            self.device_poses = DevicePoses(self.stream, self.trainset, alpha=cfg.stream_pose_alpha,
                                            tiers=cfg.stream_tiers,
                                            provisional_weight=cfg.stream_tier_provisional_weight)
        self.stream_voxel = seed_voxel_size(self.parser.points, self.parser.scene_scale)
        self.stream_horizon = StreamHorizon(cfg.max_steps, getattr(cfg.strategy, "refine_stop_iter", cfg.max_steps),
                                            cfg.stream_extend_steps, cfg.stream_max_steps_cap)
        self._views_seen = self.stream.views_registered
        self._last_view_step = 0
        self._loss_ema = None
        self.world_conv = None
        self._final_difix = None   # None | "running" | "done" (world early stop polish round)
        self._epoch, self._epoch_draws, self._epoch_start = 0, 0, 0
        self._pseudo_pool = []
        self._stop_at = None
        if cfg.early_stopping:
            self.world_conv = WorldConvergence(
                min_steps=cfg.stream_converge_min_steps, quiet_steps=cfg.stream_converge_quiet_steps,
                target_psnr=cfg.stream_converge_target_psnr, min_gain=cfg.stream_converge_min_gain,
                growth_tol=None if isinstance(cfg.strategy, MCMCStrategy) else 0.02)
            print(f"[WorldStop] On: stop when the world is reconstructed (probe every "
                  f"{cfg.stream_converge_every} steps, target {cfg.stream_converge_target_psnr} dB, "
                  f"min {cfg.stream_converge_min_steps} steps); max_steps is only a ceiling")
        self._horizon_logged = cfg.max_steps
        self._stream_failed_logged = False
        print(f"[Stream] Map ready: {len(self.parser.points)} seed points, scene scale "
              f"{self.parser.scene_scale:.3f}, seed voxel {self.stream_voxel:.4f}")

    def _stream_sync(self, step: int) -> bool:
        """Grow the model into newly seen space; True when the stream run should end."""
        from hypersplat.stream.training import inject_gaussians

        cfg = self.cfg
        s = self.stream
        if self.view_adjust is not None:
            self.view_adjust.slots.release_missing({kf.kid for kf in s.store.frames()})
        xyz, rgb = s.take_seeds()
        if self.depth_estimator is not None:
            dxyz, drgb = self._stream_depth(step)
            cap = cfg.strategy.cap_max if isinstance(cfg.strategy, MCMCStrategy) else None
            if cap is not None and len(self.splats["means"]) >= cfg.stream_depth_seed_cap_frac * cap:
                dxyz = dxyz[:0]
            if len(dxyz):
                xyz, rgb = np.concatenate([xyz, dxyz]), np.concatenate([rgb, drgb])
        if len(xyz):
            cap = cfg.strategy.cap_max if isinstance(cfg.strategy, MCMCStrategy) else None
            added, recycled = inject_gaussians(self.splats, self.optimizers, self.strategy_state, xyz, rgb,
                                               self.stream_voxel, cfg.init_opa, cfg.init_scale, cap)
            if added or recycled:
                print(f"\n[Stream] step {step}: +{added} Gaussians ({recycled} recycled) from new "
                      f"keyframes; {len(self.splats['means'])} total")
        # New views move the end of training (a live source has no frame count to size it by)
        h = self.stream_horizon
        seen = s.views_registered
        if seen > self._views_seen:
            self._last_view_step = step
            if self.world_conv is not None:
                self.world_conv.on_data_change(step)
        was_settled = h.settled
        if seen > self._views_seen and h.on_new_views(step):
            if was_settled:
                print(f"\n[Stream] step {step}: new views after settling; training resumes to {h.value} steps")
                self._horizon_logged = h.value
            elif h.value - self._horizon_logged >= h.extend:
                print(f"\n[Stream] step {step}: {seen} views registered so far; training extended to "
                      f"{h.value} steps (densify until {h.refine_stop_iter})")
                self._horizon_logged = h.value
        self._views_seen = seen
        self._walk_update()
        if self._stop_at is not None and step >= self._stop_at:
            print(f"\n[WorldStop] Ending at step {step} after the final Difix round")
            return True
        if (self.world_conv is not None and self._final_difix is None and step > 0
                and step % cfg.stream_converge_every == 0):
            if self._world_probe(step):
                return True
        if self.dashboard is not None:
            self.dashboard.update_training(step=step, max_steps=h.value, num_gs=len(self.splats["means"]),
                                           loss=getattr(self, "_last_loss", None),
                                           difix_rounds=self.difix.rounds if self.difix else None)
        if self.world_rank == 0 and cfg.tb_every > 0 and step % max(cfg.tb_every, cfg.stream_sync_every) == 0:
            st = s.stats()
            for key in ("keyframes", "registered", "unregistered", "pseudo", "store_mb", "map_version",
                        "frames_dropped"):
                self.writer.add_scalar(f"stream/{key}", st[key] or 0, step)
        if s.failed and not self._stream_failed_logged:
            print(f"\n[Stream] {s.failed}; training continues on the views collected so far")
            self._stream_failed_logged = True
        if self.stream_stop_file.requested():
            print(f"\n[Stream] STOP requested at step {step}")
            return True
        if s.finished and not h.finished and cfg.stream_tail_steps is not None:
            h.finish(step + cfg.stream_tail_steps)
            print(f"\n[Stream] Source fully localized at step {step}; {cfg.stream_tail_steps} more steps")
        if cfg.stream_settle and not h.finished and not h.settled and step >= cfg.stream_settle_min_steps:
            self._maybe_settle(step)
        if hasattr(cfg.strategy, "refine_stop_iter"):
            cfg.strategy.refine_stop_iter = h.refine_stop_iter
        return False

    def _wait_stream_settled(self, timeout: float = 600.0):
        """Deterministic runs: let a file source finish localizing before step 0, so every
        keyframe, seed and depth job exists from the start instead of arriving at a step
        that depends on wall-clock timing. Live sources never finish: no wait."""
        s = self.stream
        if s.kind != "file":
            print(f"[Stream] deterministic: {s.kind} source cannot be pinned; arrival steps vary")
            return
        t0, end = time.time(), time.time() + timeout
        while not s.finished and not s.failed and time.time() < end:
            if self.stream_stop_file.requested():
                return
            time.sleep(0.2)
        print(f"[Stream] deterministic: waited {time.time() - t0:.1f} s for the source to be fully "
              f"localized ({s.views_registered} views{'' if s.finished else ', timed out'})")

    def _maybe_settle(self, step: int):
        """Pull the end in once coverage has settled, so the LR anneals now (see
        StreamHorizon.settle) instead of on a plateau toward a far-away end."""
        cfg = self.cfg
        s = self.stream
        st = s.stats()
        quiet = step - self._last_view_step
        map_idle = s.map_ready.is_set() and st["job"] is None and st["pending"] == 0
        if not (map_idle and (s.finished or quiet >= cfg.stream_settle_quiet_steps)):
            return
        # The model must be done growing too: settling as soon as a file is localized (step
        # 200 of 2k, before depth seeds and MCMC growth) cut the fit phase short (-0.55 dB)
        if cfg.stream_depth and s.depth_backlog:
            return
        if isinstance(cfg.strategy, MCMCStrategy):
            if len(self.splats["means"]) < 0.98 * cfg.strategy.cap_max and step < self.stream_horizon.refine_stop_iter:
                return
        elif step < self.stream_horizon.refine_stop_iter // 2:
            return
        h = self.stream_horizon
        old_end = h.value
        if h.settle(step, cfg.stream_settle_tail_steps):
            why = "source fully localized" if s.finished else f"no new views for {quiet} steps"
            print(f"\n[Stream] step {step}: coverage settled ({why}); annealing to step {h.value} "
                  f"instead of {old_end} (densify until {h.refine_stop_iter})")
            self._horizon_logged = h.value
            if self.world_rank == 0 and cfg.tb_every > 0:
                self.writer.add_scalar("stream/settle_end", h.value, step)

    def _stream_depth(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Dense depth for up to stream_depth_per_sync newly registered keyframes; returns
        their seed points (normalized xyz, rgb) from the valid depth pixels."""
        from hypersplat.stream.depth import dense_depth, unproject
        from hypersplat.stream.source import decode_jpeg

        cfg = self.cfg
        s = self.stream
        pts, cols = [], []
        for kf, pix, z in s.take_depth_jobs(cfg.stream_depth_per_sync):
            rgb = decode_jpeg(kf.jpeg.to_pybytes())
            t0 = time.time()
            disp = self.depth_estimator.predict(rgb)
            res = dense_depth(disp, rgb, pix, z)
            ms = (time.time() - t0) * 1000
            if res.depth is None:
                s.set_depth(kf.kid, None, {"ms": ms, "reason": res.align.reason}, disp=res.disp,
                            version=kf.map_version)
                continue
            s.set_depth(kf.kid, res.depth, {"ms": ms, "rel_err": res.align.rel_err}, disp=res.disp,
                        version=kf.map_version)
            h, w = res.depth.shape
            world, rgbs = unproject(res.depth, res.valid, s.intrinsics(w, h), np.linalg.inv(kf.w2c), rgb,
                                    cfg.stream_depth_seed_stride)
            pts.append(s.transform_points(world.astype(np.float64)).astype(np.float32))
            cols.append(rgbs)
        c = s.depth_counters
        done = c["maps"] + c["rejected"]
        if done and done // 10 > getattr(self, "_depth_logged", 0):
            self._depth_logged = done // 10
            err = f"{c['rel_err']:.1%}" if c["rel_err"] is not None else "-"
            print(f"\n[Depth] step {step}: {c['maps']} maps, {c['rejected']} rejected"
                  f"{' (' + c['last_reject'] + ')' if c['last_reject'] else ''}, held-out error vs ACE {err}, "
                  f"{c['ms']:.0f} ms/keyframe (network {self.depth_estimator.ms:.0f} ms), backlog {s.depth_backlog}")
        if not pts:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        return np.concatenate(pts), np.concatenate(cols)

    def _world_probe(self, step: int) -> bool:
        """Feed WorldConvergence one probe; True when training should end now."""
        cfg = self.cfg
        s = self.stream
        st = s.stats()
        wait = None
        if not s.map_ready.is_set():
            wait = "first map"
        elif cfg.stream_settle and not self.stream_horizon.finished:
            done = self.stream_horizon.anneal_progress(step)
            if done < cfg.stream_converge_min_anneal:
                wait = f"LR anneal {done:.0%}" if self.stream_horizon.settled else "coverage to settle"
        conv = self.world_conv
        done = conv.update(step, psnr=self._probe_psnr(step), loss_ema=self._loss_ema,
                           num_gs=len(self.splats["means"]), views_quiet_steps=step - self._last_view_step,
                           map_idle=st["job"] is None and st["pending"] == 0, extra_wait=wait)
        if self.dashboard is not None:
            self.dashboard.update_training(world=conv.status["reason"])
        if self.world_rank == 0 and cfg.tb_every > 0:
            if conv.status.get("psnr") is not None:
                self.writer.add_scalar("stream/probe_psnr", conv.status["psnr"], step)
            self.writer.add_scalar("stream/converge_streak", conv.streak, step)
        if not done:
            return False
        print(f"\n[WorldStop] World reconstructed at step {step} (of up to {self.stream_horizon.value}): "
              f"{conv.status['reason']}")
        if s.stop_recording("world reconstructed"):
            print("[WorldStop] Recording stopped; camera released")
        if (self.difix is not None and not self.difix.stop_reason and self._final_difix is None
                and not cfg.stream_cycle):
            # Polish once on the converged model, then a short tail on its pseudo views
            if self.difix.round_pending:
                return False   # the earlier round is still being fixed: next probe
            self._final_difix = "running"
            if self._difix_round(step) is None:   # ACE holds the GPU: retry on the next probe
                self._final_difix = None
                return False
            print("[WorldStop] Final Difix round started (runs in chunks while training continues)")
            return False
        return True

    @torch.no_grad()
    def _probe_psnr(self, step: int) -> Optional[float]:
        """Mean PSNR on up to stream_converge_probe_views held-out keyframes at full
        resolution (a few ms per view); None without any. Half resolution hides the fine
        detail that keeps improving late in training, so the plateau looked earlier than it was."""
        cfg = self.cfg
        self.valset.refresh()
        n = len(self.valset)
        if n == 0:
            return None
        k = min(n, max(1, cfg.stream_converge_probe_views))
        vals = []
        for i in np.linspace(0, n - 1, k).round().astype(int):
            data = self.valset[int(i)]
            pixels = torch.as_tensor(data["image"], device=self.device).float()[None].permute(0, 3, 1, 2) / 255.0
            K = torch.as_tensor(data["K"], device=self.device).float()[None]
            h, w = pixels.shape[2:]
            colors, _, _ = self.rasterize_splats(
                camtoworlds=torch.as_tensor(data["camtoworld"], device=self.device).float()[None],
                Ks=K, width=w, height=h, sh_degree=min(step // cfg.sh_degree_interval, cfg.sh_degree),
                near_plane=cfg.near_plane, far_plane=cfg.far_plane)
            colors = colors.clamp(0.0, 1.0).permute(0, 3, 1, 2)
            mse = F.mse_loss(colors, pixels).clamp_min(1e-10)
            vals.append(float(-10.0 * torch.log10(mse)))
        return float(np.mean(vals))

    # ------------------------------------------------------------------ online Difix
    def _difix_schedule(self, max_steps: int) -> set:
        cfg = self.cfg
        if self.difix is None:
            return set()
        if cfg.difix_online_steps:
            return set(cfg.difix_online_steps)
        if self.stream is not None:
            if cfg.stream_cycle:
                return set()  # Difix runs in the per-epoch cycle
            if getattr(self, "world_conv", None) is not None:
                return {cfg.difix_online_start}  # + the final polish round on convergence
            return set(range(cfg.difix_online_start, max_steps - cfg.difix_online_every // 2,
                             cfg.difix_online_every))
        return {int(max_steps * f) for f in (0.4, 0.6, 0.8)}

    @torch.no_grad()
    def _render_uint8(self, c2w: np.ndarray, K: np.ndarray, width: int, height: int, step: int) -> np.ndarray:
        cfg = self.cfg
        renders, _, _ = self.rasterize_splats(
            camtoworlds=torch.as_tensor(c2w, dtype=torch.float32, device=self.device)[None],
            Ks=torch.as_tensor(K, dtype=torch.float32, device=self.device)[None],
            width=width, height=height,
            sh_degree=min(step // cfg.sh_degree_interval, cfg.sh_degree),
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        )
        return (renders[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def _scene_points(self, n: int = 20000) -> np.ndarray:
        """Opaque Gaussian centres: the live geometry pseudo cameras orbit around."""
        with torch.no_grad():
            means = self.splats["means"][torch.sigmoid(self.splats["opacities"]) > 0.3]
            if len(means) > n:
                # Private generator: drawing from the global RNG here would shift the trainer's
                # MCMC / sampling stream whenever (and however often) a Difix round runs
                g = torch.Generator(device=means.device).manual_seed(len(means))
                means = means[torch.randperm(len(means), device=means.device, generator=g)[:n]]
            return means.cpu().numpy().astype(np.float64)

    def _difix_sources(self):
        from hypersplat.stream.difix_online import DifixSources
        from hypersplat.stream.schema import select_rows
        from hypersplat.stream.source import decode_jpeg

        if self.stream is not None:
            _, table = self.stream.views()
            table = select_rows(table, kind="real", split="train")
            if table.num_rows < 2:
                return None
            c2ws = np.array(table.column("c2w").to_pylist()).reshape(-1, 4, 4)
            return DifixSources(
                c2ws=c2ws, K=np.array(table.column("K")[0].as_py()).reshape(3, 3),
                width=int(table.column("width")[0].as_py()), height=int(table.column("height")[0].as_py()),
                weights=np.array(table.column("weight").to_pylist(), dtype=np.float64),
                points=self._scene_points(),
                get_image=lambda i: decode_jpeg(table.column("jpeg")[i].as_py()),
                ids=table.column("kid").to_pylist(),
            )
        from hypersplat.beta.difix.augment import confidence_weights, load_ace_confidence

        ds = getattr(self.base_trainset, "dataset", self.base_trainset)
        idx = np.asarray(ds.indices)
        names = [Path(self.parser.image_names[i]).name for i in idx]
        conf_w = confidence_weights(load_ace_confidence(Path(self.parser.data_dir)))
        cam = self.parser.camera_ids[idx[0]]
        width, height = self.parser.imsize_dict[cam]
        return DifixSources(
            c2ws=self.parser.camtoworlds[idx], K=self.parser.Ks_dict[cam].copy(), width=width, height=height,
            weights=np.array([conf_w.get(n, 1.0) for n in names], dtype=np.float64),
            points=self._scene_points(),
            get_image=lambda i: ds[i]["image"].numpy().astype(np.uint8),
            ids=list(range(len(idx))),
        )

    def _difix_round(self, step: int) -> Optional[bool]:
        """Run one online Difix round; True if the batch DataLoader must be rebuilt, None if
        deferred (a stream's ACE refit holds the GPU)."""
        from hypersplat.stream.training import WithPseudo

        if self.difix.stop_reason:
            return False
        src = self._difix_sources()
        if src is None or len(src.c2ws) < 2:
            return False
        if self.stream is not None:
            # Difix and an ACE refit together OOM a small GPU: take turns (per chunk)
            if not self.stream.gpu_lease.try_acquire("difix"):
                return None
            try:
                if self.difix.scale_idx == 0 and self.difix.rounds == 0:
                    from hypersplat.stream.difix_online import SCALES
                    free_mb = torch.cuda.mem_get_info()[0] / 2 ** 20
                    if free_mb < self.cfg.difix_low_vram_mb:
                        self.difix.scale_idx = SCALES.index(0.75)
                        print(f"[Difix-online] {free_mb:.0f} MB VRAM free; starting at scale 0.75")
                self.difix.begin_round(src, step, max_views=self.cfg.difix_online_views)
            finally:
                self.stream.release_gpu("difix")
            self._difix_chunk(step)
            return False
        torch.cuda.empty_cache()
        views, sources, _ = self.difix.run_round(
            lambda c2w, K, w, h: self._render_uint8(c2w, K, w, h, step), src, step)
        self.trainset = WithPseudo(self.base_trainset, views, [src.ids[s] for s in sources])
        return True

    def _stream_cycle(self, step: int):
        """End of an epoch (one pass over the train keyframes): Difix -> depth -> ACE."""
        cfg = self.cfg
        s = self.stream
        n_train = len(s.store.registered("train"))
        if self._epoch_draws < max(n_train, 1) or step - self._epoch_start < cfg.stream_epoch_min_steps:
            return
        self._epoch += 1
        self._epoch_draws, self._epoch_start = 0, step
        t0 = time.time()
        difix = self._difix_epoch(step) if self.difix is not None else "off"
        t1 = time.time()
        realigned = self._realign_depth() if self.depth_estimator is not None else None
        if realigned is not None and cfg.stream_depth_render_align > 0 and cfg.with_eval3d:
            realigned += self._render_align_depth(step)
        fb = self._depth_feedback(step) if self.depth_estimator is not None and cfg.stream_depth_feedback > 0 else None
        t2 = time.time()
        refit = s.request_refit(cfg.stream_cycle_refit_min)
        msg = (f"[Cycle] epoch {self._epoch} @ step {step} ({n_train} keyframes): difix {difix} "
               f"{(t1 - t0):.1f}s | depth {'-' if realigned is None else f're-aligned {realigned}'} "
               f"{(t2 - t1) * 1000:.0f}ms{'' if fb is None else f' (feedback {fb[0]} views, agree {fb[1]:.2f})'} | ace {'refit started' if refit else s.stats()['job'] or 'idle'}")
        if self._epoch <= 3 or self._epoch % 10 == 0 or refit or (realigned or 0) > 0:
            print("\n" + msg)
        if self.dashboard is not None:
            self.dashboard.update_training(epoch=self._epoch)
        if self.world_rank == 0 and cfg.tb_every > 0:
            self.writer.add_scalar("stream/epoch", self._epoch, step)
            self.writer.add_scalar("stream/cycle_difix_s", t1 - t0, step)

    def _difix_epoch(self, step: int) -> str:
        """Cheap per-epoch Difix: a few views at reduced scale into a rolling pseudo pool."""
        from hypersplat.stream.difix_online import SCALES, _RoundSkipped

        cfg = self.cfg
        if self.difix.stop_reason:
            return "stopped"
        start = int(cfg.difix_cycle_start_frac * self.stream_horizon.value)
        if step < start:
            return f"waiting (starts at step {start})"
        src = self._difix_sources()
        if src is None or len(src.c2ws) < 2:
            return "no sources"
        if not self.stream.gpu_lease.try_acquire("difix"):
            return "skipped (ACE refit holds the GPU)"
        try:
            self.difix.scale_idx = max(self.difix.scale_idx, SCALES.index(cfg.difix_epoch_scale))
            torch.cuda.empty_cache()
            self.difix.begin_round(src, step, max_views=cfg.difix_epoch_views, max_angle=cfg.difix_epoch_max_angle)
            self.difix.step_round(lambda c2w, K, w, h: self._render_uint8(c2w, K, w, h, step), budget_s=None)
            views, _, _ = self.difix.finish_round(step)
        except _RoundSkipped:
            return "skipped (OOM)"
        finally:
            torch.cuda.empty_cache()
            self.stream.release_gpu("difix")
        self._pseudo_pool = (self._pseudo_pool + list(views))[-cfg.difix_online_views:]
        self.stream.set_pseudo(self._pseudo_pool)
        return f"+{len(views)} ({len(self._pseudo_pool)} pooled)"

    @torch.no_grad()
    def _depth_feedback(self, step: int) -> Tuple[int, float]:
        """Depth Anything on the 3DGS render and on the camera frame of a few train keyframes
        (round-robin, half resolution); their disagreement becomes the per-pixel trust of the
        depth loss (hypersplat/stream/depth.render_consistency)."""
        from hypersplat.stream.depth import render_consistency
        from hypersplat.stream.schema import select_rows

        cfg, s = self.cfg, self.stream
        _, table = s.views()
        table = select_rows(table, kind="real", split="train")
        if table.num_rows == 0 or s.transform is None:
            return 0, 0.0
        kids = table.column("kid").to_pylist()
        rr = getattr(self, "_feedback_rr", 0) % len(kids)
        order = kids[rr:] + kids[:rr]
        self._feedback_rr = rr + cfg.stream_depth_feedback
        row = {kid: i for i, kid in enumerate(kids)}
        c2ws = np.array(table.column("c2w").to_pylist()).reshape(-1, 4, 4)
        K0 = np.array(table.column("K")[0].as_py()).reshape(3, 3)
        W0, H0 = int(table.column("width")[0].as_py()), int(table.column("height")[0].as_py())
        n, score = 0, 0.0
        for kid, disp, _, _, _ in s.anchor_jobs(order)[: cfg.stream_depth_feedback]:
            h, w = disp.shape            # cached disparity is half res: render at that size
            K = K0.copy()
            K[0] *= w / W0
            K[1] *= h / H0
            rgb = self._render_uint8(c2ws[row[kid]], K, w, h, step)
            wmap, sc = render_consistency(disp, self.depth_estimator.predict(rgb))
            s.set_depth_weight(kid, wmap, sc)
            n, score = n + 1, score + sc
        return n, (score / n if n else 0.0)

    def _realign_depth(self) -> int:
        """Re-fit cached depth of keyframes ACE re-localized since their last alignment."""
        from hypersplat.stream.depth import dense_depth

        s = self.stream
        n = 0
        for kid, disp, pix, z, version in s.realign_jobs():
            res = dense_depth(disp, None, pix, z, guide_radius=0)
            s.set_depth(kid, res.depth, {"realign": True, "rel_err": res.align.rel_err if res.align.ok else None},
                        version=version)
            n += res.depth is not None
        return n

    @torch.no_grad()
    def _depth_guard_update(self, err: float, step: int) -> float:
        """Depth-weight multiplier from the render-vs-Depth-Anything error trend (see
        stream_depth_guard)."""
        cfg = self.cfg
        g = getattr(self, "_dguard", None)
        if g is None:
            g = self._dguard = {"ema": err, "best": err, "boost": 1.0, "n": 0}
        if err != err or err == float("inf"):
            return g["boost"]
        g["n"] += 1
        g["ema"] = 0.98 * g["ema"] + 0.02 * err
        if g["n"] > 50:   # let the EMA settle before it defines "best"
            # best creeps up so new keyframes / re-aligned maps can reset the reference
            g["best"] = min(g["best"] * 1.0005, g["ema"])
            if g["ema"] > g["best"] * (1.0 + cfg.stream_depth_guard_tol):
                g["boost"] = min(cfg.stream_depth_guard_max, g["boost"] * 1.02)
            else:
                g["boost"] = max(1.0, g["boost"] * 0.99)
        if self.writer is not None and cfg.tb_every > 0 and step % cfg.tb_every == 0:
            self.writer.add_scalar("train/depth_guard_boost", g["boost"], step)
            self.writer.add_scalar("train/depth_err_ema", g["ema"], step)
        return g["boost"]

    def _render_align_depth(self, step: int) -> int:
        """Closed depth loop: re-fit the monocular depth of a few train keyframes (round-robin)
        to confident pixels of the current render as well as ACE's points; kept only when it
        predicts held-out ACE points at least as well (hypersplat/stream/depth.align_disparity)."""
        from hypersplat.stream.depth import dense_depth
        from hypersplat.stream.schema import select_rows

        cfg = self.cfg
        s = self.stream
        _, table = s.views()
        table = select_rows(table, kind="real", split="train")
        if table.num_rows == 0 or s.transform is None:
            return 0
        kids = table.column("kid").to_pylist()
        rr = getattr(self, "_anchor_rr", 0) % len(kids)
        order = kids[rr:] + kids[:rr]
        jobs = s.anchor_jobs(order)[: cfg.stream_depth_render_align]
        self._anchor_rr = rr + cfg.stream_depth_render_align
        if not jobs:
            return 0
        row = {kid: i for i, kid in enumerate(kids)}
        c2ws = np.array(table.column("c2w").to_pylist()).reshape(-1, 4, 4)
        K0 = np.array(table.column("K")[0].as_py()).reshape(3, 3)
        W0, H0 = int(table.column("width")[0].as_py()), int(table.column("height")[0].as_py())
        to_ace = 1.0 / float(np.linalg.norm(s.transform[:3, 0]))  # trainer units -> ACE units
        n = 0
        for kid, disp, pix, z, version in jobs:
            h, w = disp.shape
            K = K0.copy()
            K[0] *= w / W0
            K[1] *= h / H0
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=torch.as_tensor(c2ws[row[kid]], dtype=torch.float32, device=self.device)[None],
                Ks=torch.as_tensor(K, dtype=torch.float32, device=self.device)[None],
                width=w, height=h, sh_degree=cfg.sh_degree, near_plane=cfg.near_plane,
                far_plane=cfg.far_plane, render_mode="ED")
            ed, a = renders[0, ..., 0], alphas[0, ..., 0]
            std = info["render_depth_var"][0, ..., 0].sqrt()
            conf = (a > 0.95) & (ed > 0) & (std < cfg.stream_depth_conf_tau * ed)
            conf[1::2] = False  # every other row/column is plenty for a 2-parameter fit
            conf[:, 1::2] = False
            ys, xs = torch.nonzero(conf, as_tuple=True)
            if len(ys) == 0:
                continue
            ys_n, xs_n = ys.cpu().numpy(), xs.cpu().numpy()
            anchors = (disp[ys_n, xs_n], ed[ys, xs].cpu().numpy() * to_ace)
            res = dense_depth(disp, None, pix, z, guide_radius=0, anchors=anchors)
            if res.depth is not None and res.align.anchored:
                s.set_depth(kid, res.depth, {"realign": True, "anchored": True, "rel_err": res.align.rel_err},
                            version=version)
                n += 1
        return n

    def _difix_chunk(self, step: int) -> bool:
        """Advance the pending stream round by one chunk; True once its views are in training."""
        from hypersplat.stream.difix_online import _RoundSkipped

        if not self.stream.gpu_lease.try_acquire("difix"):
            return False  # an ACE refit holds the GPU: next chunk slot
        try:
            torch.cuda.empty_cache()
            done = self.difix.step_round(lambda c2w, K, w, h: self._render_uint8(c2w, K, w, h, step),
                                         budget_s=self.cfg.difix_chunk_s)
        except _RoundSkipped:
            return False
        finally:
            torch.cuda.empty_cache()  # hand the blocks back before ACE takes the GPU
            self.stream.release_gpu("difix")
        if not done:
            return False
        views, _, _ = self.difix.finish_round(step)
        self.stream.set_pseudo(views)  # the stream sampler sees them on its next draw
        if self._final_difix == "running":
            self._final_difix = "done"
            self._stop_at = step + self.cfg.difix_final_tail_steps
            print(f"\n[WorldStop] Final Difix round done at step {step}; "
                  f"{self.cfg.difix_final_tail_steps} more steps on its pseudo views")
        return True

    def step_genvs(self, step):
        """Generate a new view and inject it filter dataset."""
        if not self.cfg.genvs_ckpt: return
        
        # [Adaptive Interpolation] Determine Mode based on Convergence
        # If loss is changing rapidly, we need coverage (Deterministic)
        # If loss is flat, we need regularization (Random Jitter)
        # Threshold: 1e-4 improvement per step (smoothed)
        
        # Determine Mode
        loss_delta = 1.0
        if len(self.loss_history) > 100:
             loss_window = self.loss_history[-100:]
             # Average improvement over last 100 steps
             loss_delta = abs(loss_window[0] - loss_window[-1]) / 100.0
             
        convergence_threshold = 1e-4
        if loss_delta < convergence_threshold:
             self.adaptive_mode = "stochastic"
        else:
             self.adaptive_mode = "deterministic"
             
        # print(f"[GeNVS] Loss Delta: {loss_delta:.6f} -> Mode: {self.adaptive_mode}")

        if self.adaptive_mode == "deterministic":
             # === MODE A: DETERMINISTIC GAP FILLING ===
             # Find large gaps in trajectory
             dists = torch.cdist(curr_cams, curr_cams)
             dists.fill_diagonal_(float('inf'))
             nearest_dists, nearest_indices = torch.min(dists, dim=1) # [N]
             
             gap_idx = torch.argmax(nearest_dists)
             neighbor_idx = nearest_indices[gap_idx]
             
             # Indices
             idx1 = gap_idx.item()
             idx2 = neighbor_idx.item()
             
             t = 0.5 # Rigid Midpoint
             jitter = 0.0 # No Jitter
             
        else:
             # === MODE B: STOCHASTIC JITTER (Refinement) ===
             # Still pick a gap to stay relevant, but add noise
             # Or pick random pair? Let's stick to gap to ensure we don't duplicate dense areas
             # But we relax the "Max Gap" constraint to be "Softmax" (probabilistic)
             
             # Sample pair probability proportional to gap size
             probs = torch.softmax(nearest_dists, dim=0)
             gap_idx = torch.multinomial(probs, 1).item()
             idx1 = gap_idx
             idx2 = nearest_indices[gap_idx].item()
             
             # Random t in [0.2, 0.8]
             t = 0.5 + torch.randn(1, device=self.device).item() * 0.15
             t = max(0.2, min(0.8, t))
             
             # Add Spatial Jitter
             jitter = 0.02 # 2% Scene Scale
             
        c2w1 = torch.from_numpy(self.parser.camtoworlds[idx1]).float().to(self.device)
        c2w2 = torch.from_numpy(self.parser.camtoworlds[idx2]).float().to(self.device)

        # Interpolate
        pos = c2w1[:3, 3] * t + c2w2[:3, 3] * (1-t)
        
        # Apply Jitter (if any)
        if jitter > 0:
             pos = pos + torch.randn(3, device=self.device) * (self.scene_scale * jitter)
        
        # LookAt Center (approximate scene center)
        center = torch.from_numpy(self.parser.points.mean(axis=0)).float().to(self.device) if len(self.parser.points) > 0 else torch.zeros(3, device=self.device)
        
        # LookAt Logic
        z_axis = center - pos
        z_axis = z_axis / torch.norm(z_axis)
        up = torch.tensor([0., 0., 1.], device=self.device) 
        
        # Slerp Rotation (Better than linear)
        # For simplicity in this loop, we just re-orient to look at center
        # This is "deterministic" based on geometry rather than random mix of R1/R2
        
        # Create LookAt Matrix (OpenCV: Z points forward to target)
        # Actually our renderer uses OpenCV: +Z forward, +Y down
        fwd = z_axis 
        x_axis = torch.cross(torch.tensor([0., -1., 0.], device=self.device), fwd) # Y-down up vector
        x_axis = x_axis / torch.norm(x_axis)
        y_axis = torch.cross(fwd, x_axis)
        
        R = torch.stack([x_axis, y_axis, fwd], dim=1)
        
        tgt_pose = torch.eye(4, device=self.device)
        tgt_pose[:3, :3] = R
        tgt_pose[:3, 3] = pos
        
        # Pick Source (Nearest)
        # Reuse idx1 as source since we are close to it
        src_idx = idx1
        # [GeNVS] Get Source Data using Parser Index (Bypass Dataset to allow using Test images or mismatch indices)
        # Check Cache
        if src_idx in self.parser.tensor_cache:
            src_item = self.parser.tensor_cache[src_idx]
            src_img = src_item['image'].unsqueeze(0).to(self.device).permute(0, 3, 1, 2)
            src_K = src_item['K'].unsqueeze(0).to(self.device)
            src_pose = src_item['camtoworld'].unsqueeze(0).to(self.device)
        else:
            # Load from Disk (Manual logic from Dataset.__getitem__)
            img_path = self.parser.image_paths[src_idx]
            image = imageio.imread(img_path)[..., :3]
            
            # Undistort? Config says 'pinhole' usually implies undistorted or COLMAP handles it.
            # Dataset logic:
            cam_id = self.parser.camera_ids[src_idx]
            params = self.parser.params_dict[cam_id]
            if len(params) > 0:
                # Undistort
                mapx, mapy = self.parser.mapx_dict[cam_id], self.parser.mapy_dict[cam_id]
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = self.parser.roi_undist_dict[cam_id]
                image = image[y : y + h, x : x + w]
            
            src_img = torch.from_numpy(image).float().to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
            src_K = torch.from_numpy(self.parser.Ks_dict[cam_id].copy()).float().to(self.device).unsqueeze(0)
            src_pose = torch.from_numpy(self.parser.camtoworlds[src_idx]).float().to(self.device).unsqueeze(0)
            
        # [GeNVS] Resize to multiple of 32
        H, W = src_img.shape[-2:]
        new_H = (H // 32) * 32
        new_W = (W // 32) * 32
        
        if new_H != H or new_W != W:
            src_img = F.interpolate(src_img, size=(new_H, new_W), mode='bilinear', align_corners=False)
            
            # Update Intrinsics (Clone to avoid mutating cache)
            src_K = src_K.clone()
            scale_x = new_W / W
            scale_y = new_H / H
            src_K[:, 0, 0] *= scale_x
            src_K[:, 0, 2] *= scale_x
            src_K[:, 1, 1] *= scale_y
            src_K[:, 1, 2] *= scale_y
        
        # Target Pose
        # Pipeline expects [1, 4, 4]
        tgt_pose_input = tgt_pose.unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            if self.cfg.genvs_depth_only:
                # DEPTH INJECTION MODE (Fast - no diffusion)
                # Use sample_depth_only which skips UNet entirely
                depth_result = self.genvs_pipeline.sample_depth_only(
                    src_img, src_pose, src_K, tgt_pose_input, src_K
                )  # [1, 1, H, W]
                
                # Store depth for supervision (not as RGB)
                gen_depth = depth_result[0].detach().cpu()  # [1, H, W]
                
                new_idx = len(self.parser.image_names)
                
                # For depth injection, we still need a placeholder image
                # Use the source image resized as placeholder (won't be used for RGB loss)
                placeholder_img = F.interpolate(src_img, size=(new_H, new_W), mode='bilinear', align_corners=False)
                placeholder_img = (placeholder_img[0].permute(1, 2, 0) / 255.0).detach().cpu() * 255.0  # [H, W, 3]
                
                cached_data = {
                    "K": src_K.squeeze(0).detach().cpu(),
                    "camtoworld": tgt_pose.detach().cpu(),
                    "image": placeholder_img.detach().cpu(),
                    "image_id": new_idx,
                    "genvs_depth": gen_depth,  # [1, H, W] - GeNVS-generated depth
                    "is_depth_only": True  # Flag for training loop
                }
                print(f"[GeNVS-Depth] Generated depth for novel pose (shape: {gen_depth.shape})")
            else:
                # RGB INJECTION MODE with SDS (Master Tuner)
                # We need to perform optimization here or just save the view for next iteration
                # But SDS needs differentiable rendering within the loop.
                # Since 'step_genvs' is called inside the loop, we can just return the data
                # AND compute the SDS loss right now against the current splats?
                
                # Actually, step_genvs is designed to *add data* to the dataset.
                # SDS typically runs as an *additional loss term* on a random view.
                
                # For this implementation (Latent Injection), we will calculate SDS loss
                # and backward it immediately to update the Gaussians.
                
                # 1. Render current view with gradients
                # We need to re-render because we need the graph attached
                # But 'step_genvs' separates generation from training?
                # The current architecture expects 'step_genvs' to just add a static image to the cache.
                
                # HYBRID APPROACH:
                # Use the pipeline to generate a "perfect" target image (Hallucination)
                # Then add that image to the dataset. This is "Iterative Dataset Update" (not SDS but close).
                # This matches the user request "inject latent space" by injecting the result of the latent space.
                
                # Hallucination Generation
                result = self.genvs_pipeline.sample_batch(
                    src_img, src_pose, src_K, tgt_pose_input, src_K, num_steps=30
                )
                
                # Result is [-1, 1]. Convert to [0, 255] float
                gen_tensor = (result[0] * 0.5 + 0.5) * 255.0  # [3, H, W]
                gen_tensor = gen_tensor.permute(1, 2, 0)  # [H, W, 3]
                gen_tensor = gen_tensor.detach().cpu().clone()
                
                new_idx = len(self.parser.image_names)
                
                cached_data = {
                    "K": src_K.squeeze(0).detach().cpu(),
                    "camtoworld": tgt_pose.detach().cpu(),
                    "image": gen_tensor.detach().cpu(),
                    "image_id": new_idx,
                    "is_depth_only": False
                }
        self.parser.tensor_cache[new_idx] = cached_data
        
        fname = f"gpu_gen_{new_idx}"
        self.parser.image_names.append(fname)
        self.parser.image_paths.append("GPU_MEMORY")
        
        # Update arrays
        new_c2w = tgt_pose.detach().cpu().numpy()[None]
        self.parser.camtoworlds = np.concatenate([self.parser.camtoworlds, new_c2w], axis=0)
        
        # Safe camera_id lookup - src_idx may be out of bounds for trainset.indices
        if src_idx < len(self.trainset.indices):
            real_src_idx = self.trainset.indices[src_idx]
        else:
            real_src_idx = src_idx % len(self.parser.camera_ids)
        cam_id = self.parser.camera_ids[real_src_idx]
        self.parser.camera_ids.append(cam_id)
        
        # Rebuild Dataset Indices
        self.trainset.rebuild_indices()
        
        print(f"[GeNVS-GPU] Injected View {new_idx} (Total: {len(self.trainset)})")

    def hot_reload_data(self) -> int:
        """Hot-reload new images/poses during streaming training.
        
        Returns:
            Number of new training images added.
        """
        if not self.cfg.streaming or not STREAMING_AVAILABLE:
            return 0
            
        try:
            # Refresh the parser (reloads COLMAP data from disk)
            new_images = self.parser.refresh()
            if new_images > 0:
                print(f"[Streaming] Hot-reloading {new_images} new images...")
                
                # Rebuild dataset indices
                new_train_images = self.trainset.rebuild_indices()
                
                # Update appearance embeddings if using app_opt
                if self.cfg.app_opt and hasattr(self, 'app_module'):
                    old_num = self.app_module.num_cameras
                    new_num = len(self.trainset)
                    if new_num > old_num:
                        # Extend the embedding layer
                        self.app_module.extend_embeddings(new_num - old_num)
                        
                print(f"[Streaming] Added {new_train_images} new training images.")
                return new_train_images
        except Exception as e:
            print(f"[Streaming] Hot-reload failed: {e}")
        return 0

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        opacity_keep: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        if opacity_keep is not None:  # viewer-only cull: zero opacity drops the splat
            opacities = opacities * opacity_keep

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            ut_params=self.ut_params,
            **kwargs,
        )
        if masks is not None:
             # masks is [B, 1, H, W] or [1, H, W]
             # render_colors is [1, H, W, 3] usually
             if masks.ndim == 4:
                 masks = masks[0] # [1, H, W]
             elif masks.ndim == 2:
                 masks = masks.unsqueeze(0) # [1, H, W]
             
             # Expand for broadcasting if needed, but indexing with boolean mask requires matching shape prefix
             # render_colors: [1, H, W, 3]
             # mask: [1, H, W] -> boolean mask [1, H, W]
             # render_colors[~masks.bool()] = 0
             # Ensure types
             bool_mask = masks.bool()
             # Broadcast to channels? No, standard indexing should work if first dims match.
             # render_colors[~bool_mask] sets all channels to 0.
             if bool_mask.shape == render_colors.shape[:-1]:
                  render_colors[~bool_mask] = 0
             else:
                  # Shape mismatch fallback
                  pass
        return render_colors, render_alphas, info

    def lbfgs_refine_round(self, step: int, next_batch):
        """One L-BFGS round on a fixed view batch; reverts if it does not reduce that batch's loss.

        ``next_batch()`` returns one training batch (the loop's own loader, so stream runs get
        their pose/exposure handling)."""
        cfg, device = self.cfg, self.device
        names = list(self.splats.keys())
        views = []
        for _ in range(cfg.lbfgs_views):
            d = next_batch()
            cam = d["camtoworld"].to(device)
            if getattr(self, "device_poses", None) is not None and "kid" in d:
                cam = self.device_poses.camtoworlds(d["kid"], cam)
            iid = d["image_id"].to(device)
            slots = None
            with torch.no_grad():
                if cfg.pose_opt:
                    cam = self.pose_adjust(cam, iid)
                if self.view_adjust is not None and "kid" in d:
                    slots = self.view_adjust.slot_ids(d["kid"], cam, d.get("is_pseudo"))
                    cam = self.view_adjust.adjust_pose(cam, slots)
            views.append((cam, d["K"].to(device), d["image"].to(device) / 255.0, iid, slots))
        sh_degree = cfg.sh_degree

        # Rescale each group by its Adam lr: u = p / s. Gradients wrt u are g * s.
        scale = {k: max(float(self.optimizers[k].param_groups[0]["lr"]), 1e-6) for k in names}
        u = {k: torch.nn.Parameter(self.splats[k].detach().clone() / scale[k]) for k in names}
        backup = {k: self.splats[k].detach().clone() for k in names}

        def evaluate(backward: bool) -> float:
            total = 0.0
            for cam, K, px, iid, slots in views:
                with torch.set_grad_enabled(backward):
                    colors, _, _ = self.rasterize_splats(
                        camtoworlds=cam, Ks=K, width=px.shape[2], height=px.shape[1],
                        sh_degree=sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                        image_ids=iid, render_mode="RGB")
                    if slots is not None:
                        colors = self.view_adjust.adjust_colors(colors, slots)
                    colors = torch.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
                    if cfg.ssim_impl == "fused":
                        from fused_ssim import fused_ssim
                        ssim = fused_ssim(colors.permute(0, 3, 1, 2), px.permute(0, 3, 1, 2), padding="valid")
                    else:
                        ssim = self.ssim(colors.permute(0, 3, 1, 2), px.permute(0, 3, 1, 2))
                    loss = ((1.0 - cfg.ssim_lambda) * F.l1_loss(colors, px)
                            + cfg.ssim_lambda * (1.0 - ssim)) / len(views)
                    if cfg.opacity_reg > 0.0:
                        loss = loss + cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean() / len(views)
                    if cfg.scale_reg > 0.0:
                        loss = loss + cfg.scale_reg * torch.exp(
                            torch.clamp(self.splats["scales"], max=10.0)).mean() / len(views)
                if backward:
                    loss.backward()
                total += float(loss)
            return total

        def closure():
            for k in names:
                self.splats[k].data.copy_(u[k].detach() * scale[k])
                self.splats[k].grad = None
            loss = evaluate(True)
            ok = math.isfinite(loss)
            for k in names:
                g = self.splats[k].grad
                if g is None or not ok:
                    u[k].grad = torch.zeros_like(u[k])
                else:
                    u[k].grad = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) * scale[k]
                self.splats[k].grad = None
            return torch.tensor(loss if ok else 1e6, device=device)

        try:
            before = float(closure())
            # torch's first L-BFGS step is lr / |g|_1: with ~1e7 parameters that is ~1e-7 and a short
            # round never recovers. Pick the variable scale a so the first step moves the largest
            # element by lbfgs_lr Adam-lr units (a = lr * |g|_1 / |g|_inf), then run with lr = 1.
            l1 = sum(float(x.grad.abs().sum()) for x in u.values())
            linf = max(float(x.grad.abs().max()) for x in u.values())
            if not (l1 > 0 and linf > 0 and math.isfinite(l1)):
                raise FloatingPointError("degenerate gradient")
            a = cfg.lbfgs_lr * l1 / linf
            for k in names:
                scale[k] *= a
                u[k].data.copy_(backup[k] / scale[k])
            opt = torch.optim.LBFGS(list(u.values()), lr=1.0, max_iter=cfg.lbfgs_max_iter,
                                    max_eval=int(cfg.lbfgs_max_iter * 2.5), history_size=cfg.lbfgs_history,
                                    line_search_fn="strong_wolfe")
            opt.step(closure)
            for k in names:
                self.splats[k].data.copy_(u[k].detach() * scale[k])
            after = evaluate(False)
        except FloatingPointError:
            before, after = 0.0, float("inf")
        except torch.cuda.OutOfMemoryError:
            before, after = 0.0, float("inf")
            torch.cuda.empty_cache()
        bad = (not math.isfinite(after)) or after >= before
        for k in names:
            if bad or not torch.isfinite(self.splats[k]).all():
                self.splats[k].data.copy_(backup[k])
            self.splats[k].grad = None
        del u, backup
        # exposure/pose-adjust modules saw gradients from the closure; the main optimizers must not
        for o in self.pose_optimizers + self.app_optimizers + getattr(self, "stream_exposure_optimizers", []):
            o.zero_grad(set_to_none=True)
        if step % (cfg.lbfgs_every * 10) == 0 or bad:
            print(f"[LBFGS] step {step}: batch loss {before:.5f} -> {after:.5f}" + (" (reverted)" if bad else ""))

    def _ensure_latent_enc(self, device):
        """Frozen DINO / Depth Anything stream encoder; shares the depth model when it is loaded."""
        if self._latent_enc is None:
            cfg = self.cfg
            shared = getattr(getattr(self, "depth_estimator", None), "model", None)
            self._latent_enc = LatentEncoder(
                input_side=cfg.latent_side, layer=cfg.latent_layer,
                backend=cfg.latent_backend, device=str(device),
                model=shared if cfg.latent_backend == "depth_anything" else None,
            ).load()
        return self._latent_enc

    def _build_vjepa(self, device):
        """V-JEPA 2 predictive latent loss; secondary tokens come from the shared LatentEncoder."""
        from hypersplat.stream.vjepa_latent import VJEPALatentLoss, load_vjepa2_vitl

        cfg = self.cfg
        if cfg.vjepa_checkpoint is None:
            raise ValueError("latent_model='vjepa' needs vjepa_checkpoint (path to vitl.pt)")
        enc, pred = load_vjepa2_vitl(cfg.vjepa_checkpoint, side=cfg.vjepa_side, device=str(device))
        sec_dim = self._latent_enc.features(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
        from hypersplat.stream.hf_lock import hf_load

        with hf_load():  # the hook's projection is a new module: build it outside any concurrent load
            m = VJEPALatentLoss(enc, pred, self._latent_enc, sec_dim=sec_dim, side=cfg.vjepa_side,
                                target_frac=cfg.vjepa_target_frac).to(device)
        self.latent_hook_optimizers = [torch.optim.Adam(m.trainable_parameters(), lr=cfg.vjepa_hook_lr)]
        pending = self._pending_latent_ckpt
        if pending is not None:
            m.load_hook_state(pending["proj"])
            for opt, state in zip(self.latent_hook_optimizers, pending["optimizers"]):
                opt.load_state_dict(state)
            self._pending_latent_ckpt = None
            print("[VJEPA] restored secondary-stream projection from checkpoint")
        return m

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        if cfg.resume:
            print(f"Resuming from checkpoint: {cfg.resume}")
            ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
            init_step = ckpt["step"] + 1
            
            # Handle Splat Resizing
            # Check if checkpoint num_gaussians matches current model
            ckpt_N = ckpt["splats"]["means"].shape[0]
            curr_N = self.splats["means"].shape[0]
            
            if ckpt_N != curr_N:
                print(f"[WARNING] Resizing model from {curr_N} to {ckpt_N} Gaussians to match checkpoint.")
                # Re-create parameters in self.splats with correct size
                # We need to preserve the device and dtype
                new_splats = torch.nn.ParameterDict()
                for k, v in ckpt["splats"].items():
                    # v is the tensor from checkpoint state dict
                    # We create a new Parameter with the same shape
                    # but we don't need to init it, load_state_dict will fill it.
                    # Actually, we can just replace the parameter with a new one of correct shape.
                    # We use the checkpoint data directly to initialize to avoid processing overhead later
                    new_splats[k] = torch.nn.Parameter(torch.empty_like(v).to(device))
                
                self.splats = new_splats.to(device)
                
                # Re-create optimizers because parameters have changed
                print("[INFO] Re-initializing optimizers for resized model...")
                BS = cfg.batch_size * world_size
                optimizer_class = torch.optim.Adam
                if cfg.sparse_grad: optimizer_class = torch.optim.SparseAdam
                elif cfg.visible_adam: optimizer_class = SelectiveAdam
                
                # Re-map params to LRs
                params_list = [
                    ("means", self.splats["means"], cfg.means_lr * self.scene_scale),
                    ("scales", self.splats["scales"], cfg.scales_lr),
                    ("quats", self.splats["quats"], cfg.quats_lr),
                    ("opacities", self.splats["opacities"], cfg.opacities_lr),
                    ("sh0", self.splats["sh0"], cfg.sh0_lr),
                    ("shN", self.splats["shN"], cfg.shN_lr),
                ]
                if "features" in self.splats:
                     params_list.append(("features", self.splats["features"], cfg.sh0_lr))
                if "colors" in self.splats:
                     params_list.append(("colors", self.splats["colors"], cfg.sh0_lr))

                self.optimizers = {
                    name: optimizer_class(
                        [{"params": param, "lr": lr * math.sqrt(BS), "name": name}],
                        eps=1e-15 / math.sqrt(BS),
                        betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                    )
                    for name, param, lr in params_list
                }
            
            # Now load state dicts
            self.splats.load_state_dict(ckpt["splats"])
            if ckpt.get("latent_hook") is not None:
                self._pending_latent_ckpt = {
                    "proj": ckpt["latent_hook"], "optimizers": ckpt.get("latent_hook_optimizers", []),
                }
            
            # Load optimizers
            if "optimizers" in ckpt:
                for k, v in ckpt["optimizers"].items():
                    if k in self.optimizers:
                        try:
                            self.optimizers[k].load_state_dict(v)
                        except ValueError as e:
                            print(f"Warning: Could not load optimizer state for {k}: {e}")
            
            # Load aux optimizers
            for opt_name in ["pose_optimizers", "app_optimizers", "bil_grid_optimizers"]:
                if opt_name in ckpt:
                    target_opts = getattr(self, opt_name)
                    for i, state in enumerate(ckpt[opt_name]):
                        if i < len(target_opts):
                            try:
                                target_opts[i].load_state_dict(state)
                            except Exception as e:
                                print(f"Warning: Could not load {opt_name} state: {e}")

            print(f"Resumed from step {init_step}")

        # Initialize Schedulers (AFTER resume, so they attach to the correct optimizers)
        # A stream keeps revealing new space late in the run: decay the position LR to 10%
        # instead of 1% so Gaussians seeded then can still move into place
        means_lr_final = 0.1 if self.stream is not None else 0.01
        horizon = self.stream_horizon if self.stream is not None else None
        if horizon is not None:
            # Same decay, measured against the moving end of a stream run
            means_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizers["means"], lambda s: horizon.lr_factor(s, means_lr_final)
            )
        else:
            means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=means_lr_final ** (1.0 / max_steps)
            )
        schedulers = [means_scheduler]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        elif horizon is not None and self.view_adjust is not None and self.view_adjust.pose_optimizers:
            # Per-keyframe stream pose deltas decay like batch pose_opt, against the moving end
            schedulers.append(
                torch.optim.lr_scheduler.LambdaLR(
                    self.view_adjust.pose_optimizers[0], lambda s: horizon.lr_factor(s, 0.01)
                )
            )
        if cfg.use_bilateral_grid:
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        # Load scheduler states if we resumed
        if cfg.resume and "schedulers" in list(locals().keys()) and "schedulers" in ckpt:
             for s, state in zip(schedulers, ckpt["schedulers"]):
                 try:
                     s.load_state_dict(state)
                 except:
                     pass

        num_workers = cfg.data_loader_workers
        if os.name == 'nt' and num_workers > 0:
             print("[SimpleTrainer] Windows detected. FORCING num_workers=0 to prevent Viser crash.")
             num_workers = 0
             # Actually, let's respect the config. If user passes 0, it's 0. 
             # If user passes 4, and it works, good. If not, they should pass 0.
             # But default is 4.
             pass
        def make_trainloader():
            if self.stream is not None:
                # Endless sampler over the live Arrow snapshot; workers would each hold a copy
                return torch.utils.data.DataLoader(self.trainset, batch_size=cfg.batch_size,
                                                   num_workers=0, pin_memory=True)
            return torch.utils.data.DataLoader(
                self.trainset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
                pin_memory=True,
            )

        trainloader = make_trainloader()
        trainloader_iter = iter(trainloader)
        difix_steps = self._difix_schedule(max_steps)
        if difix_steps:
            print(f"[Difix-online] Rounds at steps {sorted(difix_steps)} ({cfg.difix_model_id})")

        # V-JEPA latent model: built once before the loop, after resume so a saved projection is restored
        if cfg.latent_loss and cfg.latent_model == "vjepa" and self._vjepa is None:
            self._ensure_latent_enc(device)
            self._vjepa = self._build_vjepa(device)

        # Early stopping: plateau of the EMA training loss. Disabled when the data changes
        # mid-run (streams, online Difix, GeNVS) and multi-GPU (ranks would have to agree).
        plateau = None
        if cfg.early_stopping:
            if cfg.streaming or cfg.genvs_ckpt or world_size > 1 or self.stream is not None or self.difix:
                print("[EarlyStop] Loss plateau disabled (stream / online Difix / GeNVS / distributed run)"
                      + ("; streams use [WorldStop] instead" if self.stream is not None else ""))
            else:
                plateau = PlateauDetector(
                    patience=cfg.early_stop_patience,
                    min_delta=cfg.early_stop_min_delta,
                    min_steps=cfg.early_stop_min_steps,
                )

        # Training loop. A stream run has no fixed length: max_steps follows its horizon.
        global_tic = time.time()
        if horizon is not None:
            max_steps = horizon.value
            pbar = tqdm.tqdm(itertools.count(init_step), total=max_steps - init_step)
        else:
            pbar = tqdm.tqdm(range(init_step, max_steps))
        difix_deferred = False
        for step in pbar:
            self._profile_tick(step)
            self._pacer.tick(time.time())
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            stream_stop = False
            if self.stream is not None:
                self.stream.trainer_step = step
                if getattr(self, "device_poses", None) is not None:
                    self.device_poses.sync(step)
            if self.stream is not None and step % cfg.stream_ace_tick_every == 0:
                # Continuous ACE mapping: the sidecar runs this step's mapping work on the GPU
                # alongside the 3DGS step (no-op in refit mode)
                self.stream.tick()
            if self.stream is not None and step % cfg.stream_sync_every == 0:
                stream_stop = self._stream_sync(step)
                if horizon.value != max_steps:
                    max_steps = horizon.value
                    difix_steps = self._difix_schedule(max_steps)
                    pbar.total = max(max_steps - init_step, 1)
                    pbar.refresh()

            # Streaming: check for new poses periodically
            if cfg.streaming and step > 0 and step % cfg.streaming_check_interval == 0:
                if self.streaming_manager and self.streaming_manager.watcher.has_new_data():
                    new_count = self.hot_reload_data()
                    if new_count > 0:
                        # Rebuild the dataloader with new images
                        trainloader = torch.utils.data.DataLoader(
                            self.trainset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            persistent_workers=(num_workers > 0),
                            pin_memory=True,
                        )
                        trainloader_iter = iter(trainloader)

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            if getattr(self, "device_poses", None) is not None and "kid" in data:
                camtoworlds = camtoworlds_gt = self.device_poses.camtoworlds(data["kid"], camtoworlds)
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            
            is_depth_only = "is_depth_only" in data and data["is_depth_only"].all()
            if (cfg.depth_loss or is_depth_only) and "points" in data:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]
            else:
                # Fallback to avoid unbound variables if depth mode is active but sampling failed
                points = None
                depths_gt = None

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                is_pseudo = data["is_pseudo"].to(device) if "is_pseudo" in data else None
                if is_pseudo is None or not is_pseudo.any():
                    camtoworlds = self.pose_adjust(camtoworlds, image_ids)
                elif not is_pseudo.all():
                    camtoworlds = torch.where(is_pseudo[:, None, None], camtoworlds,
                                              self.pose_adjust(camtoworlds, image_ids))

            view_slots = None
            if self.view_adjust is not None and "kid" in data:
                view_slots = self.view_adjust.slot_ids(data["kid"], camtoworlds, data.get("is_pseudo"))
                camtoworlds = self.view_adjust.adjust_pose(camtoworlds, view_slots)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            render_mode = "RGB"
            # Stream dense depth: supervise only views that have an aligned depth map
            stream_depth_gt = None
            if "depth" in data and cfg.stream_depth_lambda > 0 and step >= cfg.depth_warmup_steps:
                gt = data["depth"].to(device)
                if (gt > 0).float().mean() > 0.05:
                    stream_depth_gt = gt
            if cfg.depth_loss or is_depth_only or stream_depth_gt is not None:
                render_mode = "RGB+ED"

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode=render_mode,
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None
            if view_slots is not None:
                colors = self.view_adjust.adjust_colors(colors, view_slots)

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            # PREVENT NaN: Clean rendered colors before loss computation
            colors = torch.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Unified Loss Logic: Keep graph topology identical for all views to avoid autograd errors
            rgb_weight = 0.0 if is_depth_only else 1.0
            
            l1loss = F.l1_loss(colors, pixels)
            view_l1 = (colors.detach() - pixels).abs().mean(dim=(1, 2, 3)) if self.stream is not None else None
            if cfg.ssim_impl == "fused":
                from fused_ssim import fused_ssim
                ssimloss = 1.0 - fused_ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2),
                                            padding="valid")
            else:
                ssimloss = 1.0 - self.ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                )
            ssimloss = torch.nan_to_num(ssimloss, nan=0.0, posinf=0.0, neginf=0.0)
            
            # RGB branch always exists in the graph, but weight can be 0.0
            # Auxiliary terms (depth, tv, bg) accumulate here; the RGB term is added below
            loss = torch.zeros((), device=device)
                
            # Depth branch
            if (cfg.depth_loss or is_depth_only) and points is not None:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                
                # ROBUST COMBINED DEPTH LOSS (Pearson + Log + L1)
                # Works regardless of scale mismatch between ZoeDepth and 3DGS
                
                valid_rendered = depths > 0.01
                valid_gt = depths_gt > 0.1
                valid_mask = valid_rendered & valid_gt
                
                if valid_mask.sum() > 20:  # Need enough valid points
                    pred_valid = depths[valid_mask]
                    gt_valid = depths_gt[valid_mask]
                    
                    # 1. PEARSON CORRELATION LOSS (scale & shift invariant)
                    pred_mean = pred_valid.mean()
                    gt_mean = gt_valid.mean()
                    pred_centered = pred_valid - pred_mean
                    gt_centered = gt_valid - gt_mean
                    
                    pred_std = pred_centered.norm() + 1e-8
                    gt_std = gt_centered.norm() + 1e-8
                    
                    pearson_corr = (pred_centered * gt_centered).sum() / (pred_std * gt_std)
                    pearson_loss = 1.0 - pearson_corr  # 0 = perfect, 2 = anti-correlated
                    
                    # 2. LOG-SPACE L1 LOSS (handles wide depth ranges)
                    log_pred = torch.log(pred_valid.clamp(min=1e-6))
                    log_gt = torch.log(gt_valid.clamp(min=1e-6))
                    log_loss = F.l1_loss(log_pred, log_gt)
                    
                    # 3. SCALE-ALIGNED L1 (using median alignment)
                    scale_factor = pred_valid.median() / (gt_valid.median() + 1e-8)
                    gt_aligned = gt_valid * scale_factor
                    l1_loss_aligned = F.l1_loss(pred_valid, gt_aligned)
                    
                    # Combine: Pearson (0.4) + Log (0.3) + L1 (0.3)
                    depthloss = 0.4 * pearson_loss + 0.3 * log_loss + 0.3 * l1_loss_aligned
                    
                    # Control NaN/Inf
                    depthloss = torch.nan_to_num(depthloss, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # PRINCIPLED FIX: Depth loss scheduler (warm-up + linear ramp)
                    if step < cfg.depth_warmup_steps:
                        effective_depth_lambda = 0.0
                    elif step < cfg.depth_warmup_steps + cfg.depth_ramp_steps:
                        ramp_progress = (step - cfg.depth_warmup_steps) / cfg.depth_ramp_steps
                        effective_depth_lambda = cfg.depth_lambda * ramp_progress
                    else:
                        effective_depth_lambda = cfg.depth_lambda
                    
                    loss += depthloss * effective_depth_lambda
                else:
                    depthloss = torch.tensor(0.0, device=device)
            if view_slots is not None and self.view_adjust.exposure is not None:
                loss = loss + cfg.stream_exposure_reg * self.view_adjust.exposure_reg()
            if stream_depth_gt is not None and depths is not None:
                # Log-depth L1 against the ACE-aligned monocular depth, at its (half) resolution
                pred = F.adaptive_avg_pool2d(depths.permute(0, 3, 1, 2), stream_depth_gt.shape[-2:])[:, 0]
                valid = (stream_depth_gt > 0) & (pred > 1e-4)
                if valid.any():
                    err = (torch.log(pred[valid]) - torch.log(stream_depth_gt[valid])).abs()
                    fbw = data["depth_w"].to(device)[valid] if cfg.stream_depth_feedback > 0 and "depth_w" in data else None
                    if fbw is not None and not (cfg.stream_depth_conf_weight and "render_depth_var" in info):
                        sdl = (err * fbw).sum() / fbw.sum().clamp(min=1e-6)
                    elif cfg.stream_depth_conf_weight and "render_depth_var" in info:
                        # Fuzzy render pixels (depth spread over several surfaces) trust the
                        # monocular target less
                        var = F.adaptive_avg_pool2d(info["render_depth_var"].permute(0, 3, 1, 2),
                                                    stream_depth_gt.shape[-2:])[:, 0]
                        rel_std = var.sqrt() / pred.clamp(min=1e-4)
                        wgt = (1.0 / (1.0 + (rel_std / cfg.stream_depth_conf_tau) ** 2)).clamp(min=0.1)
                        wgt = wgt.detach()[valid]
                        if fbw is not None:
                            wgt = wgt * fbw
                        sdl = (err * wgt).sum() / wgt.sum()
                    else:
                        sdl = err.mean()
                    ramp = min(1.0, (step - cfg.depth_warmup_steps) / max(cfg.depth_ramp_steps, 1))
                    boost = 1.0
                    if cfg.stream_depth_guard:
                        boost = self._depth_guard_update(float(sdl), step)
                    loss = loss + cfg.stream_depth_lambda * ramp * boost * torch.nan_to_num(sdl, nan=0.0, posinf=0.0)
                    self._stream_depth_loss = float(sdl)
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # Masked Loss / Hard Masking (Static Extraction)
            if cfg.use_mask and "mask" in data:
                mask = data["mask"].to(device) # [1, H, W]
                # data["mask"] is usually [1, H, W] from our wrapper
                # pixels is [1, H, W, 3]
                # mask needs to be [1, H, W, 1] for broadcasting
                if mask.shape[0] != pixels.shape[0]:
                    # Batch mismatch or single mask
                    mask = mask.expand(pixels.shape[0], -1, -1)
                
                mask_hwc = mask.unsqueeze(-1) # [B, H, W, 1]
                if mask_hwc.dim() == 5: # [B, 1, H, W, 1]
                    mask_hwc = mask_hwc.squeeze(1)

                weighted_l1 = l1loss * mask_hwc * cfg.mask_weight
                # Normalization
                l1loss = weighted_l1.sum() / (mask_hwc.sum() * 3 + 1e-6)
                
                # Check for empty mask
                if mask_hwc.sum() < 10:
                     l1loss = torch.tensor(0.0, device=device)

                # Background Opacity Regulation (Clean up outside mask)
                if cfg.bg_reg > 0.0:
                     # Penalize opacity where mask is 0
                     # alphas is available from rasterize_splats return
                     bg_opacity = alphas * (1.0 - mask_hwc)
                     loss += cfg.bg_reg * bg_opacity.mean()

                if cfg.bg_reg > 0.0:
                     # Penalize opacity where mask is 0
                     # alphas is available from rasterize_splats return
                     bg_opacity = alphas * (1.0 - mask_hwc)
                     loss += cfg.bg_reg * bg_opacity.mean()

            else:
                l1loss = l1loss.mean()
             
            # ONLINE FLOATER ELIMINATION (DQN-style Heuristic)
            if (
                cfg.dqn_pruning
                and step > cfg.floater_prune_start
                and step % cfg.floater_prune_every == 0
            ):
                 self.eliminate_floaters(verbose=True)


            # Per-image weight (ACE-Zero pose confidence / pseudo-view trust), default 1.0
            view_weight = data["loss_weight"].float().mean().to(device) if "loss_weight" in data else 1.0
            loss = loss + ((1.0 - cfg.ssim_lambda) * l1loss + cfg.ssim_lambda * ssimloss) * rgb_weight * view_weight

            # Latent (frozen-ViT feature) loss: smoother, wider-basin gradient than L1/SSIM.
            # Inside the graph before backward() so it also shapes means2d grads / densification.
            latent_loss_val = None
            if cfg.latent_loss:
                lam = latent_lambda(step, cfg.latent_lambda, cfg.latent_warmup_steps, cfg.latent_anneal_end)
                if lam > 0.0:
                    self._ensure_latent_enc(device)
                    if cfg.latent_model == "vjepa":
                        if self._vjepa is None:
                            self._vjepa = self._build_vjepa(device)
                            self._pacer.discard()  # model load is not V-JEPA step cost
                        latent_loss_val = None
                        self._vjepa_alpha_log = None
                        k = self._pacer.take()  # 0 = skip; else steps since the last run
                        if k > 0:
                            use_alpha = cfg.vjepa_alpha_condition or cfg.vjepa_alpha_lambda > 0
                            vout = self._vjepa(
                                colors.permute(0, 3, 1, 2).clamp(0, 1), pixels.permute(0, 3, 1, 2),
                                patch_weight=cfg.latent_patch_weight, weight_floor=cfg.latent_weight_floor,
                                alpha=alphas.permute(0, 3, 1, 2) if use_alpha else None,
                                alpha_condition=cfg.vjepa_alpha_condition,
                                alpha_hole_max=cfg.vjepa_alpha_hole_max,
                                valid=(stream_depth_gt > 0) if stream_depth_gt is not None else None,
                            )
                            latent_loss_val = vout.latent
                            if vout.alpha_hole is not None and cfg.vjepa_alpha_lambda > 0:
                                loss = loss + cfg.vjepa_alpha_lambda * vout.alpha_hole * view_weight * k
                            self._vjepa_alpha_log = (vout.alpha_hole, vout.hole_frac)
                            # run steps carry k steps' worth of the term, so the expected gradient is unchanged
                            loss = loss + lam * latent_loss_val * view_weight * k
                    else:
                        f_render = self._latent_enc.features(colors.permute(0, 3, 1, 2).clamp(0, 1))
                        f_target = self._latent_enc.target_features(pixels.permute(0, 3, 1, 2))
                        weight = None
                        if cfg.latent_patch_weight:
                            gh, gw = self._latent_enc.grid(pixels.shape[1], pixels.shape[2])
                            weight = patch_error_weight(
                                colors.permute(0, 3, 1, 2).clamp(0, 1), pixels.permute(0, 3, 1, 2),
                                (gh, gw), floor=cfg.latent_weight_floor,
                            )
                        latent_loss_val = latent_cosine_loss(f_render, f_target, weight)
                        loss = loss + lam * latent_loss_val * view_weight

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                # Clamp scales to prevent exp() overflow (exp(10) ≈ 22000, exp(20) ≈ 485M)
                scales_clamped = torch.clamp(self.splats["scales"], max=10.0)
                loss += cfg.scale_reg * torch.exp(scales_clamped).mean()

            # DEBUG: Identify NaN source (runs only when loss is NaN)
            if not torch.isfinite(loss):
                print(f"\n[DEBUG NaN] Step {step}:")
                print(f"  l1loss = {l1loss.item():.6f}")
                print(f"  ssimloss = {ssimloss.item():.6f}")
                print(f"  base_loss (l1+ssim) = {(l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda).item():.6f}")
                if cfg.depth_loss:
                    print(f"  depthloss = {depthloss.item():.6f}")
                if cfg.opacity_reg > 0.0:
                    opacity_val = torch.sigmoid(self.splats["opacities"]).mean().item()
                    print(f"  opacity_mean = {opacity_val:.6f}")
                if cfg.scale_reg > 0.0:
                    scale_val = torch.exp(torch.clamp(self.splats["scales"], max=10.0)).mean().item()
                    scale_raw_max = self.splats["scales"].max().item()
                    scale_raw_min = self.splats["scales"].min().item()
                    print(f"  scale_exp_mean = {scale_val:.6f}")
                    print(f"  scale_raw range = [{scale_raw_min:.3f}, {scale_raw_max:.3f}]")
                print(f"  total_loss = {loss.item()}")
                # Check Gaussian parameters for NaN
                for k, v in self.splats.items():
                    if torch.isnan(v).any():
                        print(f"  ⚠️ {k} contains NaN!")
                    if torch.isinf(v).any():
                        print(f"  ⚠️ {k} contains Inf!")
            
            # PREVENT NaN: Clamp loss to reasonable range before backward
            # This prevents gradient explosion while still propagating gradients
            loss = torch.clamp(loss, min=0.0, max=100.0)
            
            loss.backward()
            
            # PREVENT NaN: Zero out any NaN gradients before clipping
            for param in self.splats.values():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            
            # PREVENT NaN: Aggressive gradient clipping (balanced: 0.1)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.splats.values()], max_norm=0.1
            )

            loss_val = loss.item()
            self._last_loss = loss_val
            if view_l1 is not None and "kid" in data and cfg.stream_hard_view_mix > 0:
                kids = data["kid"].tolist()
                if "is_pseudo" in data:
                    kids = [-1 if p else k for k, p in zip(kids, data["is_pseudo"].tolist())]
                self.trainset.report_loss(kids, view_l1.tolist())
            if loss_val == loss_val and abs(loss_val) != float("inf"):
                self._loss_ema = loss_val if getattr(self, "_loss_ema", None) is None else \
                    0.99 * self._loss_ema + 0.01 * loss_val
            desc = f"loss={loss_val:.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # Early stop: this step becomes the final one (ckpt/ply/eval saved below as at max_steps)
            stop_now = plateau is not None and step < max_steps - 1 and plateau.update(step, loss_val)
            stop_now = stop_now or (stream_stop and step < max_steps - 1)
            if stop_now and not stream_stop:
                print(
                    f"\n[EarlyStop] Loss plateau at step {step}: EMA {plateau.ema:.5f}, no "
                    f"{cfg.early_stop_min_delta:.2%} improvement since step {plateau.best_step}"
                )
            # >=: a stream's end can be pinned at or before the current step (tail steps of 0)
            is_final_step = step >= max_steps - 1 or stop_now

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                # pre-clip total grad norm (clip_grad_norm_ returns the unclipped norm)
                self.writer.add_scalar("train/grad_norm", grad_norm.item(), step)
                grad2d_state = self.strategy_state.get("grad2d")
                if grad2d_state is not None:
                    self.writer.add_scalar(
                        "train/grad2d_mean", grad2d_state.mean().item(), step
                    )
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if getattr(self, "_vjepa_alpha_log", None) is not None and self._vjepa_alpha_log[0] is not None:
                    self.writer.add_scalar("train/vjepa_alpha_hole", self._vjepa_alpha_log[0].item(), step)
                    self.writer.add_scalar("train/vjepa_hole_frac", self._vjepa_alpha_log[1], step)
                if latent_loss_val is not None:
                    self.writer.add_scalar("train/latent_loss", latent_loss_val.item(), step)
                    self.writer.add_scalar(
                        "train/latent_lambda",
                        latent_lambda(step, cfg.latent_lambda, cfg.latent_warmup_steps, cfg.latent_anneal_end),
                        step,
                    )
                if cfg.latent_loss and cfg.latent_model == "vjepa" and self._pacer.cost_s is not None:
                    self.writer.add_scalar("train/vjepa_interval", self._pacer.interval(), step)
                    self.writer.add_scalar("train/vjepa_cost_ms", self._pacer.cost_s * 1000.0, step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or is_final_step:
                try:
                    mem_alloc = torch.cuda.memory_allocated() / 1024**3
                    mem_max = torch.cuda.max_memory_allocated() / 1024**3
                    mem = mem_max if not math.isnan(mem_max) else mem_alloc
                    if math.isnan(mem): mem = 0.0
                except:
                    mem = 0.0
                
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print(f"Step: {step} | Mem: {mem:.2f}GB | Stats: {stats}")
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {
                    "step": step, 
                    "splats": self.splats.state_dict(),
                    "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
                    "pose_optimizers": [opt.state_dict() for opt in self.pose_optimizers],
                    "app_optimizers": [opt.state_dict() for opt in self.app_optimizers],
                    "bil_grid_optimizers": [opt.state_dict() for opt in self.bil_grid_optimizers],
                    "schedulers": [s.state_dict() for s in schedulers],
                }
                if self._vjepa is not None:
                    data["latent_hook"] = self._vjepa.sec_proj.state_dict()
                    data["latent_hook_optimizers"] = [o.state_dict() for o in self.latent_hook_optimizers]
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if self.view_adjust is not None:
                    data["stream_view_adjust"] = {"state": self.view_adjust.state_dict(),
                                                  **self.view_adjust.extra_state()}
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or is_final_step
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # JOGS-style pose optimization: interleaved phases for faster convergence
            # Phase 1 (GS-only): Optimize Gaussians, freeze poses
            # Phase 2 (Pose-refine): Optimize both, focus on pose refinement
            for optimizer in self.pose_optimizers:
                if cfg.pose_opt_jogs:
                    # Check if we're past warmup
                    if step < cfg.pose_warmup_steps:
                        # Warmup: zero gradients but don't step
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        # Determine if we're in pose phase
                        step_in_interval = step % cfg.pose_opt_interval
                        pose_phase_start = cfg.pose_opt_interval - cfg.pose_phase_steps
                        # [Adaptive Interpolation] Track Loss
                        if self.cfg.genvs_ckpt:
                            loss_val = loss.item() # Assuming 'loss' is the total loss
                            self.loss_history.append(loss_val)
                            if len(self.loss_history) > 1000: # Limit memory usage
                                self.loss_history.pop(0)
                        
                        if step_in_interval >= pose_phase_start:
                            # Pose phase: step the optimizer
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            # GS-only phase: accumulate gradients but don't step
                            optimizer.zero_grad(set_to_none=True)
                else:
                    # Original behavior: step every iteration
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers + getattr(self, "stream_exposure_optimizers", []):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in getattr(self, "latent_hook_optimizers", []):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # ROOT CAUSE FIX: Detect and repair NaN in parameters after optimizer step
            # This prevents cascade failure where NaN gradients corrupt all parameters
            for name, param in self.splats.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    num_bad = (torch.isnan(param) | torch.isinf(param)).sum().item()
                    print(f"[NaN REPAIR] Step {step}: {name} has {num_bad} NaN/Inf values, repairing")
                    if name in ("scales", "opacities"):
                        # Neutralize bad Gaussians (tiny + transparent) instead of turning
                        # them into large unit-scale blobs that show up as floaters
                        param.data = torch.nan_to_num(param.data, nan=-10.0, posinf=-10.0, neginf=-10.0)
                    else:
                        param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e6, neginf=-1e6)

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                # Track loss for adaptive MCMC growth
                if "loss_history" not in self.strategy_state:
                    self.strategy_state["loss_history"] = []
                self.strategy_state["loss_history"].append(loss.item())
                # Keep only last 10 values to save memory
                if len(self.strategy_state["loss_history"]) > 10:
                    self.strategy_state["loss_history"] = self.strategy_state["loss_history"][-10:]
                
                # MCMC uses lr only to scale its position noise. A stream's means LR keeps a 10%
                # floor so late seeds can move; the noise would keep 10x the usual end-of-run
                # jitter with it (visible haze), so it decays to 1% as in a dataset run.
                noise_lr = schedulers[0].get_last_lr()[0]
                if horizon is not None:
                    noise_lr *= horizon.lr_factor(step, 0.01) / horizon.lr_factor(step, means_lr_final)
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=noise_lr,
                )
            else:
                assert_never(self.cfg.strategy)

            if (cfg.lbfgs_refine and step >= getattr(cfg.strategy, "refine_stop_iter", 0)
                    and step % cfg.lbfgs_every == 0 and step < max_steps - 1):
                def _next_batch():
                    nonlocal trainloader_iter
                    try:
                        return next(trainloader_iter)
                    except StopIteration:
                        trainloader_iter = iter(trainloader)
                        return next(trainloader_iter)
                self.lbfgs_refine_round(step, _next_batch)

            # Online Difix: pseudo views from the live model replace the previous round's.
            # A round deferred while an ACE refit holds the GPU is retried every step.
            if self.stream is not None and cfg.stream_cycle and not is_final_step:
                ps = data.get("is_pseudo")
                self._epoch_draws += int((~ps).sum()) if isinstance(ps, torch.Tensor) else pixels.shape[0]
                self._stream_cycle(step)
            if (self.difix is not None and self.stream is not None and self.difix.round_pending
                    and not is_final_step):
                if step % cfg.difix_chunk_every == 0:
                    self._difix_chunk(step)
            elif (step in difix_steps or difix_deferred) and not is_final_step:
                rebuild = self._difix_round(step)
                difix_deferred = rebuild is None
                if rebuild:
                    trainloader = make_trainloader()
                    trainloader_iter = iter(trainloader)

            # [GeNVS] Online Step
            if cfg.genvs_ckpt and (step % cfg.genvs_interval == 0) and step > 0:
                self.step_genvs(step)
                # Force cleanup of any graph fragments
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Rebuild loader to see new cached data immediately
                trainloader = torch.utils.data.DataLoader(
                    self.trainset,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    persistent_workers=(num_workers > 0),
                    pin_memory=True,
                )
                trainloader_iter = iter(trainloader)

            # eval the full set; always on the final step (early stop, or eval_steps past max_steps)
            is_eval_step = step in [i - 1 for i in cfg.eval_steps] or is_final_step
            if is_eval_step:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and is_eval_step:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

            if is_final_step:
                break
        self._profile_stop()

    def _profile_tick(self, step: int):
        """Start / stop the CUDA profiler window (profile_cuda_steps > 0)."""
        cfg = self.cfg
        if cfg.profile_cuda_steps <= 0 or self.world_rank != 0:
            return
        prof = getattr(self, "_cuda_prof", None)
        if prof is None and step == cfg.profile_cuda_start:
            from torch.profiler import ProfilerActivity, profile

            torch.cuda.synchronize()
            # HYPERSPLAT_PROFILE_STACK=1: Python call stacks too (to find host-sync call sites)
            self._cuda_prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                      with_stack=os.environ.get("HYPERSPLAT_PROFILE_STACK") == "1")
            self._cuda_prof.__enter__()
            self._cuda_prof_steps = [(step, time.time())]
            print(f"\n[Profile] CUDA profiler on for {cfg.profile_cuda_steps} steps from step {step}")
            if self.stream is not None and hasattr(self.stream.sidecar, "profile"):
                # Same window in the ACE sidecar (a mapper unit per trainer step)
                self.stream.sidecar.profile(cfg.profile_cuda_steps, f"{cfg.result_dir}/profile")
        elif prof is not None:
            self._cuda_prof_steps.append((step, time.time()))
            if step >= cfg.profile_cuda_start + cfg.profile_cuda_steps:
                self._profile_stop()

    def _profile_stop(self):
        prof = getattr(self, "_cuda_prof", None)
        if prof is None:
            return
        self._cuda_prof = None
        torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        from hypersplat.stream.profiling import write_profile

        out = write_profile(prof, self._cuda_prof_steps, f"{self.cfg.result_dir}/profile", "trainer")
        print(f"[Profile] wrote {out}")

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if self.stream is not None:
            self.valset.refresh()  # held-out keyframes as of now
            if len(self.valset) == 0:
                print("[Stream] No held-out keyframes yet; skipping evaluation")
                return
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=0 if self.stream is not None else 1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            if getattr(self, "dashboard", None) is not None:
                self.dashboard.update_training(psnr=round(stats["psnr"], 2))
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.flush()

        # Online GeNVS Step
        if cfg.genvs_ckpt and (step % cfg.genvs_interval == 0) and step > 0:
            self.step_genvs(step)

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds
        if len(camtoworlds_all) > 12:
            camtoworlds_all = camtoworlds_all[5:-5]
        if len(camtoworlds_all) < 2:
            print("Not enough cameras for a trajectory; skipping")
            return
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def render_custom_pose(self, pose_file: str):
        """[Autoregressive GeNVS] Render a specific pose defined in a JSON file."""
        print(f"Rendering custom pose from {pose_file}...")
        
        with open(pose_file, 'r') as f:
            data = json.load(f)
            
        c2w = torch.tensor(data['c2w'], dtype=torch.float32, device=self.device).unsqueeze(0)
        K = torch.tensor(data['K'], dtype=torch.float32, device=self.device).unsqueeze(0)
        width = int(data['width'])
        height = int(data['height'])
        
        # Rasterize
        colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=width,
            height=height,
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
        )
        
        # Save
        colors = torch.clamp(colors, 0.0, 1.0)
        canvas = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        out_path = f"{self.render_dir}/custom_render.png"
        imageio.imwrite(out_path, canvas)
        print(f"Custom render saved to {out_path}")

    @torch.no_grad()
    def _viewer_keep(self) -> Tensor:
        """Viewer-only splat mask: drop near-transparent splats and long needles. Seen off the
        training views these render as streaks across the screen instead of surface."""
        cfg = self.cfg
        opa = torch.sigmoid(self.splats["opacities"])
        sc = torch.exp(self.splats["scales"])
        smax, smin = sc.max(-1).values, sc.min(-1).values.clamp_min(1e-9)
        keep = opa >= cfg.viewer_min_opacity
        voxel = getattr(self, "stream_voxel", None)
        if voxel:
            big = smax > cfg.viewer_max_scale_voxels * voxel
            needle = (smax / smin > cfg.viewer_max_aniso) & (smax > 2.0 * voxel)
            keep &= ~(big | needle)
        else:
            keep &= smax / smin <= cfg.viewer_max_aniso
        return keep.to(opa.dtype)

    def _walk_update(self, force: bool = False):
        """Refresh the viewer's walk bounds from the registered cameras (stream runs)."""
        walk = getattr(self, "walk", None)
        if walk is None:
            return
        st = self.stream
        key = (st.views_registered, st.map_version)
        try:
            walk.update(self.parser.camtoworlds, self.parser.scene_scale, key=key, force=force)
        except Exception as e:  # the viewer must never take training down
            print(f"[Walk] bounds update failed: {e}")

    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        """nerfview's render thread calls os._exit(1) on any exception, which would end the
        training run: a failed viewer frame is logged and shown blank instead."""
        from nerfview._renderer import InterruptRenderException
        try:
            return self._viewer_render(camera_state, render_tab_state)
        except InterruptRenderException:
            raise  # nerfview's own cancel of a stale frame
        except Exception:
            if not getattr(self, "_viewer_error_logged", False):
                import traceback
                print("[Viewer] render failed (training continues; further errors not shown):")
                traceback.print_exc()
                self._viewer_error_logged = True
            h = max(int(render_tab_state.viewer_height), 1)
            w = max(int(render_tab_state.viewer_width), 1)
            return np.zeros((h, w, 3), dtype=np.float32)

    @torch.no_grad()
    def _viewer_render(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        walk = getattr(self, "walk", None)
        near_plane = render_tab_state.near_plane
        if walk is not None:
            near_plane = walk.near_plane(near_plane)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            opacity_keep=self._viewer_keep() if render_tab_state.clean_view else None,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    def _save_object_ply(self):
        """Save only the Gaussians inside the object masks."""
        # Simple clustering: Project all means to all training views.
        # If a Gaussian falls inside the mask for >50% of views it's visible in, keep it.
        # Or simpler: accumulate 'mask_score'.
        
        with torch.no_grad():
            means = self.splats["means"]
            accum_mask_score = torch.zeros(len(means), device=self.device)
            accum_visible = torch.zeros(len(means), device=self.device)
            
            # Subsample views for speed if needed
            for i in tqdm.trange(len(self.trainset), desc="Filtering Object"):
                 data = self.trainset[i]
                 if "mask" not in data: continue
                 
                 # Prepare camera
                 # simple_trainer dataset returns K, camtoworld
                 # K is [3, 3] or [1, 3, 3] depending on batch. assuming [3, 3] from dataset[i]
                 
                 curr_K = data["K"].to(self.device).unsqueeze(0)
                 c2w = data["camtoworld"].to(self.device).unsqueeze(0)
                 W, H = data["image"].shape[1], data["image"].shape[0]
                 mask = data["mask"].to(self.device) # [1, H, W]
                 
                 # Project means
                 # world to cam
                 w2c = torch.inverse(c2w)
                 R = w2c[:, :3, :3]
                 T = w2c[:, :3, 3]
                 
                 # [N, 3] -> [1, N, 3]
                 means_c = (R @ means.unsqueeze(-1)).squeeze(-1) + T
                 
                 # Perspective
                 valid_z = means_c[:, 2] > 0.2
                 
                 # pixel coords
                 u = (means_c[:, 0] / means_c[:, 2] * curr_K[:, 0, 0]) + curr_K[:, 0, 2]
                 v = (means_c[:, 1] / means_c[:, 2] * curr_K[:, 1, 1]) + curr_K[:, 1, 2]
                 
                 u = u.long()
                 v = v.long()
                 
                 # Check bounds
                 valid_xy = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                 valid = valid_z & valid_xy
                 
                 # Indices of valid points
                 indices = torch.where(valid)[0]
                 
                 # Sample mask
                 # mask is [1, H, W]
                 # We want mask[0, v, u]
                 if len(indices) > 0:
                     vals = mask[0, v[indices], u[indices]]
                     accum_mask_score[indices] += vals
                     accum_visible[indices] += 1.0
            
            # Normalize
            scores = accum_mask_score / (accum_visible + 1e-6)
            keep_mask = scores > 0.5
            
            print(f"[SimpleTrainer] Keeping {keep_mask.sum()} / {len(means)} Gaussians for Object PLY.")
            
            # Subset splats
            new_splats = {k: v[keep_mask] for k, v in self.splats.items()}
            
            # Save
            path = f"{self.ply_dir}/object.ply"
            export_splats(
                 new_splats,
                 path,
            )
            print(f"[SimpleTrainer] Saved object PLY to {path}")
            
            # Post-Process: DQN Pruning (DEPRECATED: Use Online Elimination)
            if self.cfg.run_pruner:
                 pass

    def eliminate_floaters(self, verbose=False):
        """
        Online Floater Elimination (Context-Aware Heuristic).
        Identifies and kills Gaussians that are, relative to the model's own
        distributions (percentile scores, see trainer_schedule.floater_prune_mask):
        1. Isolated (High KNN distance)
        2. Low opacity (Ghosting)
        3. Giant/Elongated (Scale artifacts)
        At most cfg.floater_max_frac of the Gaussians are killed per call.
        """
        with torch.no_grad():
            means = self.splats["means"].detach()
            opacities = torch.sigmoid(self.splats["opacities"].detach())
            N = len(means)
            device = self.device

            # --- Isolation (KNN) ---
            # Compute on CPU using sklearn to act as a memory safeguard
            iso = None
            try:
                from sklearn.neighbors import NearestNeighbors
                means_np = means.cpu().numpy()
                # K=5 neighbors (+ self)
                nbrs = NearestNeighbors(n_neighbors=min(6, N), algorithm='auto', n_jobs=-1).fit(means_np)
                distances, _ = nbrs.kneighbors(means_np)
                # Mean distance to 5 neighbors (excluding self at index 0)
                iso = torch.tensor(distances[:, 1:].mean(axis=1), device=device, dtype=torch.float32)
            except ImportError:
                if verbose: print("[Warning] sklearn not found, skipping isolation check.")
            except Exception as e:
                # Fallback if too large for CPU memory or other error
                if verbose: print(f"[Warning] Isolation check failed: {e}")

            # scales are log-scales; the mask helper works in log space
            to_prune = floater_prune_mask(
                iso,
                opacities.reshape(-1),
                self.splats["scales"].detach(),
                max_frac=self.cfg.floater_max_frac,
            )

            n_prune = to_prune.sum().item()
            if n_prune > 0:
                if verbose:
                    print(f"[FloaterEliminator] Pruning {n_prune} / {N} Gaussians ({100*n_prune/N:.2f}%)")
                # Kill via opacity (sigmoid(-100) ~ 0); MCMC relocates dead Gaussians
                # while refinement is active, afterwards they simply stay invisible.
                self.splats["opacities"].data[to_prune] = -100.0


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if cfg.stream_source:
        import signal
        # A plain SIGTERM would skip the cleanup of the ACE sidecar and the tmpfs work dir
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=False)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        
        # [FIX] Infer SH degree from checkpoint data to avoid rasterization assertion error
        if "shN" in ckpts[0]["splats"]:
            n_sh_coeffs = ckpts[0]["splats"]["shN"].shape[1] + 1
            inferred_degree = int(math.sqrt(n_sh_coeffs) - 1)
            if runner.cfg.sh_degree != inferred_degree:
                print(f"[FIX] Setting sh_degree to {inferred_degree} based on checkpoint.")
                runner.cfg.sh_degree = inferred_degree
        
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
        
        # [Autoregressive GeNVS] Render custom pose if requested
        if cfg.render_pose_file is not None:
            runner.render_custom_pose(cfg.render_pose_file)
            return  # Exit after rendering
            
    else:
        try:
            runner.train()
        finally:
            if runner.stream is not None:
                print(f"[Stream] Final: {runner.stream.stats()}")
                if getattr(runner, "device_poses", None) is not None:
                    print(f"[Stream] Device poses: {runner.device_poses.stats()}")
                if getattr(runner, "walk", None) is not None and runner.walk.bounds is not None:
                    runner.walk.bounds.save_npz(f"{cfg.result_dir}/walk_bounds.npz")
                runner.stream.stop()
                if runner.dashboard is not None:
                    runner.dashboard.update_training(done=True)
                    runner.dashboard.stop()
        if cfg.save_object_ply:
            print("Saving object-only PLY...")
            runner._save_object_ply()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer complete. Closing server...")


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    cfg.resolve_schedule(
        refine_stop_explicit=_flag_given(
            sys.argv, "--strategy.refine-stop-iter", "--strategy.refine_stop_iter"
        ),
        growth_explicit=_flag_given(sys.argv, "--strategy.growth"),
    )

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."
        
    # [ROBUSTNESS] Handle gsplat backtrack limitation: neither the UT projection nor the
    # eval3d rasterizer passes gradients to camera poses
    if cfg.with_ut and cfg.stream_source is not None:
        # Streams drop per-image pose_opt anyway (changing image set): UT wins
        cfg.pose_opt = False
        if cfg.stream_pose_opt:
            print("[Stream] UT has no pose gradients: disabling per-keyframe pose deltas")
            cfg.stream_pose_opt = False
    if cfg.with_ut and cfg.pose_opt:
        print("\n" + "!" * 80)
        print("CONFLICT DETECTED: Unscented Transform (--with_ut) + Pose Optimization (--pose_opt)")
        print("'gsplat' v1.5.3 does not yet support the backward pass for this combination.")
        print("Action: Automatically disabling experimental flags to prioritize Pose Optimization.")
        print("!" * 80 + "\n")
        cfg.with_ut = False
        cfg.with_eval3d = False

    cli(main, cfg, verbose=True)
