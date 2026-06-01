#!/usr/bin/env python3
"""
ace_zero_hybrid.py — Single-Pass Hybrid ACE-Zero Orchestrator

Replaces the iterative seed→register→map loop of ace_zero.py with:
  1. InitAllPoses  → init_all_poses.py (SIFT + Essential Matrix, ~10s)
  2. Single Train  → train_ace.py with all frames + pose refinement
  3. Single Register → register_mapping.py (final output)

Expected speedup: 16min → ~3min for 61 frames.

Usage:
  python ace_zero_hybrid.py "images/*.jpg" output_dir/ [--hybrid_train_iterations 15000]
"""

import logging
import shutil
import os
import time
import argparse
from pathlib import Path
from distutils.util import strtobool

import numpy as np
import ace_zero_util as zutil
import dataset_io

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Run Hybrid ACE-Zero (single-pass) for a dataset or scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('rgb_files', type=str,
                        help="Glob pattern for RGB files, e.g. 'datasets/scene/*.jpg'")
    parser.add_argument('results_folder', type=Path,
                        help='Path to output folder for result files')

    # === Hybrid-specific parameters ===
    parser.add_argument('--hybrid_train_iterations', type=int, default=15000,
                        help="Training iterations for the single ACE pass.")
    parser.add_argument('--hybrid_pose_wait', type=int, default=2000,
                        help="Freeze poses for first N iterations to let SCR stabilize.")
    parser.add_argument('--hybrid_init_confidence', type=float, default=10000,
                        help="Confidence assigned to all initialized poses.")
    parser.add_argument('--hybrid_max_features', type=int, default=2000,
                        help="Maximum SIFT features per image for initialization.")

    # === Pose refinement ===
    parser.add_argument('--refinement', type=str, default="mlp",
                        choices=['mlp', 'none', 'naive'],
                        help="How to refine poses.")
    parser.add_argument('--refinement_ortho', type=str, default="gram-schmidt",
                        choices=['gram-schmidt', 'procrustes'],
                        help="How to orthonormalize rotations.")
    parser.add_argument('--pose_refinement_lr', type=float, default=0.001,
                        help="Learning rate for pose refinement.")

    # === Calibration ===
    parser.add_argument('--refine_calibration', type=_strtobool, default=True,
                        help="Optimize focal length during mapping.")
    parser.add_argument('--use_external_focal_length', type=float, default=-1,
                        help="Provide the focal length. -1: Use 70%% of image diagonal.")

    # === ACE training parameters ===
    parser.add_argument('--image_resolution', type=int, default=480,
                        help='Base image resolution')
    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='Depth of the regression head')
    parser.add_argument('--max_dataset_passes', type=int, default=10,
                        help='Max number of repetition of mapping images')
    parser.add_argument('--repro_loss_type', type=str, default="tanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error')
    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='Hard clamping threshold')
    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='Soft clamping threshold')
    parser.add_argument('--learning_rate_max', type=float, default=0.003,
                        help="Max learning rate")
    parser.add_argument('--learning_rate_schedule', type=str, default="1cyclepoly",
                        choices=["circle", "constant", "1cyclepoly"],
                        help='LR schedule')
    parser.add_argument('--cooldown_iterations', type=int, default=5000,
                        help="Cooldown iterations after early stopping criterion")
    parser.add_argument('--cooldown_threshold', type=float, default=0.7,
                        help="Start cooldown after this %% of batch pixels are inliers")
    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='Max inplane rotation angle')
    parser.add_argument('--num_data_workers', type=int, default=12,
                        help='Number of data loading workers')
    parser.add_argument('--training_buffer_cpu', type=_strtobool, default=False,
                        help='Store training buffer on CPU')

    # === Registration parameters ===
    parser.add_argument('--registration_confidence', type=int, default=500,
                        help="Consider image registered if it has this many inlier scene coords.")
    parser.add_argument('--ransac_iterations', type=int, default=32,
                        help="RANSAC hypothesis count for registration.")
    parser.add_argument('--ransac_threshold', type=float, default=10,
                        help='RANSAC inlier threshold in pixels')

    # === Probabilistic loss parameters ===
    parser.add_argument("--loss_structure", type=str, default="dsac*",
                        choices=["dsac*", "probabilistic"],
                        help='General structure of the loss')
    parser.add_argument('--prior_loss_type', type=str, default="none",
                        choices=["none", "rgbd_laplace_nll", "laplace_nll", "laplace_wd", "diffusion"])
    parser.add_argument('--prior_loss_weight', type=float, default=1)
    parser.add_argument('--prior_loss_bandwidth', type=float, default=0.1)
    parser.add_argument('--prior_loss_location', type=float, default=1)
    parser.add_argument('--prior_diffusion_model_path', type=Path, default=None)
    parser.add_argument('--prior_diffusion_start_step', type=int, default=0)
    parser.add_argument('--prior_diffusion_warmup_steps', type=int, default=-1)
    parser.add_argument('--prior_diffusion_subsample', type=float, default=1.0)

    # === Visualization ===
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help="Render visualization frames.")
    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False)
    parser.add_argument('--render_marker_size', type=float, default=0.03)
    parser.add_argument('--iterations_output', type=int, default=500,
                        help='Print loss every N iterations')
    parser.add_argument('--render_depth_hist', type=_strtobool, default=False)

    # === Export ===
    parser.add_argument('--export_point_cloud', type=_strtobool, default=False,
                        help="Export ACE point cloud after reconstruction.")
    parser.add_argument('--dense_point_cloud', type=_strtobool, default=False)

    parser.add_argument('--random_seed', type=int, default=1305)

    opt = parser.parse_args()

    # Create output directory
    os.makedirs(opt.results_folder, exist_ok=True)

    _logger.info(f"=== HYBRID ACE-ZERO: Single-Pass Mode ===")
    _logger.info(f"Input: {opt.rgb_files}")
    _logger.info(f"Output: {opt.results_folder}")
    _logger.info(f"Train iterations: {opt.hybrid_train_iterations}")
    _logger.info(f"Pose refinement: {opt.refinement} (wait={opt.hybrid_pose_wait})")

    reconstruction_start_time = time.time()

    # ==========================================
    # PHASE 1: Initialize all poses at once
    # ==========================================
    _logger.info("=" * 60)
    _logger.info("PHASE 1: Fast Global Pose Initialization (SIFT + Essential Matrix)")
    _logger.info("=" * 60)

    init_pose_file = opt.results_folder / "poses_init.txt"

    # Use the focal length hint if provided, otherwise -1 triggers the 70% diagonal heuristic
    init_focal = opt.use_external_focal_length if opt.use_external_focal_length > 0 else -1

    init_cmd = [
        *zutil.TRAINING_EXE[:1],  # python executable
        "init_all_poses.py",
        opt.rgb_files,
        str(init_pose_file),
        "--focal_length", str(init_focal),
        "--confidence", str(opt.hybrid_init_confidence),
        "--max_features", str(opt.hybrid_max_features),
    ]

    phase1_start = time.time()
    zutil.run_cmd(init_cmd)
    phase1_time = time.time() - phase1_start

    if not init_pose_file.exists():
        _logger.error("Phase 1 FAILED: No init pose file generated.")
        exit(1)

    # Count initialized poses
    with open(init_pose_file, 'r') as f:
        num_init_poses = len(f.readlines())
    _logger.info(f"Phase 1 complete: {num_init_poses} poses initialized in {phase1_time:.1f}s")

    # ==========================================
    # PHASE 2: Single ACE Training Pass
    # ==========================================
    _logger.info("=" * 60)
    _logger.info("PHASE 2: Single ACE Training Pass (all frames, joint optimization)")
    _logger.info("=" * 60)

    iteration_id = "hybrid_pass"
    phase2_start = time.time()

    # Build the mapping command using the common utility
    mapping_cmd = [
        *zutil.TRAINING_EXE,
        opt.rgb_files,
        opt.results_folder / f"{iteration_id}.pt",
    ]

    # Core training parameters
    mapping_cmd += [
        "--iterations", str(opt.hybrid_train_iterations),
        "--use_ace_pose_file", str(init_pose_file),
        "--ace_pose_file_conf_threshold", str(opt.registration_confidence),
        "--pose_refinement", opt.refinement,
        "--pose_refinement_wait", str(opt.hybrid_pose_wait),
        "--pose_refinement_lr", str(opt.pose_refinement_lr),
        "--refinement_ortho", opt.refinement_ortho,
        "--refine_calibration", str(opt.refine_calibration),
        "--num_data_workers", str(opt.num_data_workers),
        "--training_buffer_cpu", str(opt.training_buffer_cpu),
        "--num_head_blocks", str(opt.num_head_blocks),
        "--image_resolution", str(opt.image_resolution),
        "--max_dataset_passes", str(opt.max_dataset_passes),
        "--repro_loss_type", opt.repro_loss_type,
        "--repro_loss_hard_clamp", str(opt.repro_loss_hard_clamp),
        "--repro_loss_soft_clamp", str(opt.repro_loss_soft_clamp),
        "--learning_rate_max", str(opt.learning_rate_max),
        "--learning_rate_schedule", opt.learning_rate_schedule,
        "--learning_rate_cooldown_iterations", str(opt.cooldown_iterations),
        "--learning_rate_cooldown_trigger_percent_threshold", str(opt.cooldown_threshold),
        "--aug_rotation", str(opt.aug_rotation),
        "--render_visualization", str(opt.render_visualization),
        "--render_target_path", str(zutil.get_render_path(opt.results_folder)),
        "--render_marker_size", str(opt.render_marker_size),
        "--render_flipped_portrait", str(opt.render_flipped_portrait),
        "--iterations_output", str(opt.iterations_output),
        "--render_depth_hist", str(opt.render_depth_hist),
    ]

    # Loss structure
    mapping_cmd += [
        "--loss_structure", opt.loss_structure,
        "--prior_loss_type", opt.prior_loss_type,
        "--prior_loss_weight", str(opt.prior_loss_weight),
        "--prior_loss_bandwidth", str(opt.prior_loss_bandwidth),
        "--prior_loss_location", str(opt.prior_loss_location),
        "--prior_diffusion_start_step", str(opt.prior_diffusion_start_step),
        "--prior_diffusion_warmup_steps", str(opt.prior_diffusion_warmup_steps),
        "--prior_diffusion_subsample", str(opt.prior_diffusion_subsample),
    ]

    if opt.prior_diffusion_model_path is not None:
        mapping_cmd += ["--prior_diffusion_model_path", str(opt.prior_diffusion_model_path)]

    zutil.run_cmd(mapping_cmd)

    phase2_time = time.time() - phase2_start
    _logger.info(f"Phase 2 complete: Training done in {phase2_time / 60:.1f} minutes")

    # ==========================================
    # PHASE 3: Final Registration
    # ==========================================
    _logger.info("=" * 60)
    _logger.info("PHASE 3: Final Registration (all frames against trained map)")
    _logger.info("=" * 60)

    phase3_start = time.time()

    # Get the focal length from the preliminary pose file produced by training
    preliminary_pose_file = opt.results_folder / f"poses_{iteration_id}_preliminary.txt"

    if preliminary_pose_file.exists():
        _, _, focal_lengths, _ = dataset_io.load_dataset_ace(
            pose_file=preliminary_pose_file,
            confidence_threshold=opt.registration_confidence)
        if focal_lengths and len(focal_lengths) > 0:
            est_focal = focal_lengths[0]
            _logger.info(f"Using estimated focal length from training: {est_focal:.1f}")
        else:
            est_focal = opt.use_external_focal_length
    else:
        est_focal = opt.use_external_focal_length
        _logger.warning("No preliminary pose file found, using external focal length.")

    reg_cmd = [
        *zutil.REGISTER_EXE,
        opt.rgb_files,
        opt.results_folder / f"{iteration_id}.pt",
        "--render_visualization", str(opt.render_visualization),
        "--render_target_path", str(zutil.get_render_path(opt.results_folder)),
        "--render_marker_size", str(opt.render_marker_size),
        "--render_flipped_portrait", str(opt.render_flipped_portrait),
        "--session", iteration_id,
        "--confidence_threshold", str(opt.registration_confidence),
        "--image_resolution", str(opt.image_resolution),
        "--hypotheses", str(opt.ransac_iterations),
        "--threshold", str(opt.ransac_threshold),
        "--num_data_workers", str(opt.num_data_workers),
        "--hypotheses_max_tries", "16",
    ]

    if est_focal and est_focal > 0:
        reg_cmd += ["--use_external_focal_length", str(est_focal)]

    zutil.run_cmd(reg_cmd)

    phase3_time = time.time() - phase3_start
    _logger.info(f"Phase 3 complete: Registration done in {phase3_time:.1f}s")

    # ==========================================
    # PHASE 4: Finalize & Report
    # ==========================================
    reconstruction_end_time = time.time()
    reconstruction_time = reconstruction_end_time - reconstruction_start_time

    # Copy final pose file
    final_pose_file = opt.results_folder / f"poses_{iteration_id}.txt"
    if final_pose_file.exists():
        shutil.copy(final_pose_file, opt.results_folder / "poses_final.txt")
    else:
        _logger.error(f"Final pose file not found: {final_pose_file}")
        exit(1)

    # Check registration rates
    registration_rates = zutil.get_registration_rates(
        pose_file=opt.results_folder / "poses_final.txt",
        thresholds=[500, 1000, 2000, 4000])

    # Report
    _logger.info("=" * 60)
    _logger.info("HYBRID ACE-ZERO COMPLETE")
    _logger.info("=" * 60)

    stats_report = (
        f"Total Time: {reconstruction_time / 60:.1f} min\n"
        f"  Phase 1 (Init):     {phase1_time:.1f}s\n"
        f"  Phase 2 (Train):    {phase2_time / 60:.1f} min\n"
        f"  Phase 3 (Register): {phase3_time:.1f}s\n"
        f"Registration Rates:\n"
        f"  @500:  {registration_rates[0] * 100:.1f}%\n"
        f"  @1000: {registration_rates[1] * 100:.1f}%\n"
        f"  @2000: {registration_rates[2] * 100:.1f}%\n"
        f"  @4000: {registration_rates[3] * 100:.1f}%"
    )
    _logger.info(stats_report)

    # Export point cloud if requested
    if opt.export_point_cloud:
        _logger.info("Exporting point cloud...")
        zutil.run_cmd([
            "./export_point_cloud.py",
            opt.results_folder / "pc_final.ply",
            "--network", opt.results_folder / f"{iteration_id}.pt",
            "--pose_file", opt.results_folder / "poses_final.txt",
            "--convention", "opencv",
            "--dense_point_cloud", str(opt.dense_point_cloud),
        ])
