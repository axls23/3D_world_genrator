#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

import argparse
import logging
import math
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import eval_poses_util as tutil
import dataset_io

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Compute pose error metrics for an ACE pose file using (pseudo) ground truth pose files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ace_pose_file', type=Path, help='Path to an ACE pose file with one line per image.')

    parser.add_argument('gt_pose_files', type=str,
                        help="Glob pattern for pose files, e.g. 'datasets/scene/*.txt', each file is assumed to "
                             "contain a 4x4 pose matrix, cam2world, correspondence with rgb files in the ACE pose "
                             "file is assumed by alphabetical order")

    parser.add_argument('--estimate_alignment', type=str, default='ransac', choices=['ransac', 'least_squares', 'none'],
                        help='Method for estimating rigid body transformation between estimates and ground truth.')

    parser.add_argument('--estimate_alignment_scale', type=_strtobool, default=True,
                        help='Estimate similarity transformation when estimating alignment')

    parser.add_argument('--estimate_alignment_conf_threshold', type=float, default=500,
                        help='Only consider pose estimates with higher confidence when estimates the alignment.')

    parser.add_argument('--pose_error_thresh_t', type=float, default=0.05,
                        help='Pose threshold (translation) for evaluation and alignment')

    parser.add_argument('--pose_error_thresh_r', type=float, default=5,
                        help='Pose threshold (rotation) for evaluation and alignment')

    parser.add_argument('--reg_rate_conf_threshold', type=float, default=1000,
                        help='Calculate the registration rate as the ratio of estimates with confidence larger than this.')

    parser.add_argument('--results_file', type=str, default=None,
                        help='Store evaluation stats in text file, in the same folder as the ACE pose file. None: No file is created.')

    opt = parser.parse_args()

    _logger.info("Reading ACE pose file.")

    with open(opt.ace_pose_file, 'r') as f:
        ace_pose_data = f.readlines()

    # Dict mapping file name to ACE estimate
    ace_estimates = {}

    # parse pose file data
    for pose_line in ace_pose_data:
        # image info as: file_name, q_w, q_x, q_y, q_z, t_x, t_y, t_z, focal_length, confidence
        pose_tokens = pose_line.split()

        # read file name and confidence
        file_name = pose_tokens[0]
        confidence = float(pose_tokens[-1])

        # read pose
        q_wxyz = [float(t) for t in pose_tokens[1:5]]
        t_xyz = [float(t) for t in pose_tokens[5:8]]

        # quaternion to rotation matrix
        R = Rotation.from_quat(q_wxyz[1:] + [q_wxyz[0]]).as_matrix()

        # construct full pose matrix
        T_world2cam = np.eye(4)
        T_world2cam[:3, :3] = R
        T_world2cam[:3, 3] = t_xyz

        # pose files contain world-to-cam but we need cam-to-world
        T_cam2world = np.linalg.inv(T_world2cam)

        # store ACE estimate
        ace_estimates[file_name] = (T_cam2world, confidence)

    _logger.info(f"Read {len(ace_estimates)} poses from: {opt.ace_pose_file}")

    # sort ACE estimates by file names
    sorted_ace_poses = [ace_estimates[key] for key in sorted(ace_estimates.keys())]

    # load ground truth poses, sorted by file name
    sorted_gt_poses = dataset_io.load_pose_files(opt.gt_pose_files)
    # convert torch to numpy
    sorted_gt_poses = [pose.numpy() for pose in sorted_gt_poses]

    _logger.info(f"Loaded {len(sorted_gt_poses)} ground truth poses.")

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain threshold from their GT pose.
    accuracy = 0

    if opt.estimate_alignment != 'none':
        # alignment needs a list of pose correspondences with confidences
        pose_correspondences = []

        # walk through ACE estimates and GT poses in parallel
        for (ace_pose, ace_confidence), gt_pose in zip(sorted_ace_poses, sorted_gt_poses):
            pose_correspondences.append((tutil.TestEstimate(
                pose_est=ace_pose,
                pose_gt=gt_pose,
                confidence=ace_confidence,
                image_file=None,
                focal_length=None
            )))

        # calculate the alignment, the transformation aligns estimated poses to GT
        if opt.estimate_alignment == 'ransac':

            alignment_transformation, alignment_scale = tutil.estimate_alignment_ransac(
                estimates=pose_correspondences,
                confidence_threshold=opt.estimate_alignment_conf_threshold,
                estimate_scale=opt.estimate_alignment_scale,
                inlier_threshold_r=opt.pose_error_thresh_r,
                inlier_threshold_t=opt.pose_error_thresh_t,
            )

        elif opt.estimate_alignment == 'least_squares':

            # calculate the alignment, the transformation aligns estimated poses to GT
            alignment_transformation, alignment_scale = tutil.estimate_alignment_least_squares(
                estimates=pose_correspondences,
                confidence_threshold=opt.estimate_alignment_conf_threshold,
                estimate_scale=opt.estimate_alignment_scale
            )
        else:
            raise ValueError(f"Unknown alignment method {opt.estimate_alignment}")

        if alignment_transformation is None:
            _logger.info(f"Alignment requested but failed. Setting all pose errors to {math.inf}.")

    else:
        alignment_transformation = np.eye(4)
        alignment_scale = 1.

    # Evaluation Loop
    gt_poses = []
    pred_poses_aligned = []

    for (ace_pose, ace_confidence), gt_pose in zip(sorted_ace_poses, sorted_gt_poses):

        # Check if the GT pose is valid.
        if np.isinf(gt_pose).any() or np.isnan(gt_pose).any():
            _logger.info(f"Skipping invalid GT pose.")
            continue

        if alignment_transformation is not None:
            # Apply alignment transformation to predicted pose (alignment is gt to predicted, so we need the inverse)
            alignment_transformation_inv = np.linalg.inv(alignment_transformation)
            ace_pose = alignment_transformation_inv @ ace_pose
            # Scale correction to rotation component (to make it orthonormal again):
            # Alignment is a similarity transform which includes scaling.
            # This cannot be applied to rotations without correction.
            ace_pose[:3, :3] *= alignment_scale

            # Store for ATE and RPE computation
            gt_poses.append(gt_pose)
            pred_poses_aligned.append(ace_pose)

            # Calculate translation error.
            t_err = float(np.linalg.norm(gt_pose[0:3, 3] - ace_pose[0:3, 3]))

            # Rotation error.
            gt_R = gt_pose[0:3, 0:3]
            out_R = ace_pose[0:3, 0:3]

            r_err = np.matmul(out_R, np.transpose(gt_R))
            # Compute angle-axis representation.
            r_err = cv2.Rodrigues(r_err)[0]
            # Extract the angle.
            r_err = np.linalg.norm(r_err) * 180 / math.pi
        else:
            pose_gt = None
            t_err, r_err = math.inf, math.inf

        _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

        # Save the errors.
        rErrs.append(r_err)
        tErrs.append(t_err * 100) # in cm

        # Check various thresholds.
        if r_err < opt.pose_error_thresh_r and t_err < opt.pose_error_thresh_t:
            accuracy += 1

    total_frames = len(rErrs)

    gt_poses = np.array(gt_poses)
    pred_poses_aligned = np.array(pred_poses_aligned)

    # Compute ATE and RPE.
    ate = tutil.compute_ATE(gt_poses, pred_poses_aligned) * 100 # in cm
    rpe = tutil.compute_RPE(gt_poses, pred_poses_aligned) * 100 # in cm

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    # Compute final precision.
    accuracy = accuracy / total_frames * 100 # in %

    # Compute registration rate
    reg_rate = sum([conf >= opt.reg_rate_conf_threshold for _, conf in sorted_ace_poses]) / len(sorted_ace_poses) * 100 # in %

    _logger.info("===================================================")

    _logger.info(f"Registration Rate (@{opt.reg_rate_conf_threshold} conf): {reg_rate:.1f}%")
    _logger.info(f'Accuracy (@{opt.pose_error_thresh_t}m/{opt.pose_error_thresh_r}°): {accuracy:.1f}%')
    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")

    if opt.estimate_alignment != 'least_squares':
        _logger.warning(f"ATE/RPE stats have been computed with alignment: {opt.estimate_alignment}, usually: least_squares.")
    _logger.info(f"ATE: {ate:.1f}cm, RPE: {rpe:.1f}cm")

    if opt.results_file is not None:
        # Save results to txt.
        with open(opt.ace_pose_file.parent / opt.results_file, 'w') as f:
            f.write(f'RegistrationRate(@{opt.reg_rate_conf_threshold}conf) Accuracy(@{opt.pose_error_thresh_t}m/{opt.pose_error_thresh_r}°) MedianRot(°) MedianTrans(cm) ATE(cm) RPE(cm)\n')
            f.write(f'{reg_rate:.2f} {accuracy:.2f} {median_rErr:.2f} {median_tErr:.2f} {ate:.2f} {rpe:.2f}\n')
        _logger.info(f"Results written to {opt.ace_pose_file.parent / opt.results_file}")

    _logger.info("Test complete.")