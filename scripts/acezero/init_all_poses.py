#!/usr/bin/env python3
"""
init_all_poses.py — Fast All-Frame Pose Initializer for Hybrid ACE-Zero

Generates initial camera pose estimates for ALL input frames simultaneously
using classical computer vision (SIFT + Essential Matrix), bypassing the
iterative seed→register→map bootstrapping loop of standard ACE-Zero.

Algorithm:
  1. Extract SIFT features from all N frames
  2. Match consecutive frame pairs (i, i+1) via BFMatcher + Lowe's ratio test
  3. Compute essential matrix E for each pair (RANSAC)
  4. Decompose E → relative R, t (translation up-to-scale)
  5. Chain relative poses into global poses: T_i = T_{i-1} @ dT_{i-1→i}
  6. Write all poses in ACE pose file format

Output format (per line):
  image_path qw qx qy qz tx ty tz focal_length confidence

Usage:
  python init_all_poses.py "images/*.jpg" output_poses.txt [--focal_length 500]
"""

import argparse
import glob
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_logger = logging.getLogger(__name__)


def extract_sift_features(image_path: str, max_features: int = 2000):
    """Extract SIFT keypoints and descriptors from an image.

    Args:
        image_path: Path to the image file.
        max_features: Maximum number of features to retain.

    Returns:
        Tuple of (keypoints, descriptors) or (None, None) on failure.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        _logger.warning(f"Cannot read image: {image_path}")
        return None, None

    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def match_features(desc_a, desc_b, ratio_threshold: float = 0.75):
    """Match SIFT descriptors using BFMatcher with Lowe's ratio test.

    Args:
        desc_a: Descriptors from image A.
        desc_b: Descriptors from image B.
        ratio_threshold: Lowe's ratio test threshold.

    Returns:
        List of good DMatch objects.
    """
    if desc_a is None or desc_b is None:
        return []
    if len(desc_a) < 8 or len(desc_b) < 8:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    try:
        raw_matches = bf.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return []

    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def estimate_relative_pose(kp_a, kp_b, matches, focal_length: float, pp: tuple):
    """Estimate relative pose between two frames using the essential matrix.

    Args:
        kp_a: Keypoints from image A.
        kp_b: Keypoints from image B.
        matches: Good feature matches.
        focal_length: Estimated focal length in pixels.
        pp: Principal point (cx, cy).

    Returns:
        Tuple of (R, t, num_inliers) or (None, None, 0) on failure.
    """
    if len(matches) < 12:
        _logger.warning(f"Too few matches ({len(matches)}) for essential matrix estimation.")
        return None, None, 0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

    # Compute essential matrix with RANSAC
    E, mask = cv2.findEssentialMat(
        pts_a, pts_b,
        focal=focal_length,
        pp=pp,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None or mask is None:
        return None, None, 0

    num_inliers = int(mask.sum())

    # Recover rotation and translation from E
    _, R, t, pose_mask = cv2.recoverPose(E, pts_a, pts_b, focal=focal_length, pp=pp, mask=mask)

    return R, t.flatten(), num_inliers


def chain_poses(relative_poses: list):
    """Chain relative poses into global (world-to-camera) poses.

    The first camera is placed at the origin (identity pose).
    Each subsequent camera pose is computed as: T_i = T_{i-1} @ dT_{i-1→i}

    Args:
        relative_poses: List of (R, t) tuples for consecutive pairs.
                        R is 3x3 rotation, t is 3-vector translation.

    Returns:
        List of 4x4 world-to-camera pose matrices.
    """
    global_poses = [np.eye(4)]  # First camera at origin

    for R, t in relative_poses:
        if R is None or t is None:
            # If relative pose estimation failed, propagate the previous pose
            # (this frame will have the same pose as the previous one)
            _logger.warning("Failed relative pose — copying previous camera pose.")
            global_poses.append(global_poses[-1].copy())
            continue

        # Build relative transform matrix
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t

        # Chain: global_i = global_{i-1} @ dT
        T_global = global_poses[-1] @ dT
        global_poses.append(T_global)

    return global_poses


def write_ace_pose_file(output_path: str, image_files: list, poses_w2c: list,
                        focal_length: float, confidence: float):
    """Write poses in ACE pose file format.

    Format per line:
      image_path qw qx qy qz tx ty tz focal_length confidence

    Poses are stored as world-to-camera.

    Args:
        output_path: Path to the output pose file.
        image_files: List of image file paths.
        poses_w2c: List of 4x4 world-to-camera numpy matrices.
        focal_length: Focal length in pixels.
        confidence: Confidence value to assign to all poses.
    """
    with open(output_path, 'w') as f:
        for img_file, pose in zip(image_files, poses_w2c):
            R_33 = pose[:3, :3]
            t_xyz = pose[:3, 3]

            # Convert rotation matrix to quaternion (scipy uses xyzw, ACE uses wxyz)
            q_xyzw = Rotation.from_matrix(R_33).as_quat()
            qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]

            line = (f"{Path(img_file)} "
                    f"{qw} {qx} {qy} {qz} "
                    f"{t_xyz[0]} {t_xyz[1]} {t_xyz[2]} "
                    f"{focal_length} {confidence}\n")
            f.write(line)

    _logger.info(f"Wrote {len(image_files)} poses to {output_path}")


def init_all_poses(images_glob: str, output_path: str, focal_length: float = -1,
                   confidence: float = 10000, max_features: int = 2000):
    """Main entry point: initialize all camera poses from sequential frames.

    Args:
        images_glob: Glob pattern matching input images.
        output_path: Path for the output ACE pose file.
        focal_length: Focal length in pixels. -1 = use 70% of image diagonal (ACE heuristic).
        confidence: Confidence value assigned to all initialized poses.
        max_features: Maximum SIFT features per image.

    Returns:
        Number of successfully initialized poses.
    """
    start_time = time.time()

    # Resolve image files
    image_files = sorted(glob.glob(images_glob))
    if len(image_files) < 2:
        _logger.error(f"Need at least 2 images, found {len(image_files)}")
        return 0

    _logger.info(f"Initializing poses for {len(image_files)} frames...")

    # Determine focal length and principal point from first image
    sample_img = cv2.imread(image_files[0])
    if sample_img is None:
        _logger.error(f"Cannot read sample image: {image_files[0]}")
        return 0

    h, w = sample_img.shape[:2]
    pp = (w / 2.0, h / 2.0)

    if focal_length < 0:
        # ACE-Zero heuristic: 70% of image diagonal
        diag = np.sqrt(w ** 2 + h ** 2)
        focal_length = 0.7 * diag
        _logger.info(f"Using heuristic focal length: {focal_length:.1f}px "
                     f"(70% of {diag:.0f}px diagonal)")

    # Step 1: Extract features from all frames
    _logger.info("Step 1/3: Extracting SIFT features...")
    all_keypoints = []
    all_descriptors = []

    for img_file in image_files:
        kp, desc = extract_sift_features(img_file, max_features=max_features)
        all_keypoints.append(kp)
        all_descriptors.append(desc)

    feature_time = time.time() - start_time
    _logger.info(f"Feature extraction: {feature_time:.1f}s")

    # Step 2: Match consecutive pairs and estimate relative poses
    _logger.info("Step 2/3: Matching consecutive pairs & estimating relative poses...")
    relative_poses = []
    total_inliers = 0
    failed_pairs = 0

    for i in range(len(image_files) - 1):
        matches = match_features(all_descriptors[i], all_descriptors[i + 1])

        R, t, num_inliers = estimate_relative_pose(
            all_keypoints[i], all_keypoints[i + 1],
            matches, focal_length, pp
        )

        if R is not None:
            # Normalize translation to unit length (scale-agnostic chaining)
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-8:
                t = t / t_norm
            relative_poses.append((R, t))
            total_inliers += num_inliers
        else:
            relative_poses.append((None, None))
            failed_pairs += 1

    match_time = time.time() - start_time - feature_time
    _logger.info(f"Matching: {match_time:.1f}s, "
                 f"avg inliers: {total_inliers / max(1, len(image_files) - 1 - failed_pairs):.0f}, "
                 f"failed pairs: {failed_pairs}/{len(image_files) - 1}")

    # Step 3: Chain into global poses
    _logger.info("Step 3/3: Chaining into global poses...")
    global_poses_w2c = chain_poses(relative_poses)

    # Write output
    write_ace_pose_file(output_path, image_files, global_poses_w2c, focal_length, confidence)

    total_time = time.time() - start_time
    _logger.info(f"All {len(image_files)} poses initialized in {total_time:.1f}s")

    return len(image_files)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Initialize all camera poses from sequential frames using SIFT + Essential Matrix.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('rgb_files', type=str,
                        help="Glob pattern for RGB files, e.g. 'images/*.jpg'")
    parser.add_argument('output_pose_file', type=str,
                        help='Output ACE pose file path')
    parser.add_argument('--focal_length', type=float, default=-1,
                        help='Focal length in pixels. -1 = use 70%% of image diagonal')
    parser.add_argument('--confidence', type=float, default=10000,
                        help='Confidence value assigned to all initialized poses')
    parser.add_argument('--max_features', type=int, default=2000,
                        help='Maximum SIFT features per image')

    args = parser.parse_args()

    num_poses = init_all_poses(
        images_glob=args.rgb_files,
        output_path=args.output_pose_file,
        focal_length=args.focal_length,
        confidence=args.confidence,
        max_features=args.max_features
    )

    if num_poses > 0:
        _logger.info(f"SUCCESS: {num_poses} poses initialized.")
    else:
        _logger.error("FAILED: No poses initialized.")
        exit(1)
