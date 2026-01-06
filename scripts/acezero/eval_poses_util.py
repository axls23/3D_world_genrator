import math
import random
from collections import namedtuple
from scipy.spatial.transform import Rotation
import numpy as np
import logging

_logger = logging.getLogger(__name__)

TestEstimate = namedtuple("TestEstimate", [
    "pose_est",
    "pose_gt",
    "focal_length",
    "confidence",
    "image_file"
])


def kabsch(pts1, pts2, estimate_scale=False):
    c_pts1 = pts1 - pts1.mean(axis=0)
    c_pts2 = pts2 - pts2.mean(axis=0)

    covariance = np.matmul(c_pts1.T, c_pts2) / c_pts1.shape[0]

    U, S, VT = np.linalg.svd(covariance)

    d = np.sign(np.linalg.det(np.matmul(VT.T, U.T)))
    correction = np.eye(3)
    correction[2, 2] = d

    if estimate_scale:
        pts_var = np.mean(np.linalg.norm(c_pts2, axis=1) ** 2)
        scale_factor = pts_var / np.trace(S * correction)
    else:
        scale_factor = 1.

    R = scale_factor * np.matmul(np.matmul(VT.T, correction), U.T)
    t = pts2.mean(axis=0) - np.matmul(R, pts1.mean(axis=0))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, scale_factor


def print_hyp(hypothesis, hyp_name):
    h_translation = np.linalg.norm(hypothesis['transformation'][:3, 3])
    h_angle = np.linalg.norm(Rotation.from_matrix(hypothesis['transformation'][:3, :3]).as_rotvec()) * 180 / math.pi
    _logger.debug(f"{hyp_name}: score={hypothesis['score']}, translation={h_translation:.2f}m, "
                  f"rotation={h_angle:.1f}deg.")


def get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r):

    # h_T aligns ground truth poses with estimates poses
    poses_gt_transformed = h_T @ poses_gt

    # calculate differences in position and rotations
    translations_delta = poses_gt_transformed[:, :3, 3] - poses_est[:, :3, 3]
    rotations_delta = poses_gt_transformed[:, :3, :3] @ poses_est[:, :3, :3].transpose([0, 2, 1])

    # translation inliers
    inliers_t = np.linalg.norm(translations_delta, axis=1) < inlier_threshold_t
    # rotation inliers
    inliers_r = Rotation.from_matrix(rotations_delta).magnitude() < (inlier_threshold_r / 180 * math.pi)
    # intersection of both
    return np.logical_and(inliers_r, inliers_t)


def _filter_and_extract_poses(estimates, confidence_threshold, min_confident_estimates):
    """
    Filters estimates by confidence and validity, then extracts poses into NumPy arrays.
    """
    # Filter out estimates with invalid ground truth poses (inf or nan)
    valid_estimates = [
        e for e in estimates if not np.any(np.isinf(e.pose_gt) | np.isnan(e.pose_gt))
    ]
    # Filter by confidence threshold
    confident_estimates = [
        e for e in valid_estimates if e.confidence > confidence_threshold
    ]
    num_confident_estimates = len(confident_estimates)
    _logger.debug(f"{num_confident_estimates} estimates considered confident.")

    # Check if we have enough estimates to proceed
    if num_confident_estimates < min_confident_estimates:
        _logger.debug(f"Too few confident estimates. Aborting alignment.")
        return None, None

    # Gather estimated and ground truth poses into NumPy arrays
    poses_est = np.array([e.pose_est for e in confident_estimates])
    poses_gt = np.array([e.pose_gt for e in confident_estimates])

    return poses_gt, poses_est


def estimate_alignment_ransac(estimates,
                       confidence_threshold,
                       min_confident_estimates=10,
                       inlier_threshold_t=0.05,
                       inlier_threshold_r=5,
                       ransac_iterations=10000,
                       refinement_max_hyp=12,
                       refinement_max_it=8,
                       estimate_scale=False
                       ):
    _logger.debug("Estimate transformation with RANSAC.")

    # filter and extract poses
    poses_gt, poses_est = _filter_and_extract_poses(
        estimates, confidence_threshold, min_confident_estimates
    )

    # abort if data preparation failed
    if poses_gt is None:
        return None, 1

    num_confident_estimates = poses_gt.shape[0]

    # start robust RANSAC loop
    ransac_hypotheses = []

    for hyp_idx in range(ransac_iterations):

        # sample hypothesis
        min_sample_size = 3
        samples = random.sample(range(num_confident_estimates), min_sample_size)
        h_pts1 = poses_gt[samples, :3, 3]
        h_pts2 = poses_est[samples, :3, 3]

        h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

        # calculate inliers
        inliers = get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r)

        if inliers[samples].sum() >= 3:
            # only keep hypotheses if minimal sample is all inliers
            ransac_hypotheses.append({
                "transformation": h_T,
                "inliers": inliers,
                "score": inliers.sum(),
                "scale": h_scale
            })

    if len(ransac_hypotheses) == 0:
        _logger.debug(f"Did not fine a single valid RANSAC hypothesis, abort alignment estimation.")
        return None, 1

    # sort according to score
    ransac_hypotheses = sorted(ransac_hypotheses, key=lambda x: x['score'], reverse=True)

    for hyp_idx, hyp in enumerate(ransac_hypotheses):
        print_hyp(hyp, f"Hypothesis {hyp_idx}")

    # create shortlist of best hypotheses for refinement
    _logger.debug(f"Starting refinement of {refinement_max_hyp} best hypotheses.")
    ransac_hypotheses = ransac_hypotheses[:refinement_max_hyp]

    # refine all hypotheses in the short list
    for ref_hyp in ransac_hypotheses:

        print_hyp(ref_hyp, "Pre-Refinement")

        # refinement loop
        for ref_it in range(refinement_max_it):

            # re-solve alignment on all inliers
            h_pts1 = poses_gt[ref_hyp['inliers'], :3, 3]
            h_pts2 = poses_est[ref_hyp['inliers'], :3, 3]

            h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

            # calculate new inliers
            inliers = get_inliers(h_T, poses_gt, poses_est, inlier_threshold_t, inlier_threshold_r)

            # check whether hypothesis score improved
            refined_score = inliers.sum()

            if refined_score > ref_hyp['score']:

                ref_hyp['transformation'] = h_T
                ref_hyp['inliers'] = inliers
                ref_hyp['score'] = refined_score
                ref_hyp['scale'] = h_scale

                print_hyp(ref_hyp, f"Refinement interation {ref_it}")

            else:
                _logger.debug(f"Stopping refinement. Score did not improve: New score={refined_score}, "
                              f"Old score={ref_hyp['score']}")
                break

    # re-sort refined hypotheses
    ransac_hypotheses = sorted(ransac_hypotheses, key=lambda x: x['score'], reverse=True)

    for hyp_idx, hyp in enumerate(ransac_hypotheses):
        print_hyp(hyp, f"Hypothesis {hyp_idx}")

    return ransac_hypotheses[0]['transformation'], ransac_hypotheses[0]['scale']


def estimate_alignment_least_squares(estimates,
                                     confidence_threshold,
                                     min_confident_estimates=10,
                                     estimate_scale=False):
    _logger.debug("Estimate transformation with least squares.")

    # filter and extract poses
    poses_gt, poses_est = _filter_and_extract_poses(
        estimates, confidence_threshold, min_confident_estimates
    )

    # abort if data preparation failed
    if poses_gt is None:
        return None, 1

    # compute alignment using all confident points
    h_pts1 = poses_gt[:, :3, 3]
    h_pts2 = poses_est[:, :3, 3]
    h_T, h_scale = kabsch(h_pts1, h_pts2, estimate_scale)

    return h_T, h_scale


def compute_RPE(poses_a, poses_b):
    """Compute the relative pose error (RPE)
    Args:
        poses_a: a list of poses, shape Nx4x4
        poses_b: a list of corresponding poses, shape Nx4x4
    """
    errs_trans = []

    for i in range(len(poses_a) - 1):
        # calculate delta pose between current pose and next pose
        pose_a_delta = np.linalg.inv(poses_a[i]) @ poses_a[i+1]
        pose_b_delta = np.linalg.inv(poses_b[i]) @ poses_b[i+1]

        # compare delta poses
        pose_delta = np.linalg.inv(pose_a_delta) @ pose_b_delta

        # relative error is the length of the translation of the delta
        errs_trans.append(np.linalg.norm(pose_delta[:3, 3]))

    # return mean error
    return sum(errs_trans) / len(errs_trans)

def compute_ATE(poses_a, poses_b):
    """Compute the absolute trajectory error (ATE)
    Args:
        poses_a: a list of poses, shape Nx4x4
        poses_b: a list of corresponding poses, shape Nx4x4
    """

    # select translation component of poses
    trans_a = poses_a[:, :3, 3]
    trans_b = poses_b[:, :3, 3]

    # distances between camera centers
    errs_trans = np.linalg.norm(trans_a - trans_b, axis=1)
    # RMSE of distances
    rmse = np.linalg.norm(errs_trans) / np.sqrt(len(errs_trans))

    return rmse
