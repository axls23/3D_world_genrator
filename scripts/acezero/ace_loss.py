import torch
from ace_util import to_homogeneous


class AceLoss:
    """
    Encapsulates the main ACE loss calculation, supporting different structures.

    This class orchestrates the overall loss computation, switching between different
    strategies like 'dsac*' and 'probabilistic'.

    Supported loss structures:
    - dsac*:         A robust loss function that separates predictions into 'valid' and
                     'invalid' sets. It applies a reprojection loss to the valid set
                     and a proxy loss (based on distance to a target plane or ground
                     truth coordinates) to the invalid set. Can optionally include
                     diffusion priors.
    - probabilistic: A simpler structure that combines a reprojection loss with a
                     probabilistic prior loss on depth.
    """

    def __init__(self,
                 loss_structure,
                 repro_loss,
                 prior_loss,
                 prior_loss_weight,
                 use_depth,
                 depth_min,
                 depth_max,
                 repro_loss_hard_clamp,
                 learning_rate_cooldown_trigger_px_threshold,
                 depth_target):
        """
        Initializes the AceLoss class.

        Args:
            loss_structure (str): The loss structure to use ('dsac*' or 'probabilistic').
            repro_loss: An initialized ReproLoss object.
            prior_loss: An initialized PriorLoss object.
            prior_loss_weight (float): The weight to apply to the prior loss.
            use_depth (bool): Flag indicating if ground truth depth is available.
            depth_min (float): Minimum valid depth for the 'dsac*' loss.
            depth_max (float): Maximum valid depth for the 'dsac*' loss.
            repro_loss_hard_clamp (float): Hard clamp for reprojection error in 'dsac*'.
            learning_rate_cooldown_trigger_px_threshold (float): Inlier threshold for early stopping.
            depth_target (float): Target depth for proxy coordinates in 'dsac*'.
        """

        if loss_structure == "dsac*" and prior_loss.type not in ["none", "diffusion"]:
            raise ValueError("Only diffusion prior_loss_type is supported with the DSAC* style loss.")

        self.loss_structure = loss_structure
        self.repro_loss = repro_loss
        self.prior_loss = prior_loss
        self.prior_loss_weight = prior_loss_weight
        self.use_depth = use_depth
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.repro_loss_hard_clamp = repro_loss_hard_clamp
        self.learning_rate_cooldown_trigger_px_threshold = learning_rate_cooldown_trigger_px_threshold
        self.depth_target = depth_target

    def compute(self,
                pred_cam_coords_b31,
                pred_scene_coords_b31,
                reprojection_error_b1,
                target_crds_b3,
                target_px_b2,
                invKs_b33,
                iteration,
                max_iterations,):
        """
        Computes the main loss based on the configured structure.

        Args:
            pred_cam_coords_b31: Predicted camera coordinates.
            pred_scene_coords_b31: Predicted scene coordinates.
            reprojection_error_b1: Per-pixel reprojection error.
            target_crds_b3: Ground truth scene coordinates.
            target_px_b2: Target pixel coordinates.
            invKs_b33: Inverse camera intrinsics.
            iteration (int): The current training iteration.
            max_iterations (int): The maximum number of training iterations currently scheduled.

        Returns:
            tuple[float, torch.Tensor]: A tuple containing the batch inlier ratio (float)
                                        and the final computed loss (scalar tensor).
        """
        if self.loss_structure == "dsac*":
            return self._loss_dsacstar(pred_cam_coords_b31, pred_scene_coords_b31,
                                       reprojection_error_b1, target_crds_b3, target_px_b2, invKs_b33, iteration, max_iterations)
        elif self.loss_structure == "probabilistic":
            batch_size = pred_cam_coords_b31.shape[0]
            if batch_size == 0:
                return 0.0, torch.tensor(0.0, device=pred_cam_coords_b31.device)

            # batch inliers for early stopping
            batch_inliers = reprojection_error_b1 < self.learning_rate_cooldown_trigger_px_threshold
            batch_inliers = float(batch_inliers.sum()) / batch_size

            # calculate reprojection loss
            loss_repro_b1 = self.repro_loss.compute(reprojection_error_b1, iteration)
            loss_repro = loss_repro_b1.sum() / batch_size

            # calculate prior loss
            loss_prior = self.prior_loss.compute(pred_cam_coords_b31, pred_scene_coords_b31, reprojection_error_b1,
                                                 iteration, max_iterations, target_crds_b3, self.use_depth)

            # combine reprojection loss and prior
            loss = loss_repro + self.prior_loss_weight * loss_prior
            return batch_inliers, loss
        else:
            raise ValueError(f"loss_structure {self.loss_structure} is not supported.")

    def _loss_dsacstar(self,
                       pred_cam_coords_b31,
                       pred_scene_coords_b31,
                       reprojection_error_b1,
                       target_crds_b3,
                       target_px_b2,
                       invKs_b33,
                       iteration,
                       max_iterations=None):
        """
        Computes the DSAC* style loss.
        
        If a diffusion prior is configured, it will be added to the loss
        and weighted by the prior_loss_weight.
        """
        batch_size = pred_cam_coords_b31.shape[0]
        if batch_size == 0:
            return 0.0, torch.tensor(0.0, device=pred_cam_coords_b31.device)

        # Compute masks used to ignore invalid pixels.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.depth_min
        invalid_repro_b1 = reprojection_error_b1 > self.repro_loss_hard_clamp
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.depth_max
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)

        if self.use_depth:
            invalid_target_crds_b1 = (
                        torch.linalg.norm(target_crds_b3 - pred_scene_coords_b31.squeeze(), dim=1) > 0.1).unsqueeze(1)
            target_crds_available_b1 = (target_crds_b3.abs().sum(dim=1) > 0.00001).unsqueeze(1)
            invalid_mask_b1 = invalid_mask_b1 | (invalid_target_crds_b1 & target_crds_available_b1)

        valid_mask_b1 = ~invalid_mask_b1

        if valid_mask_b1.sum() > 0:
            valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
            loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, iteration)
            batch_inliers = valid_reprojection_error_b1 < self.learning_rate_cooldown_trigger_px_threshold
            batch_inliers = float(batch_inliers.sum()) / batch_size
        else:
            loss_valid = 0
            batch_inliers = 0

        if not self.use_depth:
            pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
            target_camera_coords_b31 = self.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)
            invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
            loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(
                invalid_mask_b11).sum()
        else:
            if invalid_mask_b1.sum() > 0:
                target_crds_available_b1 = (target_crds_b3.abs().sum(dim=1) > 0.00001).unsqueeze(1)
                invalid_mask_b1 = invalid_mask_b1 & target_crds_available_b1
                loss_invalid = torch.linalg.norm(target_crds_b3 - pred_scene_coords_b31.squeeze(), dim=1)
                loss_invalid = loss_invalid[invalid_mask_b1.squeeze()].sum()
            else:
                loss_invalid = 0

        loss = (loss_valid + loss_invalid) / batch_size
        
        # Add diffusion prior if enabled
        if self.prior_loss.type == "diffusion":
            # Calculate diffusion prior loss (assumes it's already normalized by batch size)
            loss_prior = self.prior_loss.compute(pred_cam_coords_b31, pred_scene_coords_b31, reprojection_error_b1,
                                                 iteration, max_iterations, target_crds_b3, self.use_depth)
            # Add weighted diffusion prior to the dsac* loss
            loss = loss + self.prior_loss_weight * loss_prior
        
        return batch_inliers, loss

