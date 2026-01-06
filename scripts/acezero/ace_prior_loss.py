# Copyright © Niantic, Inc. 2025.

import torch
import torch.distributions as distributions

class PriorLoss:
    """
    Computes a prior loss on predicted depth values using different configurable approaches.

    This class is designed to apply a regularizing loss on the depth component of
    predicted 3D coordinates.

    Supported loss types:
    - none:             Returns a zero loss. Useful for disabling the prior loss.
    - rgbd_laplace_nll: Applies a Negative Log-Likelihood (NLL) loss using a Laplace
                        distribution centered at the target depth values from RGB-D data.
                        This loss is only applied where valid target depth is available.
    - laplace_nll:      Applies an NLL loss based on a Laplace distribution with a
                        user-defined location (mean) and bandwidth (scale).
    - laplace_wd:       Calculates the 1-Wasserstein distance between the distribution of
                        predicted depths and a target Laplace distribution with user-defined
                        parameters. This provides a measure of distance between the two
                        probability distributions.
    """

    def __init__(self,
                 prior_loss_type,
                 prior_loss_bandwidth,
                 prior_loss_location,
                 device,
                 prior_diffusion_model_path=None,
                 prior_diffusion_start_step=0,
                 prior_diffusion_warmup_steps=-1,
                 prior_diffusion_subsample=1.0):
        """
        Initializes the PriorLoss class.

        Args:
            prior_loss_type (str): The type of prior loss to use.
            prior_loss_bandwidth (float): The scale/bandwidth for the Laplace distribution.
            prior_loss_location (float): The location/mean for the Laplace distribution.
            device (torch.device): The device on which the prior loss will be computed.
            prior_diffusion_model_path (Path): Path to the diffusion model to be used in the diffusion prior.
            prior_diffusion_start_step (int): Start diffusion regularization after n iterations. Default is 0.
            prior_diffusion_warmup_steps (int): Linear increase of diffusion weight in first N iterations. Default is -1 (disabled).
            prior_diffusion_subsample (float): Sub-sample ratio for inputs to diffusion prior. Default is 1.0 (no subsampling).
        """
        self.type = prior_loss_type
        self.prior_loss_bandwidth = prior_loss_bandwidth
        self.prior_loss_location = prior_loss_location
        self.prior_diffusion_start_step = prior_diffusion_start_step
        self.prior_diffusion_warmup_steps = prior_diffusion_warmup_steps
        self.prior_diffusion_subsample = prior_diffusion_subsample

        if prior_diffusion_model_path is not None:

            # the following packages pull in further requirements that are only needed for the diffusion prior
            # therefore we do a conditional import here
            from diffusion.models.ddpm import GaussianDiffusion
            from diffusion.models.diffusion_utils import DiffusionRegulariser

            model_dict = torch.load(prior_diffusion_model_path)

            diff_model = GaussianDiffusion(sampling_timesteps=200).to(device)
            
            # Handle both checkpoint formats: with 'state_dict' key or direct state_dict
            if 'state_dict' in model_dict:
                # Checkpoint format with nested state_dict
                diff_model.load_state_dict(model_dict['state_dict'])
            else:
                # Direct state_dict format
                diff_model.load_state_dict(model_dict)
            
            diff_model.eval()
            self.diffusion_prior = DiffusionRegulariser(diff_model, device)
        else:
            self.diffusion_prior = None

    def compute(self,
                pred_cam_coords_b31,
                pred_scene_coords_b31,
                reprojection_error_b1,
                iteration,
                max_iterations,
                target_crds_b3=None,
                use_depth=False):
        """
        Computes the prior loss based on the type specified during initialization.

        Args:
            pred_cam_coords_b31 (torch.Tensor): A tensor of shape (batch_size, 3, 1)
                                                containing predicted 3D coordinates in the camera frame.
                                                Used for depth based priors.
            pred_scene_coords_b31 (torch.Tensor): A tensor of shape (batch_size, 3, 1)
                                                  containing predicted 3D coordinates in the scene frame.
                                                  Used for the diffusion prior.
            reprojection_error_b1 (torch.Tensor): A tensor of shape (batch_size, 1)
                                                  containing the reprojection errors of scene coordinates.
                                                  Used in the diffusion prior to filter predictions.
            iteration (int): The current training iteration. Used in the diffusion prior for a weight schedule.
            max_iterations (int): The maximum number of training iterations currently scheduled.
            target_crds_b3 (torch.Tensor, optional): A tensor of shape (batch_size, 3)
                                                     containing target 3D coordinates.
                                                     Required for 'rgbd_laplace_nll'. Defaults to None.
            use_depth (bool, optional): A boolean indicating if depth data from the dataset is being used.
                                        Required for 'rgbd_laplace_nll'. Defaults to False.

        Returns:
            torch.Tensor: The computed prior loss as a scalar tensor.
        """
        # Get the device and batch size from the input tensor.
        device = pred_cam_coords_b31.device
        batch_size = pred_cam_coords_b31.shape[0]

        if self.type == "none":
            raise ValueError("Probabilistic loss needs a prior_loss_type. Using none will almost certainly degenerate.")

        if batch_size == 0:
            return torch.tensor(0.0, device=device)

        # Extract the depth component (z-coordinate) from the predicted coordinates.
        pred_depth_b = pred_cam_coords_b31[:, 2].squeeze()

        if self.type == "rgbd_laplace_nll":
            if not use_depth:
                raise ValueError("Depth prior 'rgbd_laplace_nll' is enabled, but 'use_depth' is False.")
            if target_crds_b3 is None:
                raise ValueError("Target coordinates ('target_crds_b3') are required for 'rgbd_laplace_nll' prior.")

            # Get target depth values from target coordinates
            target_depth_b = target_crds_b3[:, 2]

            # Create a mask to apply the prior only where target depth is available (greater than 0)
            target_depth_available_b = (target_depth_b > 0)
            pred_depth_masked_b = pred_depth_b[target_depth_available_b]
            target_depth_masked_b = target_depth_b[target_depth_available_b]

            if pred_depth_masked_b.nelement() == 0:
                return torch.tensor(0.0, device=device)

            # The prior is a Laplace distribution centered at the target depth values.
            l_locs = target_depth_masked_b
            l_scales = torch.full_like(target_depth_masked_b, self.prior_loss_bandwidth)
            target_dist = distributions.Laplace(l_locs, l_scales)
            log_probs = target_dist.log_prob(pred_depth_masked_b)

            # Final loss is the weighted, normalized negative log-likelihood.
            loss_prior = -torch.sum(log_probs) / batch_size

        elif self.type == "laplace_nll":
            # The prior is a Laplace distribution with user-defined parameters.
            target_dist = distributions.Laplace(self.prior_loss_location, self.prior_loss_bandwidth)
            log_probs = target_dist.log_prob(pred_depth_b)

            # Final loss is the weighted, normalized negative log-likelihood.
            loss_prior = -torch.sum(log_probs) / batch_size

        elif self.type == "laplace_wd":
            N = pred_depth_b.size(0)
            if N == 0:
                return torch.tensor(0.0, device=device)

            # Sort the predicted depth samples to compute the Wasserstein distance.
            sorted_samples, _ = torch.sort(pred_depth_b)

            # The prior is a Laplace distribution with user-defined parameters.
            target_dist = distributions.Laplace(self.prior_loss_location, self.prior_loss_bandwidth)

            # Generate quantiles from the target Laplace distribution.
            # An epsilon is used to avoid numerical instability at the extremes (0 and 1).
            probs = torch.linspace(1e-6, 1 - 1e-6, N, device=device)
            quantiles = target_dist.icdf(probs)

            # The 1-Wasserstein distance is the mean absolute difference between sorted samples and quantiles.
            wasserstein_dist = torch.mean(torch.abs(sorted_samples.squeeze() - quantiles))
            loss_prior = wasserstein_dist

        elif self.type == "diffusion":

            if self.diffusion_prior is None:
                raise ValueError("Loss prior is 'diffusion' but no diffusion model was loaded. Set prior_diffusion_model_path.")

            start_step = self.prior_diffusion_start_step
            warmup_steps = self.prior_diffusion_warmup_steps
            subsample_ratio = self.prior_diffusion_subsample
            
            initial_diffusion_time = 0.05
            diff_freq = 4          
            mask_threshold = 30

            if iteration <= start_step:
                # diffusion regularization not used in the beginning of training
                #print("Not starting diffusion prior for iteration %d" % iteration)
                return 0

            if iteration % diff_freq != 0:
                # diffusion regularization not used in every step to save mapping time
                #print("Skipping diffusion prior for iteration %d" % iteration)
                return 0

            # map ACE training progress to a diffusion time step
            lambda_t = (iteration - start_step) / (max_iterations - start_step)
            lambda_t = min(max(lambda_t, 0.), 1.)
            timestep = initial_diffusion_time * (1. - lambda_t)

            # linear increase of the diffusion weight in the first N iterations
            if warmup_steps > 0 and iteration - start_step < warmup_steps:
                weight = (iteration - start_step) / warmup_steps
            else:
                weight = 1.0

            # sub-sample inputs to the diffusion prior for large batch sizes
            if subsample_ratio < 1.0:
                random_mask = torch.rand(batch_size, device=pred_scene_coords_b31.device) < subsample_ratio
                pred_scene_coords_b31 = pred_scene_coords_b31[random_mask]

            # actual prediction of the diffusion model
            diff_loss, _ = self.diffusion_prior.get_diff_loss(pred_scene_coords_b31.permute(2, 0, 1), timestep)

            # filter out diffusion prediction for points with low reprojection error
            diff_mask = reprojection_error_b1 > mask_threshold
            if subsample_ratio < 1.0:
                diff_mask = diff_mask[random_mask]
            diff_loss = diff_loss * diff_mask.squeeze()[None, None, :]

            # prior is the average negative prediction of the model
            diff_loss = diff_loss.sum(1)
            diff_loss = diff_loss[diff_loss != 0.0]
            if len(diff_loss) == 0:
                diff_loss = 0.
            else:
                diff_loss = -torch.mean(diff_loss)

                try:
                    assert torch.isfinite(diff_loss)
                except AssertionError:
                    print("Diffusion pseudo loss is not finite")

            loss_prior = weight * diff_loss
            #print(f"Diffusion prior {diff_loss}, weight {weight}, timestep {timestep}")

        else:
            raise ValueError(f"Unknown prior type: {self.type}")

        return loss_prior

