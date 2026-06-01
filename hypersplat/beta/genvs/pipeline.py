
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import tqdm

from .encoder import GeometryEncoder
from .feature_volume import FrustumFeatureVolume
from .rendering import NeuralVolumeRenderer
from .unet_2d import DiffusionUNet
from .dit_2d import DiT_S_2

class GeNVSPipeline(nn.Module):
    """
    Main Orchestrator for True GeNVS.
    
    Coordinates:
    1. Geometry Stage: Image -> Volume -> Rendered Feature
    2. Generative Stage: Noisy RGB + Feature -> Denoised RGB (via DDIM)
    """
    
    def __init__(self, device='cuda', model_channels=64, **unet_kwargs):
        super().__init__()
        self.device = device
        self.model_channels = model_channels
        
        # Initialize components
        self.encoder = GeometryEncoder().to(device)
        self.volume_struct = FrustumFeatureVolume().to(device)
        self.renderer = NeuralVolumeRenderer().to(device)
        
        # 2. Generative Component: DiT
        self.unet = DiT_S_2(in_channels=19, out_channels=3, input_size=128).to(device)

        # 3. Noise Schedule (Linear Beta) - Consistent with train.py
        self.num_timesteps = 1000
        beta = torch.linspace(1e-4, 0.02, self.num_timesteps).to(device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

    def geometric_stage_batch(self, 
                        source_image: torch.Tensor, 
                        source_pose: torch.Tensor, 
                        source_K: torch.Tensor,
                        target_poses: torch.Tensor,
                        target_Ks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the geometry pass for a batch of TARGET views (single source).
        
        Args:
            source_image: [1, 3, H, W]
            target_poses: [B, 4, 4]
            target_Ks: [B, 3, 3] or [B, 4, 4]
            
        Returns:
            feature_images: [B, 16, H, W]
            depth_images: [B, 1, H, W]
        """
        B_targets = target_poses.shape[0]
        
        # 1. Encode -> 3D Volume (Once per source)
        # source_image: [1, 3, H, W]
        with torch.no_grad():
            flattened_features = self.encoder(source_image) # [1, C, D, H, W]
            volume = self.volume_struct(flattened_features)
            
            # 2. Render Target Views (Batch)
            # Expand volume for the renderer: [B_targets, 1_src, C, D, H, W]
            volume_expanded = volume.expand(B_targets, -1, -1, -1, -1, -1)
            
            # Expand source pose
            source_pose_expanded = source_pose.expand(B_targets, 1, 4, 4)
            source_K_expanded = source_K.expand(B_targets, 1, 3, 3) 
            
            feature_images, depth_images = self.renderer(
                volume_expanded, source_pose_expanded, source_K_expanded, target_poses, target_Ks
            ) # [B, 16, H, W], [B, 1, H, W]
            
        return feature_images, depth_images

    def sample_batch(self, 
               source_image: torch.Tensor, 
               source_pose: torch.Tensor, 
               source_K: torch.Tensor,
               target_poses: torch.Tensor, 
               target_Ks: torch.Tensor,
               num_steps: int = 50,
               batch_size: int = 4) -> torch.Tensor:
        """
        Generate novel views in batches to optimize GPU usage.
        """
        self.eval()
        B_total = target_poses.shape[0]
        generated_images = []
        
        print(f"GeNVS: Generating {B_total} views with batch size {batch_size}...")
        
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, B_total, batch_size)):
                # Slice batch
                batch_poses = target_poses[i : i + batch_size]
                batch_Ks = target_Ks[i : i + batch_size]
                current_B = batch_poses.shape[0]
                
                # 1. Geometric Stage (Batched)
                cond_features, batch_depths = self.geometric_stage_batch(
                    source_image, source_pose, source_K, batch_poses, batch_Ks
                ) # [current_B, 16, H, W], [current_B, 1, H, W]
                
                # 2. Diffusion Loop (Batched)
                x_t = torch.randn(current_B, 3, cond_features.shape[2], cond_features.shape[3], device=self.device)
                timesteps = torch.linspace(999, 0, num_steps, device=self.device).long()
                
                for t_idx, t in enumerate(timesteps):
                    t_batch = t.expand(current_B)
                    
                    noise_pred = self.unet(x_t, t_batch, cond_features)
                    
                    # DDIM Update (Geometric Schedule)
                    # Use actual alpha_bar from the schedule
                    ab_t = self.alpha_bar[t]
                    ab_prev = self.alpha_bar[timesteps[min(t_idx+1, num_steps-1)]]
                    
                    sigma = 0 # Deterministic DDIM
                    
                    # x_0 prediction
                    pred_x0 = (x_t - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
                    # Direction pointing to x_t
                    dir_xt = torch.sqrt(1 - ab_prev - sigma**2) * noise_pred
                    # Update
                    x_t = torch.sqrt(ab_prev) * pred_x0 + dir_xt
                
                generated_images.append(x_t.clamp(-1, 1).cpu())
                
        return torch.cat(generated_images, dim=0) # [B_total, 3, H, W]

    def sample_depth_only(self, 
               source_image: torch.Tensor, 
               source_pose: torch.Tensor, 
               source_K: torch.Tensor,
               target_poses: torch.Tensor, 
               target_Ks: torch.Tensor) -> torch.Tensor:
        """
        Generate depth maps for novel views WITHOUT diffusion (fast!).
        Skips the UNet entirely and returns depth directly from volume rendering.
        
        Returns:
            depth_images: [B_total, 1, H, W] - Depth maps for each target view
        """
        self.eval()
        B_total = target_poses.shape[0]
        
        with torch.no_grad():
            # Single pass through geometric stage - no diffusion needed!
            _, depth_images = self.geometric_stage_batch(
                source_image, source_pose, source_K, target_poses, target_Ks
            )
            
    def calc_sds_loss(self, 
                      rgb_render: torch.Tensor,
                      source_image: torch.Tensor,
                      source_pose: torch.Tensor,
                      source_K: torch.Tensor,
                      target_pose: torch.Tensor,
                      target_K: torch.Tensor,
                      guidance_scale: float = 100.0) -> torch.Tensor:
        """
        Calculate Score Distillation Sampling (SDS) Gradient.
        
        Args:
           rgb_render: [1, 3, H, W] - Differentiable render from 3DGS (range [0, 1])
           source_image: [1, 3, H, W] - Conditioning image
           target_pose: [1, 4, 4] - Camera pose of the render
           
        Returns:
           grad: Gradient to backpropagate into rgb_render
        """
        # 1. Prepare Inputs
        # SDS requires inputs in [-1, 1]
        x_0 = rgb_render * 2.0 - 1.0 # [0,1] -> [-1,1]
        
        # 2. Geometry Pass (Conditioning)
        # We need the feature conditioning for the TARGET POSE
        # (Where the 'camera' is currently looking)
        with torch.no_grad():
            cond_features, _ = self.geometric_stage_batch(
                source_image, source_pose, source_K, 
                target_pose, target_K # Target is the current render view
            )
            
            # 3. Add Noise (Forward Diffusion)
            # Sample random timestep t ~ [0.02, 0.98] to avoid singularities
            t = torch.randint(20, 980, (1,), device=self.device).long()
            noise = torch.randn_like(x_0)
            
            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
            alpha_bar_t = self.alpha_bar[t].view(1, 1, 1, 1)
            x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
            
            # 4. Predict Noise (UNet)
            noise_pred = self.unet(x_t, t, cond_features)
            
            # 5. Compute SDS Gradient
            # grad = w(t) * (noise_pred - noise)
            # We assume w(t) = 1 for simplicity (or constant)
            grad = guidance_scale * (noise_pred - noise)
            
            # [Optional] Manifold Constraint (MHC-inspired):
            # Project gradient to remove components that violate 3D consistency?
            # For now, standard SDS.
            
        # 6. Backpropagate Gradient Manually
        # We want to minimize separation between 'noise_pred' and 'noise'
        # The gradient of Loss w.r.t x_0 is proportional to (noise_pred - noise)
        # We start autograd chain here using backward hook
        
        # Since x_0 requires grad (it comes from 3DGS), we can just compute a surrogate loss
        # Loss = 0.5 * MSE(x_0 - (x_t - pred_noise...)) ?
        # Standard SDS Pattern:
        # loss = w * (noise_pred - noise).detach() * x_0
        # This pushes x_0 in direction of 'estimated score'
            
        # Ref: DreamFusion
        # w = 1 - alpha_bar[t] (for certain formulation) 
        # Here we just use the raw difference
        
        # We construct a loss scalar whose derivative is the SDS gradient
        # grad_term = (noise_pred - noise)
        # loss = (grad_term * x_0).sum()
        
        grad_term = grad.detach()
        loss = (grad_term * x_0).sum()
            
        return loss

    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint saved by GeNVSTrainer.
        
        The trainer saves each component's state_dict separately:
        {'encoder': ..., 'volume': ..., 'renderer': ..., 'unet': ..., 'step': ...}
        
        Note: The trainer uses 'volume' as the key, but this pipeline stores
        the FrustumFeatureVolume as 'self.volume_struct'.
        
        Returns:
            step: The training step at which this checkpoint was saved.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load each component's state dict
        if 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if 'volume' in checkpoint:
            # Key mapping: trainer saves as 'volume', pipeline stores as 'volume_struct'
            self.volume_struct.load_state_dict(checkpoint['volume'])
        elif 'volume_struct' in checkpoint:
            self.volume_struct.load_state_dict(checkpoint['volume_struct'])
        if 'renderer' in checkpoint:
            self.renderer.load_state_dict(checkpoint['renderer'])
        if 'unet' in checkpoint:
            try:
                self.unet.load_state_dict(checkpoint['unet'])
            except RuntimeError as e:
                logging.warning(f"[GeNVS] Failed to load UNet/DiT weights (architecture mismatch?): {e}")
        
        step = checkpoint.get('step', 0)
        logging.info(f"[GeNVS] Loaded checkpoint from step {step} ({path})")
        return step

    def save_checkpoint(self, path: str, step: int):
        """
        Save a checkpoint in the same format as GeNVSTrainer.
        Enables symmetric load/save from both trainer and pipeline.
        """
        torch.save({
            'step': step,
            'encoder': self.encoder.state_dict(),
            'volume': self.volume_struct.state_dict(),
            'renderer': self.renderer.state_dict(),
            'unet': self.unet.state_dict(),
            'config': {
                'model_channels': self.model_channels,
            }
        }, path)
        logging.info(f"[GeNVS] Saved checkpoint at step {step} to {path}")

    def save_config(self, output_path: str):
        """Dump the internal diffusion hyperparameters to a YAML file."""
        import yaml
        
        config = {
            "model_name": "GeNVS-Core",
            "diffusion": {
                "num_timesteps": self.num_timesteps,
                "beta_start": float(self.beta[0]),
                "beta_end": float(self.beta[-1]),
                "scheduler": "linear"
            },
            "architecture": {
                "encoder": "GeometryEncoder",
                "unet_channels": 64, # Default from __init__ call in simple_trainer
                "feature_volume_dim": 32, # Default
                "volume_depth": 32 
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"[GeNVS] Config saved to {output_path}")
