
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralVolumeRenderer(nn.Module):
    """
    Differentiable Neural Volume Renderer.
    Technique: Ray Marching through Frustum Features.
    
    Paper Spec:
    - Sampling: Stratified (64 samples per ray). No Hierarchical.
    - Interpolation: Trilinear (grid_sample).
    - Aggregation: Mean pooling across feature volumes.
    - Decoder: MLP (16 -> 64 -> 64 -> 17).
    - Resoluton: Renders at 64x64, Upsamples to 128x128.
    """
    
    def __init__(self, 
                 c_feat=16, 
                 n_samples=64, 
                 render_res=64,
                 target_res=128):
        super().__init__()
        self.c_feat = c_feat
        self.n_samples = n_samples
        self.render_res = render_res
        self.target_res = target_res
        
        # Decoding MLP (Section 4.3)
        # Input: 16 (Latent)
        # Output: 1 (Density) + 16 (Feature)
        self.decoder = nn.Sequential(
            nn.Linear(c_feat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1 + c_feat) # Density + Residual Feature
        )
        
    def get_rays(self, output_size, K, c2w, device):
        """
        Generate rays for target viewpoint.
        Args:
            output_size: (H, W)
            K: [B, 3, 3] or [B, 4, 4]
            c2w: [B, 4, 4]
            device: torch.device
        Returns:
            rays_o: [B, H, W, 3]
            rays_d: [B, H, W, 3]
        """
        H, W = output_size
        B = K.shape[0]
        
        # Grid creation [H, W]
        i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device),
                              torch.linspace(0, H-1, H, device=device), indexing='xy')
        
        # Expand to batch: [B, H, W]
        i = i.unsqueeze(0).expand(B, H, W)
        j = j.unsqueeze(0).expand(B, H, W)
        
        # Extract intrinsics [B, 1, 1]
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)
        
        # Camera directions [B, H, W, 3]
        # OpenCV coordinate convention: +X right, +Y down, +Z forward
        # Matches ACE-Zero and COLMAP
        dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)
        
        # Rotate to world [B, H, W, 3]
        # c2w: [B, 4, 4] -> R: [B, 3, 3]
        R = c2w[:, :3, :3] # [B, 3, 3]
        
        # dirs: [B, H, W, 3] -> [B, H*W, 3]
        dirs_flat = dirs.reshape(B, -1, 3)
        
        # Rotate: (R @ dirs^T)^T = dirs @ R^T
        rays_d_flat = torch.bmm(dirs_flat, R.transpose(1, 2))
        rays_d = rays_d_flat.reshape(B, H, W, 3)
        
        # Origin [B, H, W, 3]
        rays_o = c2w[:, :3, -1].view(B, 1, 1, 3).expand(B, H, W, 3)
        
        return rays_o, rays_d

    def sample_stratified(self, rays_o, rays_d, z_near, z_far, n_samples):
        """
        Stratified sampling along rays.
        """
        t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)
        z_vals = z_near * (1.-t_vals) + z_far * (t_vals)
        
        if self.training:
            # Add noise to samples during training (Regularization)
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=rays_o.device)
            z_vals = lower + (upper - lower) * t_rand
            
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        return pts, z_vals

    def forward(self, 
                feature_volumes, 
                source_poses, 
                source_Ks, 
                target_pose, 
                target_K,
                z_near=0.1, 
                z_far=10.0):
        """
        Render a novel view.
        
        Args:
            feature_volumes: [B, N_src, C, D, H, W] (Frustum Volumes)
            source_poses: [B, N_src, 4, 4] (World2Cam usually needed for projection)
            target_pose: [B, 4, 4]
        
        Returns:
            feature_image: [B, C, H, W]
        """
        B, N_src, C, D, H_vol, W_vol = feature_volumes.shape
        device = feature_volumes.device
        
        # 1. Generate Rays for TARGET view
        # We render at low resolution first (64x64) per Section 4.5
        rays_o, rays_d = self.get_rays((self.render_res, self.render_res), target_K, target_pose, device)
        rays_o = rays_o.reshape(B, -1, 3) 
        rays_d = rays_d.reshape(B, -1, 3)
        
        # 2. Stratified Sampling
        pts, z_vals = self.sample_stratified(rays_o, rays_d, z_near, z_far, self.n_samples)
        # pts: [B, N_rays, N_samples, 3]
        
        # 3. Project points into SOURCE views and Sample
        # This is the core "Frustum feature querying"
        # We need to transform World Points -> Source Camera -> NDC -> Grid Sample
        
        # Flatten points: [B, N_rays * N_samples, 3]
        num_rays = rays_o.shape[1]
        num_samples = self.n_samples
        pts_flat = pts.reshape(B, -1, 3) # [B, N_pts, 3]
        N_pts = pts_flat.shape[1]
        
        # 3. Vectorized Projection and Sampling
        # Goal: Compute features for all sources in parallel
        
        # 3a. World -> Source Camera (Batch * N_src)
        # source_poses: [B, N_src, 4, 4]
        # pts_flat: [B, N_pts, 3]
        
        # Expand Poses for batch processing: [B * N_src, 4, 4]
        # But we need to match points to them.
        # Let's expand points to [B, N_src, N_pts, 3] first
        pts_expanded = pts_flat.unsqueeze(1).expand(-1, N_src, -1, -1) # [B, N_src, N_pts, 3]
        pts_reshaped = pts_expanded.reshape(B * N_src, N_pts, 3) 
        
        # Invert Poses
        c2w_reshaped = source_poses.reshape(B * N_src, 4, 4)
        w2c_reshaped = torch.inverse(c2w_reshaped) # [B*N_src, 4, 4]
        
        # Transform Points
        ones = torch.ones_like(pts_reshaped[..., :1])
        pts_homo = torch.cat([pts_reshaped, ones], dim=-1) # [B*N_src, N_pts, 4]
        
        # (w2c @ pts^T)^T = pts @ w2c^T
        cam_pts = torch.bmm(pts_homo, w2c_reshaped.transpose(1, 2)) # [B*N_src, N_pts, 4]
        cam_pts = cam_pts[..., :3]
        
        # 3b. Camera -> Image (NDC)
        # source_Ks: [B, N_src, 3, 3]
        K_reshaped = source_Ks.reshape(B * N_src, 3, 3)
        
        fx = K_reshaped[:, 0, 0].unsqueeze(1) # [B*N_src, 1]
        fy = K_reshaped[:, 1, 1].unsqueeze(1)
        cx = K_reshaped[:, 0, 2].unsqueeze(1)
        cy = K_reshaped[:, 1, 2].unsqueeze(1)
        
        x_cam = cam_pts[..., 0]
        y_cam = cam_pts[..., 1]
        z_cam = cam_pts[..., 2]
        
        # Avoid division by zero
        z_cam = torch.clamp(z_cam, min=1e-5)
        
        # Project
        u = (fx * x_cam) / z_cam + cx
        v = (fy * y_cam) / z_cam + cy
        
        # Normalize to [-1, 1]
        w_size = W_vol
        h_size = H_vol
        
        u_norm = (u / (w_size - 1)) * 2 - 1
        v_norm = (v / (h_size - 1)) * 2 - 1
        
        # 3c. Depth -> Disparity -> Normalized Grid Z
        disp = 1.0 / z_cam
        disp_min = 1.0 / z_far
        disp_max = 1.0 / z_near
        
        d_norm = 2 * (disp - disp_max) / (disp_min - disp_max) - 1
        
        # Stack Grid: [B*N_src, N_pts, 3]
        sample_grid = torch.stack([u_norm, v_norm, d_norm], dim=-1)
        sample_grid = sample_grid.view(B * N_src, 1, 1, N_pts, 3)
        
        # Reshape Volume for Sampling
        # [B, N_src, C, D, H, W] -> [B*N_src, C, D, H, W]
        vol_reshaped = feature_volumes.reshape(B * N_src, C, D, H_vol, W_vol)
        
        # Grid Sample
        sampled = F.grid_sample(
            vol_reshaped,
            sample_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ) # [B*N_src, C, 1, 1, N_pts]
        
        sampled = sampled.squeeze(2).squeeze(2).permute(0, 2, 1) # [B*N_src, N_pts, C]
        
        # Aggregate (Mean Pooling)
        # Reshape back to [B, N_src, N_pts, C]
        sampled_grouped = sampled.reshape(B, N_src, N_pts, C)
        
        # Mean over sources (dim 1)
        sampled_features = torch.mean(sampled_grouped, dim=1) # [B, N_pts, C]
        
        # Reshape back to ray/sample structure
        sampled_features = sampled_features.reshape(B, num_rays, num_samples, C) 
        
        # 4. Decoding MLP
        # [B, N_rays, N_samples, C] -> [..., 1+C]
        raw_output = self.decoder(sampled_features)
        density = raw_output[..., 0] # logits
        feat_res = raw_output[..., 1:]
        
        # Residual connection (Section 4.3): Output = MLP(Input) + Input
        features = feat_res + sampled_features
        
        # 5. Volumetric Integration
        # Standard NeRF rendering equations
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1)
        
        alpha = 1. - torch.exp(-F.softplus(density) * dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], alpha.shape[1], 1), device=device), 1.-alpha + 1e-10], -1), -1)[..., :-1]
        
        rendered_feat = torch.sum(weights[...,None] * features, -2) # [B, N_rays, C]
        
        # Render Depth (Expected depth along ray)
        rendered_depth = torch.sum(weights * z_vals, -1)  # [B, N_rays]
        
        # Reshape to Image
        feature_image = rendered_feat.reshape(B, self.render_res, self.render_res, C).permute(0, 3, 1, 2)
        depth_image = rendered_depth.reshape(B, self.render_res, self.render_res, 1).permute(0, 3, 1, 2)
        
        # 6. Upsample
        if self.target_res != self.render_res:
            feature_image = F.interpolate(feature_image, size=(self.target_res, self.target_res), mode='bilinear', align_corners=False)
            depth_image = F.interpolate(depth_image, size=(self.target_res, self.target_res), mode='bilinear', align_corners=False)
            
        return feature_image, depth_image

import numpy as np
