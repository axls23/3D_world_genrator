
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrustumFeatureVolume(nn.Module):
    """
    Frustum-Aligned Voxel Grid.
    
    This is primarily a data structure wrapper that handles the 3D grid parameters.
    It doesn't have learnable parameters itself, but defines the coordinate system
    for the renderer.
    """
    
    def __init__(self, 
                 c_feat=16, 
                 depth_planes=64, 
                 spatial_res=128,
                 z_near=0.1,
                 z_far=10.0):
        super().__init__()
        self.c_feat = c_feat
        self.depth_planes = depth_planes
        self.spatial_res = spatial_res
        self.z_near = z_near
        self.z_far = z_far
        
        # Pre-compute depth bins (linear or inverse depth?)
        # Paper typically uses linear in disparity (1/depth) or linear depth.
        # Let's assume linear disparity for better close-up detail.
        self.register_buffer('depth_bins', self._make_depth_bins())

    def _make_depth_bins(self):
        # Linear in disparity space
        disp_min = 1.0 / self.z_far
        disp_max = 1.0 / self.z_near
        
        disps = torch.linspace(disp_max, disp_min, self.depth_planes + 1)
        depths = 1.0 / disps
        # Sort increasingly
        depths, _ = torch.sort(depths)
        return depths

    def forward(self, features):
        """
        Pass-through, just ensures shape compliance.
        Input: [B, C_feat, D, H, W]
        """
        assert features.shape[1] == self.c_feat, f"Expected {self.c_feat} channels, got {features.shape[1]}"
        assert features.shape[2] == self.depth_planes, f"Expected {self.depth_planes} depth planes, got {features.shape[2]}"
        return features
