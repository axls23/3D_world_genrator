import torch
import numpy as np
from plyfile import PlyData
import os

class StateExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')

    def load_ply(self, path):
        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        
        # Extract basic properties
        # Positions
        x = torch.tensor(vertex['x'].astype(np.float32))
        y = torch.tensor(vertex['y'].astype(np.float32))
        z = torch.tensor(vertex['z'].astype(np.float32))
        positions = torch.stack([x, y, z], dim=1)

        # Opacities
        opacities = torch.tensor(vertex['opacity'].astype(np.float32)).unsqueeze(1)
        
        # Scales (usually 3 dimensions)
        s0 = torch.tensor(vertex['scale_0'].astype(np.float32))
        s1 = torch.tensor(vertex['scale_1'].astype(np.float32))
        s2 = torch.tensor(vertex['scale_2'].astype(np.float32))
        scales = torch.stack([s0, s1, s2], dim=1)
        
        # Colors (SH DC - usually f_dc_0, f_dc_1, f_dc_2)
        # Note: naming might vary, checking standard gsplat output
        # Usually f_dc_0, f_dc_1, f_dc_2 for RGB
        try:
            c0 = torch.tensor(vertex['f_dc_0'].astype(np.float32))
            c1 = torch.tensor(vertex['f_dc_1'].astype(np.float32))
            c2 = torch.tensor(vertex['f_dc_2'].astype(np.float32))
            colors = torch.stack([c0, c1, c2], dim=1)
        except:
            # Fallback if names are different
            colors = torch.zeros_like(positions)

        return {
            'positions': positions,
            'opacities': opacities,
            'scales': scales,
            'colors': colors,
            'plydata': plydata # Keep original for saving
        }

    def compute_density(self, positions):
        """
        Compute isolation score based on K nearest neighbors.
        Returns: (N, 1) tensor of mean distance to K neighbors
        """
        # Uses CPU with sklearn to avoid GPU OOM on large clouds
        from sklearn.neighbors import NearestNeighbors
        
        pos_np = positions.cpu().numpy()
        N = pos_np.shape[0]
        K = self.config.DENSITY_K
        
        # Fit NN
        nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto', n_jobs=-1).fit(pos_np)
        distances, _ = nbrs.kneighbors(pos_np)
        
        # Exclude self (index 0)
        mean_dist = distances[:, 1:].mean(axis=1)
        
        return torch.tensor(mean_dist.astype(np.float32)).unsqueeze(1)

    def extract_features(self, data):
        """
        Construct the feature vector S
        """
        pos = data['positions']
        ops = data['opacities']
        scl = data['scales']
        col = data['colors']
        
        # 1. Density / Isolation
        print(f"Computing density for {pos.shape[0]} points...")
        isolation = self.compute_density(pos).to(self.device)
        
        # Move others to device
        ops = ops.to(self.device)
        scl = scl.to(self.device)
        col = col.to(self.device)
        pos = pos.to(self.device)
        
        # Normalize positions (center around 0, scale to unit cube approx)
        center = pos.mean(dim=0, keepdim=True)
        scale = pos.std(dim=0, keepdim=True) + 1e-6
        pos_norm = (pos - center) / scale
        
        # Normalize scales
        scl_mean = scl.mean(dim=0, keepdim=True)
        scl_std = scl.std(dim=0, keepdim=True) + 1e-6
        scl_norm = (scl - scl_mean) / scl_std
        
        # 2. Local variance (simple heuristic: distance from mean color/scale of neighbors)
        # This is expensive to compute exactly, using placeholder or simplified version
        # For now, we stick to point-intrinsic properties + global statistical stats
        
        # Feature assembly
        # [Opacity(1), Scale(3), Color(3), Isolation(1), PosNorm(3)] = 11 features
        # Padding to 16 for future expansion (gradients, view dependent, etc)
        
        features = torch.cat([
            ops,            # 1
            scl_norm,       # 3
            col,            # 3
            isolation,      # 1
            pos_norm,       # 3
        ], dim=1)
        
        # Pad with zeros to match INPUT_DIM
        pad_size = self.config.INPUT_DIM - features.shape[1]
        if pad_size > 0:
            padding = torch.zeros((features.shape[0], pad_size), device=self.device)
            features = torch.cat([features, padding], dim=1)
            
        return features

    def save_ply(self, original_plydata, mask, output_path):
        from plyfile import PlyData, PlyElement
        
        # Mask is boolean tensor [N]
        mask_np = mask.cpu().numpy()
        
        # Filter vertices
        # We need to reconstruct the PlyElement with filtered data
        vertex_data = original_plydata['vertex'].data
        filtered_data = vertex_data[mask_np]
        
        el = PlyElement.describe(filtered_data, 'vertex')
        PlyData([el], text=False).write(output_path)
