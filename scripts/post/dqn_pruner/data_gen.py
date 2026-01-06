import torch
import numpy as np

class NoiseInjector:
    def __init__(self, config):
        self.config = config
        
    def inject_mist(self, points, intensity=0.1, bounds=None):
        """
        Injects a cloud of low-opacity points (Mist).
        """
        N = points.shape[0]
        n_mist = int(N * intensity)
        
        if bounds is None:
            min_pt = points.min(dim=0)[0]
            max_pt = points.max(dim=0)[0]
            bounds = (min_pt, max_pt)
            
        # Generate random positions within bounds
        # Mist tends to be clustered, so we pick a few centers
        n_centers = max(1, n_mist // 100)
        centers = torch.rand(n_centers, 3) * (bounds[1] - bounds[0]) + bounds[0]
        
        # Distribute points around centers
        mist_points = []
        points_per_center = n_mist // n_centers
        
        for c in centers:
            # Gaussian blob
            blob = torch.randn(points_per_center, 3) * 0.5 + c
            mist_points.append(blob)
            
        mist_pos = torch.cat(mist_points)[:n_mist]
        
        return mist_pos
        
    def inject_walls(self, points, intensity=0.05, bounds=None):
        """
        Injects planar noise (Walls).
        """
        N = points.shape[0]
        n_wall = int(N * intensity)
        
        if bounds is None:
            min_pt = points.min(dim=0)[0]
            max_pt = points.max(dim=0)[0]
            bounds = (min_pt, max_pt)
            
        # Pick a random plane axis (x, y, or z)
        axis = np.random.randint(0, 3)
        fixed_val = torch.rand(1) * (bounds[1][axis] - bounds[0][axis]) + bounds[0][axis]
        
        wall_pos = torch.rand(n_wall, 3) * (bounds[1] - bounds[0]) + bounds[0]
        wall_pos[:, axis] = fixed_val + (torch.randn(n_wall) * 0.01) # Thin wall
        
        return wall_pos

    def generate_batch(self, clean_data_dict):
        """
        Takes a dictionary of clean data (positions, opacities, etc.)
        Returns:
            - noisy_features: Tensor [N+Noise, FeatureDim]
            - labels: Tensor [N+Noise] (0=Real, 1=Fake)
        """
        real_pos = clean_data_dict['positions']
        real_ops = clean_data_dict['opacities']
        real_scl = clean_data_dict['scales']
        real_col = clean_data_dict['colors']
        
        # 1. Generate Noise Positions
        mist_pos = self.inject_mist(real_pos, intensity=0.2)
        wall_pos = self.inject_walls(real_pos, intensity=0.05)
        
        noise_pos = torch.cat([mist_pos, wall_pos])
        n_noise = noise_pos.shape[0]
        
        # 2. Generate Noise Attributes
        # Opacity: Floaters usually low, walls can be high
        # We randomize opacity for noise to make it challenging
        noise_ops = torch.rand(n_noise, 1) * -1.0 - 1.0 # Loggit range roughly [-2, -1]
        
        # Scales: Usually messy
        noise_scl = torch.rand(n_noise, 3) * -4.0 # Small scales
        
        # Colors: Greyish or random
        noise_col = torch.rand(n_noise, 3) * 0.5 + 0.25
        
        # 3. Combine Features (Using StateExtractor logic manually for speed)
        # Note: We need a simplified feature assembly here mimicking StateExtractor
        
        # Compute Density (Critical Feature)
        # We need to compute density for the COMBINED set
        full_pos = torch.cat([real_pos, noise_pos])
        
        # For efficiency in training, we might approximate or skip sklearn here
        # But for correctness, we reuse the Extractor's density logic if possible, 
        # or use a fast batch approximation.
        
        # Labels
        labels_real = torch.zeros(real_pos.shape[0])
        labels_noise = torch.ones(noise_pos.shape[0])
        labels = torch.cat([labels_real, labels_noise])
        
        # Shuffle
        perm = torch.randperm(full_pos.shape[0])
        
        # Return RAW data components so StateExtractor can process them properly
        # This is better than trying to fake the features directly
        combined_data = {
            'positions': full_pos[perm],
            'opacities': torch.cat([real_ops, noise_ops])[perm],
            'scales': torch.cat([real_scl, noise_scl])[perm],
            'colors': torch.cat([real_col, noise_col])[perm]
        }
        
        return combined_data, labels[perm]
