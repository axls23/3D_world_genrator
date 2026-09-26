import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        # Director Agent Architecture from insights
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim//2), # 256 -> 128
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
        
        # Dimensions from insights: State=32, Action=4, Hidden=256
        input_dim = getattr(config, 'INPUT_DIM', 32)
        hidden_dim = getattr(config, 'HIDDEN_DIM', 256)
        action_dim = getattr(config, 'ACTION_DIM', 4)
        
        self.q_net = QNetwork(input_dim, hidden_dim, action_dim).to(self.device)
        self.q_net.eval()
        
    def load(self, path):
        if torch.cuda.is_available():
            self.q_net.load_state_dict(torch.load(path))
        else:
            self.q_net.load_state_dict(torch.load(path, map_location='cpu'))
            
    def predict(self, state_tensor):
        """
        Predict action index for a single state or batch.
        state_tensor: [Batch, 32] or [32]
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        with torch.no_grad():
            state_tensor = state_tensor.to(self.device)
            q_values = self.q_net(state_tensor)
            action_idx = torch.argmax(q_values, dim=1)
            
        return action_idx

class HeuristicAgent:
    """
    Fallback agent that uses strict thresholds (Rule-Based)
    Instead of learned Q-values, it checks:
    1. Is opacity too low?
    2. Is point too isolated?
    3. Is scale too large?
    """
    def __init__(self, config):
        self.config = config
        
    def predict(self, states, raw_data):
        # Raw data contains: positions, opacities, scales, densities
        # Features mapping from StateExtractor:
        # 0: Opacity
        # 1-3: Scale Norm
        # 4-6: Color
        # 7: Isolation (Density)
        # 8-10: Pos Norm
        
        opacities = states[:, 0]
        isolation = states[:, 7]
        
        # Sigmoid activation on opacity usually in gsplat, but here we assume raw values
        # If opacity < threshold -> PRUNE (1)
        # If isolation > distance_threshold -> PRUNE (1)
        
        # Defaults from Config
        OPACITY_THR = self.config.PRUNE_OPACITY_THR
        ISOLATION_THR = self.config.PRUNE_ISOLATION_THR
        
        # simple logic: Prune if (Transparent AND Isolated) OR (Very Isolated)
        
        # Hard isolation prune
        is_floater = isolation > ISOLATION_THR
        
        # Soft noise prune
        is_ghost = (opacities < OPACITY_THR) & (isolation > (ISOLATION_THR * 0.5))
        
        to_prune = is_floater | is_ghost
        
        return to_prune.long() # 0 = KEEP (False), 1 = PRUNE (True)


class ContextAwareAgent:
    """
    Advanced context-aware pruning agent that considers:
    1. Isolation (spatial density)
    2. Scale anomalies (too large or elongated)
    3. Color outliers (weird colors compared to neighbors)
    4. Depth consistency (behind/in-front of scene)
    5. Neighborhood coherence (weighted similarity score)
    
    Returns a confidence score [0, 1] which is then thresholded.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
        
    def predict(self, states, raw_data):
        """
        Context-aware pruning with multi-feature scoring.
        
        Features in states tensor (from StateExtractor):
        0: Opacity
        1-3: Scale (normalized)
        4-6: Color (SH DC)
        7: Isolation (mean neighbor distance)
        8-10: Position (normalized)
        """
        N = states.shape[0]
        scores = torch.zeros(N, device=self.device)  # Higher = more likely to prune
        
        # Move to device
        states = states.to(self.device)
        
        # Extract features
        opacities = states[:, 0]
        scales_norm = states[:, 1:4]
        colors = states[:, 4:7]
        isolation = states[:, 7]
        pos_norm = states[:, 8:11]
        
        # Get raw scales for ratio check
        raw_scales = raw_data['scales'].to(self.device)
        
        print(f"[ContextAware] Running multi-feature analysis on {N} Gaussians...")
        
        # ============ FEATURE 1: Isolation Score ============
        # Normalized to [0, 1] range
        iso_mean = isolation.mean()
        iso_std = isolation.std() + 1e-6
        iso_score = torch.clamp((isolation - iso_mean) / (3 * iso_std), 0, 1)
        scores += iso_score * 0.25
        
        # ============ FEATURE 2: Opacity Score ============
        # Low opacity = more likely floater
        # Using sigmoid assumption: raw values are logits
        opacity_prob = torch.sigmoid(opacities)
        opacity_score = 1.0 - opacity_prob  # Lower opacity = higher prune score
        scores += opacity_score * 0.20
        
        # ============ FEATURE 3: Scale Anomaly Score ============
        # 3a: Elongation (max/min ratio)
        scale_max = raw_scales.max(dim=1).values
        scale_min = raw_scales.min(dim=1).values + 1e-6
        elongation = scale_max / scale_min
        elongation_score = torch.clamp((elongation - 1) / self.config.PRUNE_SCALE_RATIO_THR, 0, 1)
        scores += elongation_score * 0.10
        
        # 3b: Size outlier (giant splats)
        scale_magnitude = raw_scales.abs().max(dim=1).values
        scale_mean = scale_magnitude.mean()
        scale_std = scale_magnitude.std() + 1e-6
        size_outlier = (scale_magnitude - scale_mean) / scale_std
        size_score = torch.clamp(size_outlier / self.config.PRUNE_SCALE_OUTLIER_THR, 0, 1)
        scores += size_score * 0.10
        
        # ============ FEATURE 4: Color Outlier Score ============
        color_mean = colors.mean(dim=0, keepdim=True)
        color_dist = torch.norm(colors - color_mean, dim=1)
        color_std = color_dist.std() + 1e-6
        color_outlier = color_dist / (self.config.PRUNE_COLOR_OUTLIER_THR * color_std)
        color_score = torch.clamp(color_outlier, 0, 1)
        scores += color_score * 0.10
        
        # ============ FEATURE 5: Depth Outlier Score ============
        # Points too far from scene center (normalized)
        depth_from_center = torch.norm(pos_norm, dim=1)
        depth_mean = depth_from_center.mean()
        depth_std = depth_from_center.std() + 1e-6
        depth_outlier = (depth_from_center - depth_mean) / depth_std
        depth_score = torch.clamp(depth_outlier / self.config.PRUNE_DEPTH_OUTLIER_THR, 0, 1)
        scores += depth_score * 0.15
        
        # ============ FEATURE 6: Combined Suspicious Pattern ============
        # Suspicious = isolated AND (low opacity OR weird scale OR weird color)
        suspicious = (iso_score > 0.3) & (
            (opacity_score > 0.5) | 
            (size_score > 0.3) | 
            (color_score > 0.5)
        )
        scores += suspicious.float() * 0.10
        
        # ============ Threshold Decision ============
        # Score range [0, 1], threshold at 0.4 for balanced pruning
        threshold = 0.4
        to_prune = scores > threshold
        
        # Stats
        n_prune = to_prune.sum().item()
        print(f"[ContextAware] Score distribution: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
        print(f"[ContextAware] Decisions: KEEP={N - n_prune}, PRUNE={n_prune} ({100*n_prune/N:.1f}%)")
        
        return to_prune.long()  # 0 = KEEP, 1 = PRUNE
