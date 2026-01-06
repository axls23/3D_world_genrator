from dataclasses import dataclass
from typing import Optional

@dataclass
class PrunerConfig:
    # Reward weights (tune for quality vs preservation)
    ALPHA_PSNR: float = 1.0      # ↑ = prioritize visual quality
    BETA_OVERPRUNE: float = 2.0  # ↑ = preserve more geometry (penalty for holes)
    GAMMA_UNDERPRUNE: float = 0.5 # ↑ = be more aggressive (penalty for floaters)

    # Thresholds for heuristic fallback and feature extraction
    DENSITY_K: int = 16          # Number of neighbors for KNN
    DENSITY_RADIUS: float = 0.1  # Radius for density calculation
    
    # Heuristic Thresholds (Tunable)
    PRUNE_OPACITY_THR: float = -2.0  # Loggit opacity threshold (approx 0.1 after sigmoid)
    PRUNE_ISOLATION_THR: float = 0.5 # Mean neighbor distance threshold
    
    # Context-Aware Thresholds
    PRUNE_SCALE_RATIO_THR: float = 5.0    # Prune if max_scale/min_scale > this (elongated splats)
    PRUNE_SCALE_OUTLIER_THR: float = 3.0  # Prune if scale > mean + this * std (giant splats)
    PRUNE_COLOR_OUTLIER_THR: float = 2.5  # Prune if color distance > this * std (weird colors)
    PRUNE_DEPTH_OUTLIER_THR: float = 2.0  # Prune if depth from center > this * std (behind scene)
    
    # Neighborhood coherence
    NEIGHBOR_COLOR_WEIGHT: float = 0.3    # Weight for color similarity check
    NEIGHBOR_SCALE_WEIGHT: float = 0.3    # Weight for scale similarity check
    NEIGHBOR_OPACITY_WEIGHT: float = 0.4  # Weight for opacity similarity check
    
    # Model Architecture
    INPUT_DIM: int = 16          # Feature vector dimension
    HIDDEN_DIM: int = 128
    ACTION_DIM: int = 2          # 0=KEEP, 1=PRUNE
    LEARNING_RATE: float = 1e-4
    GAMMA_DQN: float = 0.99      # Discount factor

    # Inference settings
    BATCH_SIZE: int = 4096       # Batch size for processing gaussians
    USE_GPU: bool = True
    
    # Context-aware mode
    CONTEXT_AWARE: bool = True   # Enable context-aware pruning by default
    CAMERAS_PATH: Optional[str] = None  # Path to camera poses for visibility check
