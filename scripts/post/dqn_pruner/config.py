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
    
    # Heuristic Thresholds (Tunable). Absolute fallbacks, used only when the matching
    # *_PCT below is None; otherwise thresholds come from the model's own distribution
    # (mean kNN distance is in scene units and opacity logits drift with training, so a
    # fixed value prunes very different fractions on different scenes).
    PRUNE_OPACITY_THR: float = -2.0  # Loggit opacity threshold (approx 0.1 after sigmoid)
    PRUNE_ISOLATION_THR: float = 0.5 # Mean neighbor distance threshold
    HEURISTIC_OPACITY_PCT: Optional[float] = 10.0    # ghost: opacity below this percentile...
    HEURISTIC_GHOST_ISOLATION_PCT: Optional[float] = 90.0  # ...and isolation above this one
    HEURISTIC_ISOLATION_PCT: Optional[float] = 99.0  # floater: isolation above this percentile

    # Context-aware pruning budget and safety gates (percentiles of the model itself)
    PRUNE_BUDGET_FRAC: float = 0.03       # prune at most this fraction (top scores only)
    PRUNE_PROTECT_OPACITY_PCT: float = 50.0  # never prune opacity above this percentile
    PRUNE_MIN_ISOLATION_PCT: float = 75.0    # only prune isolation above this percentile
    
    # Context-Aware Thresholds
    PRUNE_SCALE_RATIO_THR: float = 5.0    # Needle score saturates at max/mid scale ratio 1 + this
    PRUNE_SCALE_OUTLIER_THR: float = 3.0  # Giant: log(max scale / local spacing) > median + this * robust std
    PRUNE_COLOR_OUTLIER_THR: float = 2.5  # Colour vs kNN neighbours > this * robust std
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

from enum import IntEnum

class ActionSpace(IntEnum):
    STEADY = 0
    FIX_POSES = 1
    FILL_HOLES = 2
    PRUNE_POLISH = 3

class StateSpace(IntEnum):
    ACE_CONFIDENCE = 0
    DENSITY = 1
    PSNR = 2
    GENVS_AGG = 3
