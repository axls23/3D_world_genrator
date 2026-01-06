"""
GeNVS-Lite Configuration
Optimized for RTX 4050 (6GB VRAM)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class EnsembleConfig:
    """Weak model ensemble configuration."""
    
    # MiDaS depth estimation
    MIDAS_MODEL: str = "MiDaS_small"  # Smallest variant ~50MB
    MIDAS_WEIGHT: float = 0.25
    
    # Normal estimation
    NORMAL_MODEL: str = "omnidata_tiny"
    NORMAL_WEIGHT: float = 0.20
    
    # Edge detection
    EDGE_METHOD: str = "canny"  # "canny" or "hed"
    EDGE_WEIGHT: float = 0.10
    
    # SAM segmentation
    SAM_MODEL: str = "mobile_sam"  # ~40MB
    SAM_WEIGHT: float = 0.20
    
    # Symmetry prior (heuristic)
    SYMMETRY_WEIGHT: float = 0.25
    SYMMETRY_AXIS: str = "auto"  # "x", "y", "z", or "auto"


@dataclass
class DiffusionConfig:
    """Lightweight diffusion configuration."""
    
    # Model selection
    MODEL_ID: str = "stabilityai/sd-turbo"  # Fast 4-step model
    # Alternative: "latent-consistency/lcm-lora-sdv1-5"
    
    # Generation settings
    NUM_STEPS: int = 4  # SD-Turbo needs only 4 steps
    GUIDANCE_SCALE: float = 0.0  # Turbo doesn't need CFG
    
    # Resolution
    WIDTH: int = 512
    HEIGHT: int = 512
    
    # Conditioning
    CONDITION_SCALE: float = 1.0
    
    # Memory optimization
    ENABLE_ATTENTION_SLICING: bool = True
    ENABLE_VAE_TILING: bool = True
    USE_FP16: bool = True


@dataclass 
class NovelViewConfig:
    """Novel view generation settings."""
    
    # View sampling
    NUM_VIEWS_BACK: int = 20      # Views for back hemisphere
    NUM_VIEWS_SIDES: int = 10     # Views for each side
    AZIMUTH_RANGE: tuple = (90, 270)  # Back hemisphere in degrees
    ELEVATION_RANGE: tuple = (-30, 30)  # Slight vertical variation
    
    # Camera settings
    CAMERA_DISTANCE: float = 2.5  # Distance from center
    LOOK_AT_CENTER: bool = True


@dataclass
class GeNVSConfig:
    """Main GeNVS-Lite configuration."""
    
    # Sub-configs
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    novel_view: NovelViewConfig = field(default_factory=NovelViewConfig)
    
    # Paths
    OUTPUT_DIR: Path = Path("genvs_output")
    CACHE_DIR: Path = Path(".cache/genvs")
    
    # Device
    DEVICE: str = "cuda"
    
    # Memory management
    MAX_BATCH_SIZE: int = 1
    CLEAR_CACHE_FREQUENCY: int = 5  # Clear CUDA cache every N views
    
    # Quality settings
    UPSCALE_FACTOR: int = 2  # Upscale from 256 to 512 after generation
    
    def __post_init__(self):
        self.OUTPUT_DIR = Path(self.OUTPUT_DIR)
        self.CACHE_DIR = Path(self.CACHE_DIR)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Default config instance
default_config = GeNVSConfig()
