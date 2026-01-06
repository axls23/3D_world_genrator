"""
Hybrid COLMAP/ACE-Zero Processor

This module provides a unified interface for video-to-dataset processing,
using ACE-Zero as the primary method and COLMAP as a fallback.

The hybrid approach:
1. Tries ACE-Zero first (fast, COLMAP-free, AI-powered)
2. Falls back to COLMAP if ACE-Zero fails (established, requires COLMAP installed)
3. Can optionally use depth estimation to enhance either method

Usage:
    from hybrid_processor import HybridVideoProcessor
    
    processor = HybridVideoProcessor(output_dir="./data/scene")
    success = processor.process_video("video.mp4")
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Import the ACE-Zero wrapper
try:
    from .ace_zero_wrapper import ACEZeroPoseEstimator, ACEZERO_AVAILABLE
except ImportError:
    try:
        from ace_zero_wrapper import ACEZeroPoseEstimator, ACEZERO_AVAILABLE
    except ImportError:
        ACEZERO_AVAILABLE = False
        ACEZeroPoseEstimator = None


class ProcessingMethod(Enum):
    """Available processing methods."""
    COLMAP = "colmap"
    ACE_ZERO = "ace_zero"
    AUTO = "auto"  # Try ACE-Zero first, fallback to COLMAP


class HybridVideoProcessor:
    """
    Unified video processor with ACE-Zero and COLMAP support.
    
    Attempts ACE-Zero first for speed and COLMAP-free processing.
    Falls back to COLMAP when ACE-Zero fails or produces insufficient results.
    """
    
    def __init__(
        self,
        output_dir: Path,
        method: ProcessingMethod = ProcessingMethod.AUTO,
        colmap_config: Optional[Dict] = None,
        ace_zero_config: Optional[Dict] = None
    ):
        self.output_dir = Path(output_dir)
        self.method = method
        self.colmap_config = colmap_config or {}
        self.ace_zero_config = ace_zero_config or {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track which method was used
        self.method_used: Optional[ProcessingMethod] = None
        self.colmap_success = False
        self.ace_zero_success = False
        
        logger.info(f"Hybrid processor initialized with method: {method.value}")
        logger.info(f"ACE-Zero available: {ACEZERO_AVAILABLE}")
    
    def process_video(
        self,
        video_path: str,
        fps: float = 1.5,
        min_frames: int = 10
    ) -> Tuple[bool, str]:
        """
        Process video to create 3DGS-ready dataset.
        
        Args:
            video_path: Path to input video
            fps: Frame extraction rate
            min_frames: Minimum frames required for valid dataset
            
        Returns:
            Tuple of (success, message)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            return False, f"Video not found: {video_path}"
        
        if self.method == ProcessingMethod.COLMAP:
            return self._process_with_colmap(video_path, fps)
        
        elif self.method == ProcessingMethod.ACE_ZERO:
            return self._process_with_ace_zero(video_path, fps)
        
        else:  # AUTO mode
            # Try ACE-Zero first (faster, no COLMAP dependency)
            logger.info("AUTO mode: Attempting ACE-Zero first...")
            success, message = self._process_with_ace_zero(video_path, fps)
            
            if success:
                self.method_used = ProcessingMethod.ACE_ZERO
                return True, f"ACE-Zero succeeded: {message}"
            
            # Fallback to COLMAP
            logger.warning(f"ACE-Zero failed: {message}. Trying COLMAP fallback...")
            success, message = self._process_with_colmap(video_path, fps)
            
            if success:
                self.method_used = ProcessingMethod.COLMAP
                return True, f"COLMAP succeeded (fallback): {message}"
            else:
                return False, f"Both ACE-Zero and COLMAP failed. Last error: {message}"
    
    def _process_with_ace_zero(self, video_path: Path, fps: float) -> Tuple[bool, str]:
        """Process with ACE-Zero AI pose estimation."""
        try:
            estimator = ACEZeroPoseEstimator(
                output_dir=self.output_dir,
                **self.ace_zero_config
            )
            
            success = estimator.process_video(str(video_path), fps=fps)
            
            if success:
                self.ace_zero_success = True
                return True, f"Created dataset at {self.output_dir}"
            else:
                return False, "ACE-Zero processing failed"
                
        except Exception as e:
            return False, f"ACE-Zero error: {str(e)}"
    
    def _process_with_colmap(self, video_path: Path, fps: float) -> Tuple[bool, str]:
        """Process with traditional COLMAP."""
        try:
            # Import the enhanced video to colmap module
            script_dir = Path(__file__).parent
            sys.path.insert(0, str(script_dir))
            
            from enhanced_video_to_colmap import EnhancedVideoToColmap
            
            processor = EnhancedVideoToColmap(
                video_path=str(video_path),
                output_dir=str(self.output_dir),
                fps=fps,
                **self.colmap_config
            )
            
            processor.run()
            
            # Validate output
            sparse_dir = self.output_dir / "sparse" / "0"
            if sparse_dir.exists() and any(sparse_dir.glob("*.bin")):
                self.colmap_success = True
                return True, f"Created COLMAP dataset at {self.output_dir}"
            else:
                return False, "COLMAP ran but produced no sparse reconstruction"
                
        except FileNotFoundError as e:
            return False, f"COLMAP not found: {str(e)}"
        except Exception as e:
            return False, f"COLMAP processing error: {str(e)}"
    
    def get_status(self) -> Dict:
        """Get processing status."""
        return {
            "method_configured": self.method.value,
            "method_used": self.method_used.value if self.method_used else None,
            "colmap_success": self.colmap_success,
            "ace_zero_success": self.ace_zero_success,
            "ace_zero_available": ACEZERO_AVAILABLE,
            "output_dir": str(self.output_dir)
        }


def create_hybrid_processor(output_dir: str, prefer_colmap: bool = False) -> HybridVideoProcessor:
    """
    Factory function to create a hybrid processor.
    
    Args:
        output_dir: Output directory for processed data
        prefer_colmap: If True, try COLMAP first instead of ACE-Zero
        
    Returns:
        Configured HybridVideoProcessor
    """
    method = ProcessingMethod.AUTO
    if prefer_colmap:
        method = ProcessingMethod.COLMAP
    
    return HybridVideoProcessor(output_dir=Path(output_dir), method=method)
