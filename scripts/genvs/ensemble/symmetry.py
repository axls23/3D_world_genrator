"""
Symmetry Prior
Heuristic for generating back views by mirroring front geometry
Especially useful for buildings and vehicles
"""

import numpy as np
from typing import Tuple, Optional


class SymmetryPrior:
    """
    Generate back-view priors by mirroring front-view data.
    Works best for symmetric objects (buildings, vehicles).
    """
    
    def __init__(self, axis: str = "auto"):
        """
        Args:
            axis: Symmetry axis - "x", "y", "z", or "auto"
        """
        self.axis = axis
        
    def mirror_depth(
        self, 
        front_depth: np.ndarray,
        axis: str = "horizontal"
    ) -> np.ndarray:
        """
        Mirror depth map assuming back is symmetric to front.
        
        Args:
            front_depth: Front view depth [H, W]
            axis: "horizontal" or "vertical"
            
        Returns:
            back_depth: Estimated back depth [H, W]
        """
        if axis == "horizontal":
            # Flip left-right
            back_depth = np.fliplr(front_depth)
        elif axis == "vertical":
            # Flip top-bottom
            back_depth = np.flipud(front_depth)
        else:
            raise ValueError(f"Unknown axis: {axis}")
            
        # Invert depth for back view (what's close becomes far)
        # This is a simple heuristic - actual geometry requires more info
        max_depth = front_depth.max()
        back_depth = max_depth - back_depth + front_depth.min()
        
        return back_depth
    
    def mirror_rgb(
        self, 
        front_rgb: np.ndarray,
        axis: str = "horizontal"
    ) -> np.ndarray:
        """
        Mirror RGB image for back view estimation.
        
        Args:
            front_rgb: Front view RGB [H, W, 3]
            axis: "horizontal" or "vertical"
            
        Returns:
            back_rgb: Estimated back RGB [H, W, 3]
        """
        if axis == "horizontal":
            return np.fliplr(front_rgb)
        elif axis == "vertical":
            return np.flipud(front_rgb)
        else:
            raise ValueError(f"Unknown axis: {axis}")
    
    def estimate_symmetry_axis(
        self, 
        image: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> str:
        """
        Auto-detect likely symmetry axis from image.
        
        Args:
            image: RGB image [H, W, 3]
            depth: Optional depth map [H, W]
            
        Returns:
            axis: "horizontal" or "vertical"
        """
        # Simple heuristic: compare horizontal vs vertical flip similarity
        h, w = image.shape[:2]
        
        # Horizontal symmetry score
        left = image[:, :w//2]
        right = np.fliplr(image[:, w//2:])
        h_score = -np.mean(np.abs(left.astype(float) - right[:, :left.shape[1]].astype(float)))
        
        # Vertical symmetry score  
        top = image[:h//2, :]
        bottom = np.flipud(image[h//2:, :])
        v_score = -np.mean(np.abs(top.astype(float) - bottom[:top.shape[0], :].astype(float)))
        
        return "horizontal" if h_score > v_score else "vertical"
    
    def generate_prior(
        self,
        front_depth: np.ndarray,
        front_rgb: Optional[np.ndarray] = None,
        axis: Optional[str] = None
    ) -> dict:
        """
        Generate back-view prior from front view.
        
        Args:
            front_depth: Front view depth
            front_rgb: Optional front RGB
            axis: Symmetry axis (auto-detect if None)
            
        Returns:
            prior: Dict with estimated back depth and optional RGB
        """
        # Auto-detect axis
        if axis is None:
            if front_rgb is not None:
                axis = self.estimate_symmetry_axis(front_rgb)
            else:
                axis = "horizontal"
                
        prior = {
            "depth": self.mirror_depth(front_depth, axis),
            "axis": axis,
        }
        
        if front_rgb is not None:
            prior["rgb"] = self.mirror_rgb(front_rgb, axis)
            
        return prior
    
    def __call__(
        self, 
        front_depth: np.ndarray,
        front_rgb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get mirrored depth prior."""
        prior = self.generate_prior(front_depth, front_rgb)
        return prior["depth"]


# Standalone test
if __name__ == "__main__":
    sym = SymmetryPrior()
    
    # Test
    depth = np.random.rand(128, 128).astype(np.float32)
    back_depth = sym(depth)
    
    print(f"Back depth shape: {back_depth.shape}")
    print(f"Back depth range: [{back_depth.min():.3f}, {back_depth.max():.3f}]")
