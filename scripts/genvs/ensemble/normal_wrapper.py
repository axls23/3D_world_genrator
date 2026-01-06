"""
Normal Estimation Wrapper
Estimates surface normals from RGB image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class NormalEstimator:
    """
    Lightweight normal estimation using gradient-based method.
    Falls back to learned model if available.
    """
    
    def __init__(self, method: str = "gradient", device: str = "cuda"):
        """
        Args:
            method: "gradient" (fast, no model) or "learned" (needs model)
            device: "cuda" or "cpu"
        """
        self.method = method
        self.device = device
        self.model = None
        
    def load(self):
        """Load learned model if using that method."""
        if self.method == "gradient":
            return  # No model needed
            
        # TODO: Load omnidata-tiny if needed
        print("[Normals] Using gradient-based estimation (no model needed)")
        
    def unload(self):
        """Free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
    
    def estimate_from_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Estimate normals from depth map using gradients.
        
        Args:
            depth: Depth map [H, W] float32
            
        Returns:
            normals: Normal map [H, W, 3] float32, normalized to [-1, 1]
        """
        # Compute gradients
        dz_dx = np.gradient(depth, axis=1)
        dz_dy = np.gradient(depth, axis=0)
        
        # Normal from gradients: n = (-dz/dx, -dz/dy, 1) normalized
        normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
        
        # Normalize
        magnitude = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
        normals = normals / magnitude
        
        return normals.astype(np.float32)
    
    def estimate_from_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate normals from RGB using intensity gradients.
        Less accurate than depth-based but works without depth.
        
        Args:
            image: RGB image [H, W, 3]
            
        Returns:
            normals: Normal map [H, W, 3] float32
        """
        # Convert to grayscale
        if image.ndim == 3:
            gray = np.mean(image.astype(np.float32), axis=-1) / 255.0
        else:
            gray = image.astype(np.float32)
            
        return self.estimate_from_depth(gray)
    
    def __call__(self, image_or_depth: np.ndarray, is_depth: bool = False) -> np.ndarray:
        """
        Estimate normals.
        
        Args:
            image_or_depth: RGB image [H,W,3] or depth map [H,W]
            is_depth: True if input is depth, False if RGB
        """
        if is_depth:
            return self.estimate_from_depth(image_or_depth)
        else:
            return self.estimate_from_rgb(image_or_depth)


# Standalone test
if __name__ == "__main__":
    import cv2
    
    estimator = NormalEstimator()
    
    # Test with depth
    depth = np.random.rand(256, 256).astype(np.float32)
    normals = estimator(depth, is_depth=True)
    
    print(f"Normals shape: {normals.shape}")
    print(f"Normals range: [{normals.min():.3f}, {normals.max():.3f}]")
    
    # Visualize (map [-1,1] to [0,255])
    normals_vis = ((normals + 1) / 2 * 255).astype(np.uint8)
    cv2.imwrite("normals_output.png", normals_vis)
