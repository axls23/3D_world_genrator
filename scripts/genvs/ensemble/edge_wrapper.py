"""
Edge Detection Wrapper
Uses Canny or Sobel - no external model needed
"""

import numpy as np
import cv2
from typing import Tuple


class EdgeDetector:
    """Lightweight edge detection using classical CV methods."""
    
    def __init__(self, method: str = "canny"):
        """
        Args:
            method: "canny", "sobel", or "laplacian"
        """
        self.method = method
        
    def detect(
        self, 
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detect edges in image.
        
        Args:
            image: RGB or grayscale image
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold
            
        Returns:
            edges: Edge map [H, W] float32, normalized to [0, 1]
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        if self.method == "canny":
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        elif self.method == "sobel":
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        elif self.method == "laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.abs(edges)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Normalize to [0, 1]
        return (edges / 255.0).astype(np.float32)
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return self.detect(image, **kwargs)


# Standalone test
if __name__ == "__main__":
    detector = EdgeDetector("canny")
    
    # Test with random image
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    edges = detector(img)
    
    print(f"Edges shape: {edges.shape}")
    print(f"Edges range: [{edges.min():.3f}, {edges.max():.3f}]")
