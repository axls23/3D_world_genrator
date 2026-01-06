"""
Prior Fusion Layer
Combines outputs from all weak models into unified prior
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class PriorFusion:
    """
    Fuses multiple weak priors into a unified conditioning signal.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        output_channels: int = 16,
        device: str = "cuda"
    ):
        """
        Args:
            weights: Dict of prior weights (depth, normals, edges, segments, symmetry)
            output_channels: Number of output channels after fusion
            device: "cuda" or "cpu"
        """
        self.weights = weights or {
            "depth": 0.25,
            "normals": 0.20,
            "edges": 0.10,
            "segments": 0.20,
            "symmetry": 0.25,
        }
        self.output_channels = output_channels
        self.device = device
        
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    def to_channels(self, x: np.ndarray, n_channels: int = 1) -> np.ndarray:
        """
        Expand to specified number of channels.
        
        Args:
            x: Input [H, W] or [H, W, C]
            n_channels: Target channels
            
        Returns:
            output: [H, W, n_channels]
        """
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
            
        c = x.shape[2]
        if c == n_channels:
            return x
        elif c < n_channels:
            # Repeat channels
            repeats = (n_channels + c - 1) // c
            x = np.tile(x, (1, 1, repeats))[:, :, :n_channels]
        else:
            # Average pool channels
            x = x[:, :, :n_channels]
            
        return x
    
    def fuse(
        self,
        priors: Dict[str, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Fuse multiple priors into unified feature map.
        
        Args:
            priors: Dict with keys: depth, normals, edges, segments, symmetry
                   Each value is [H, W] or [H, W, C] numpy array
            target_size: Optional (H, W) to resize all priors
            
        Returns:
            fused: [H, W, output_channels] numpy array
        """
        fused_parts = []
        
        # Determine target size
        if target_size is None:
            # Use size of first non-None prior
            for key, arr in priors.items():
                if arr is not None:
                    target_size = arr.shape[:2]
                    break
                    
        if target_size is None:
            raise ValueError("No valid priors provided")
            
        H, W = target_size
        
        # Channels per prior
        channels_per = max(1, self.output_channels // len(self.weights))
        
        for key, weight in self.weights.items():
            if key not in priors or priors[key] is None:
                # Zero fill if prior not available
                part = np.zeros((H, W, channels_per), dtype=np.float32)
            else:
                arr = priors[key]
                
                # Normalize
                arr = self.normalize(arr)
                
                # Resize if needed
                if arr.shape[:2] != (H, W):
                    import cv2
                    if arr.ndim == 3:
                        arr = cv2.resize(arr, (W, H))
                    else:
                        arr = cv2.resize(arr, (W, H))
                
                # Expand to channels
                part = self.to_channels(arr, channels_per)
                
                # Apply weight
                part = part * weight
                
            fused_parts.append(part)
        
        # Concatenate
        fused = np.concatenate(fused_parts, axis=2)
        
        # Ensure exact output channels
        if fused.shape[2] > self.output_channels:
            fused = fused[:, :, :self.output_channels]
        elif fused.shape[2] < self.output_channels:
            padding = np.zeros((H, W, self.output_channels - fused.shape[2]), dtype=np.float32)
            fused = np.concatenate([fused, padding], axis=2)
            
        return fused.astype(np.float32)
    
    def to_tensor(self, fused: np.ndarray) -> torch.Tensor:
        """Convert fused array to PyTorch tensor [1, C, H, W]."""
        # [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(fused).permute(2, 0, 1)
        # Add batch dim
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)
    
    def __call__(
        self,
        priors: Dict[str, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        return_tensor: bool = True
    ):
        """
        Fuse priors and optionally return as tensor.
        
        Args:
            priors: Dict of prior arrays
            target_size: Optional target resolution
            return_tensor: If True, return PyTorch tensor
            
        Returns:
            fused: [1, C, H, W] tensor or [H, W, C] numpy array
        """
        fused = self.fuse(priors, target_size)
        
        if return_tensor:
            return self.to_tensor(fused)
        return fused


# Standalone test
if __name__ == "__main__":
    fusion = PriorFusion(output_channels=16)
    
    H, W = 128, 128
    priors = {
        "depth": np.random.rand(H, W).astype(np.float32),
        "normals": np.random.rand(H, W, 3).astype(np.float32),
        "edges": np.random.rand(H, W).astype(np.float32),
        "segments": np.random.rand(H, W).astype(np.float32),
        "symmetry": np.random.rand(H, W).astype(np.float32),
    }
    
    fused = fusion(priors, return_tensor=False)
    print(f"Fused shape: {fused.shape}")  # [128, 128, 16]
    print(f"Fused range: [{fused.min():.3f}, {fused.max():.3f}]")
    
    tensor = fusion(priors, return_tensor=True)
    print(f"Tensor shape: {tensor.shape}")  # [1, 16, 128, 128]
