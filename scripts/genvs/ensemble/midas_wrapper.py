"""
MiDaS Depth Estimation Wrapper
Uses MiDaS-Small for VRAM efficiency (~200MB)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class MiDaSDepth:
    """Lightweight MiDaS depth estimator."""
    
    def __init__(self, model_type: str = "MiDaS_small", device: str = "cuda"):
        """
        Args:
            model_type: MiDaS model variant 
                       Options: "MiDaS_small" (~50MB), "DPT_Hybrid", "DPT_Large"
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model_type = model_type
        self.model = None
        self.transform = None
        
    def load(self):
        """Lazy load model to save VRAM."""
        if self.model is not None:
            return
            
        print(f"[MiDaS] Loading {self.model_type}...")
        
        try:
            # Load from torch hub
            self.model = torch.hub.load(
                "intel-isl/MiDaS", 
                self.model_type,
                trust_repo=True
            )
            self.model.to(self.device).eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if "DPT" in self.model_type:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
                
            print(f"[MiDaS] Loaded successfully")
        except Exception as e:
            print(f"[MiDaS] Error loading {self.model_type}: {e}")
            print("[MiDaS] Falling back to gradient-based depth estimation")
            self.model = "fallback"
        
    def unload(self):
        """Free VRAM by unloading model."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            print("[MiDaS] Unloaded")
    
    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate relative depth from RGB image.
        
        Args:
            image: RGB image [H, W, 3] uint8 or float
            
        Returns:
            depth: Relative depth map [H, W] float32, normalized to [0, 1]
        """
        self.load()
        
        # Fallback: gradient-based depth approximation
        if self.model == "fallback":
            gray = np.mean(image.astype(np.float32), axis=-1) / 255.0
            # Use edge strength as pseudo-depth
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            depth = np.sqrt(grad_x**2 + grad_y**2)
            depth = 1.0 - (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return depth.astype(np.float32)
        
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply transforms
        input_batch = self.transform(image).to(self.device)
        
        # Inference
        prediction = self.model(input_batch)
        
        # Resize to original resolution
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Normalize to [0, 1]
        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth.astype(np.float32)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.estimate(image)


# Standalone usage
if __name__ == "__main__":
    import cv2
    
    # Test
    midas = MiDaSDepth("DPT_Small")
    
    # Load test image
    img = cv2.imread("test.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Estimate depth
    depth = midas(img)
    
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    
    # Save
    cv2.imwrite("depth_output.png", (depth * 255).astype(np.uint8))
