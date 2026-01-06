"""
ZoeDepth Wrapper for GeNVS-Lite Ensemble

Uses the same ZoeDepth model as ACE-Zero for metric depth estimation.
Provides higher quality depth than MiDaS but uses more VRAM (~2-3GB).
"""

import numpy as np
import torch
from typing import Optional


class ZoeDepthWrapper:
    """
    ZoeDepth depth estimation wrapper.
    Uses ZoeD_NK (best performing variant) from isl-org/ZoeDepth.
    
    Advantages over MiDaS:
    - Metric depth output (actual meters)
    - Better depth quality
    - Same model used by ACE-Zero for consistency
    
    Tradeoffs:
    - Higher VRAM (~2-3GB vs ~300MB for MiDaS)
    - Slower inference
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_variant: str = "ZoeD_NK"  # Options: ZoeD_N, ZoeD_K, ZoeD_NK
    ):
        self.device = device
        self.model_variant = model_variant
        self.model = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy load ZoeDepth model."""
        if self._loaded:
            return
        
        print(f"[ZoeDepth] Loading {self.model_variant}...")
        try:
            # ZoeDepth requires MiDaS as dependency
            torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", trust_repo=True)
            
            # Load ZoeDepth
            repo = "isl-org/ZoeDepth"
            self.model = torch.hub.load(
                repo, 
                self.model_variant, 
                pretrained=True, 
                trust_repo=True
            )
            self.model.eval().to(self.device)
            self._loaded = True
            print(f"[ZoeDepth] Model loaded successfully")
            
        except Exception as e:
            print(f"[ZoeDepth] Failed to load model: {e}")
            print("[ZoeDepth] Falling back to gradient-based depth estimation")
            self._loaded = False
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from an RGB image.
        
        Args:
            image: RGB image as numpy array (H, W, 3), values 0-255 or 0-1
            
        Returns:
            depth: Metric depth map (H, W) in meters
        """
        self._load_model()
        
        if not self._loaded or self.model is None:
            # Fallback to gradient-based estimation
            return self._gradient_depth(image)
        
        # Normalize image if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Convert to PIL Image for ZoeDepth
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray((image * 255).astype(np.uint8))
        
        with torch.no_grad():
            depth = self.model.infer_pil(pil_image)
        
        # ZoeDepth returns metric depth in meters
        return depth.cpu().numpy()
    
    def _gradient_depth(self, image: np.ndarray) -> np.ndarray:
        """Fallback gradient-based depth estimation."""
        if image.max() > 1.0:
            gray = np.mean(image, axis=2) / 255.0
        else:
            gray = np.mean(image, axis=2)
        
        # Simple edge-based depth heuristic
        from scipy import ndimage
        gx = ndimage.sobel(gray, axis=1)
        gy = ndimage.sobel(gray, axis=0)
        depth = 1.0 / (np.sqrt(gx**2 + gy**2) + 0.1)
        
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)
    
    def unload(self):
        """Unload model to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[ZoeDepth] Unloaded")


def verify_zoedepth(image_path: str, output_path: str = "zoedepth_verify.png"):
    """Quick verification of ZoeDepth output."""
    import cv2
    import matplotlib.pyplot as plt
    
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Estimate depth
    zoe = ZoeDepthWrapper(device="cuda")
    depth = zoe.estimate(img_rgb)
    zoe.unload()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    im = axes[1].imshow(depth, cmap="plasma")
    axes[1].set_title(f"ZoeDepth (Metric, range: {depth.min():.2f}m - {depth.max():.2f}m)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (meters)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[ZoeDepth] Saved verification to {output_path}")
    return depth


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify ZoeDepth")
    parser.add_argument("--image", type=str, required=True, help="Input image")
    parser.add_argument("--output", type=str, default="zoedepth_verify.png", help="Output path")
    args = parser.parse_args()
    
    verify_zoedepth(args.image, args.output)
