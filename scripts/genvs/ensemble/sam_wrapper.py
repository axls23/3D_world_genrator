"""
SAM Segmentation Wrapper
Uses lightweight Mobile-SAM for segments
Falls back to simple segmentation if SAM unavailable
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple


class SAMSegmenter:
    """
    Lightweight semantic segmentation.
    Uses Mobile-SAM if available, else simple superpixels.
    """
    
    def __init__(self, method: str = "superpixel", device: str = "cuda"):
        """
        Args:
            method: "sam" (needs model) or "superpixel" (no model)
            device: "cuda" or "cpu"
        """
        self.method = method
        self.device = device
        self.model = None
        
    def load(self):
        """Load SAM model if using that method."""
        if self.method == "superpixel":
            print("[SAM] Using superpixel fallback (no model needed)")
            return
            
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            # Try to load Mobile-SAM
            print("[SAM] Loading Mobile-SAM...")
            sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
            sam.to(self.device)
            self.model = SamAutomaticMaskGenerator(sam)
            print("[SAM] Loaded successfully")
        except Exception as e:
            print(f"[SAM] Could not load SAM: {e}")
            print("[SAM] Falling back to superpixels")
            self.method = "superpixel"
            
    def unload(self):
        """Free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            import torch
            torch.cuda.empty_cache()
    
    def segment_superpixels(self, image: np.ndarray, n_segments: int = 100) -> np.ndarray:
        """
        Simple superpixel segmentation using SLIC.
        
        Args:
            image: RGB image [H, W, 3]
            n_segments: Number of segments
            
        Returns:
            segments: Segment labels [H, W] int32
        """
        from skimage.segmentation import slic
        
        segments = slic(
            image, 
            n_segments=n_segments, 
            compactness=10,
            start_label=0
        )
        return segments.astype(np.int32)
    
    def segment_sam(self, image: np.ndarray) -> List[dict]:
        """
        Segment using SAM.
        
        Args:
            image: RGB image [H, W, 3] uint8
            
        Returns:
            masks: List of mask dictionaries
        """
        if self.model is None:
            self.load()
            
        if self.model is None:
            # Fallback
            return self.segment_superpixels(image)
            
        masks = self.model.generate(image)
        return masks
    
    def get_segment_map(self, image: np.ndarray) -> np.ndarray:
        """
        Get segment labels as image.
        
        Args:
            image: RGB image
            
        Returns:
            segment_map: [H, W] with segment IDs
        """
        if self.method == "superpixel":
            return self.segment_superpixels(image)
        else:
            masks = self.segment_sam(image)
            
            # Convert masks to label image
            h, w = image.shape[:2]
            segment_map = np.zeros((h, w), dtype=np.int32)
            
            for i, mask_dict in enumerate(masks):
                mask = mask_dict['segmentation']
                segment_map[mask] = i + 1
                
            return segment_map
    
    def get_segment_colors(self, segment_map: np.ndarray) -> np.ndarray:
        """
        Convert segment map to colorized visualization.
        
        Args:
            segment_map: [H, W] segment labels
            
        Returns:
            colored: [H, W, 3] uint8 visualization
        """
        n_segments = segment_map.max() + 1
        
        # Generate random colors for each segment
        np.random.seed(42)
        colors = np.random.randint(0, 255, (n_segments, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background
        
        colored = colors[segment_map]
        return colored
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Get segment map normalized to [0, 1]."""
        segment_map = self.get_segment_map(image)
        # Normalize
        return (segment_map / (segment_map.max() + 1)).astype(np.float32)

# Alias for compatibility
SAMWrapper = SAMSegmenter


# Standalone test
if __name__ == "__main__":
    segmenter = SAMSegmenter("superpixel")
    
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    segments = segmenter(img)
    
    print(f"Segments shape: {segments.shape}")
    print(f"Segments range: [{segments.min():.3f}, {segments.max():.3f}]")
