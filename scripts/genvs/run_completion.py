"""
GeNVS-Lite: Novel View Completion for 3DGS
Main runner script
"""

# Fix OpenMP duplicate library issue (common in conda)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from typing import List, Dict, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genvs.config import GeNVSConfig
from genvs.ensemble import (
    MiDaSDepth,
    ZoeDepthWrapper,
    NormalEstimator,
    EdgeDetector,
    SAMWrapper,
    SymmetryPrior,
    PriorFusion,
    Zero123Wrapper,
    flush_vram
)
from genvs.diffusion import LightweightDiffusion


class GeNVSLite:
    """
    GeNVS-Lite: Weak ensemble + lightweight diffusion for novel view synthesis.
    Designed for 6GB VRAM RTX 4050.
    
    Now uses ZoeDepth (same as ACE-Zero) for consistent metric depth!
    """
    
    def __init__(self, config: Optional[GeNVSConfig] = None, use_zoedepth: bool = True, use_zero123: bool = False):
        self.config = config or GeNVSConfig()
        self.use_zoedepth = use_zoedepth
        self.use_zero123 = use_zero123
        
        # Depth estimator (ZoeDepth by default for ACE-Zero consistency)
        if use_zoedepth:
            print("[GeNVS] Using ZoeDepth (same as ACE-Zero) for depth estimation")
            self.depth_estimator = ZoeDepthWrapper(device=self.config.DEVICE)
        else:
            print("[GeNVS] Using MiDaS (lightweight) for depth estimation")
            self.depth_estimator = MiDaSDepth(
                model_type=self.config.ensemble.MIDAS_MODEL,
                device=self.config.DEVICE
            )
        
        # Other ensemble components
        self.normals = NormalEstimator(device=self.config.DEVICE)
        self.edges = EdgeDetector(method=self.config.ensemble.EDGE_METHOD)
        self.segments = SAMWrapper(method="superpixel", device=self.config.DEVICE)
        self.symmetry = SymmetryPrior()
        self.fusion = PriorFusion(
            weights={
                "depth": self.config.ensemble.MIDAS_WEIGHT,
                "normals": self.config.ensemble.NORMAL_WEIGHT,
                "edges": self.config.ensemble.EDGE_WEIGHT,
                "segments": self.config.ensemble.SAM_WEIGHT,
                "symmetry": self.config.ensemble.SYMMETRY_WEIGHT,
            },
            device=self.config.DEVICE
        )
        
        # Diffusion
        self.diffusion = LightweightDiffusion(
            model_id=self.config.diffusion.MODEL_ID,
            device=self.config.DEVICE,
            use_fp16=self.config.diffusion.USE_FP16
        )
        
        # Zero123 for multi-view synthesis (optional, replaces symmetry)
        self.zero123 = None
        if use_zero123:
            print("[GeNVS] Zero123 mode enabled (better for asymmetric objects)")
            self.zero123 = Zero123Wrapper(device=self.config.DEVICE, use_fp16=True)
        
    def extract_priors(self, image: np.ndarray, depth: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract all priors from front view image.
        
        Args:
            image: RGB image [H, W, 3] uint8
            depth: Optional known depth [H, W]
            
        Returns:
            priors: Dict of prior maps
        """
        print("[GeNVS] Extracting priors...")
        
        priors = {}
        
        # 1. Depth (use known or estimate)
        if depth is not None:
            priors["depth"] = depth.astype(np.float32)
            print("  - Using provided depth")
        else:
            priors["depth"] = self.depth_estimator.estimate(image)
            depth_type = "ZoeDepth" if self.use_zoedepth else "MiDaS"
            print(f"  - Estimated depth via {depth_type}")
        
        # 2. Normals (from depth)
        priors["normals"] = self.normals(priors["depth"], is_depth=True)
        print("  - Computed normals from depth")
        
        # 3. Edges
        priors["edges"] = self.edges(image)
        print("  - Detected edges")
        
        # 4. Segments
        priors["segments"] = self.segments(image)
        print("  - Generated segments")
        
        # 5. Symmetry prior (estimated back depth)
        priors["symmetry"] = self.symmetry(priors["depth"], image)
        print("  - Generated symmetry prior")
        
        return priors
    
    def generate_novel_views(
        self,
        front_image: np.ndarray,
        front_depth: Optional[np.ndarray] = None,
        num_views: int = 20,
        init_images: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Generate novel back/side views from front view.
        
        Args:
            front_image: Front RGB [H, W, 3]
            front_depth: Optional front depth [H, W]
            num_views: Number of views to generate
            init_images: Optional list of images to use as initialization (Reference)
                         If provided, len(init_images) must match num_views.
                         This enables Autoregressive/Iterative refinement.
            
        Returns:
            views: List of generated RGB images
        """
        print(f"[GeNVS] Generating {num_views} novel views...")
        
        # Extract priors
        priors = self.extract_priors(front_image, front_depth)
        
        # Fuse priors
        fused = self.fusion(priors, return_tensor=False)
        print(f"[GeNVS] Fused prior shape: {fused.shape}")
        
        views = []
        
        for i in range(num_views):
            print(f"[GeNVS] Generating view {i+1}/{num_views}...")
            
            # Vary the symmetry influence for different angles
            angle_factor = i / num_views  # 0 to 1
            
            # Modify fused prior for this angle
            # More symmetry influence for back views
            modified_fused = fused.copy()
            
            # Create coarse initialization for this view
            # If we have an explicit init image (e.g. from 3DGS render), use it
            if init_images is not None and i < len(init_images):
                coarse_init = init_images[i]
                print(f"  - Using provided init image for view {i}")
            else:
                # Fallback: Simple heuristic: flip image for "back" views
                # We assume the sequence covers a semi-circle or full circle
                # For now, just apply horizontal flip to simulate back view structure
                coarse_init = self.symmetry.mirror_rgb(front_image, axis="horizontal")
            
            # Generate
            # We use the coarse_init as the starting point (reference) for img2img
            # Strength 0.65 means "keep 35% of structure, change 65%"
            novel_view = self.diffusion(
                modified_fused,
                reference=coarse_init,
                prompt="same scene, back view, consistent lighting, photorealistic",
                num_steps=self.config.diffusion.NUM_STEPS,
                guidance_scale=self.config.diffusion.GUIDANCE_SCALE,
                strength=0.65 
            )
            
            # Grounding: Enforce strict color consistency via Histogram Matching
            # This ensures the back view statistics match the front view exacty
            try:
                from skimage.exposure import match_histograms
                novel_view = match_histograms(novel_view, front_image, channel_axis=-1)
                novel_view = novel_view.astype(np.uint8)
            except ImportError:
                print("[GeNVS] skimage not found, skipping histogram matching")

            views.append(novel_view)
            
            # Clear cache periodically
            if (i + 1) % self.config.CLEAR_CACHE_FREQUENCY == 0:
                torch.cuda.empty_cache()
        
        return views
    
    def complete_3dgs(
        self,
        input_ply: Path,
        front_images_dir: Path,
        output_dir: Path
    ) -> Path:
        """
        Complete a partial 3DGS model by generating back views.
        
        Args:
            input_ply: Path to partial PLY
            front_images_dir: Directory with front view images
            output_dir: Where to save novel views
            
        Returns:
            output_dir: Path with generated views
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load front images
        front_images = sorted(Path(front_images_dir).glob("*.png"))
        if not front_images:
            front_images = sorted(Path(front_images_dir).glob("*.jpg"))
            
        if not front_images:
            raise ValueError(f"No images found in {front_images_dir}")
        
        print(f"[GeNVS] Found {len(front_images)} front images")
        
        # Use best (middle) front image as reference
        ref_idx = len(front_images) // 2
        ref_img = cv2.imread(str(front_images[ref_idx]))
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        
        # Generate novel views
        novel_views = self.generate_novel_views(
            ref_img,
            num_views=self.config.novel_view.NUM_VIEWS_BACK
        )
        
        # Save
        for i, view in enumerate(novel_views):
            out_path = output_dir / f"novel_view_{i:04d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
            
        print(f"[GeNVS] Saved {len(novel_views)} novel views to {output_dir}")
        
        return output_dir
    
    def cleanup(self):
        """Unload all models to free VRAM."""
        if hasattr(self, 'depth_estimator') and hasattr(self.depth_estimator, 'unload'):
            self.depth_estimator.unload()
        if hasattr(self, 'diffusion') and hasattr(self.diffusion, 'unload'):
            self.diffusion.unload()
        if self.zero123 is not None:
            self.zero123.unload()
        flush_vram()
        print("[GeNVS] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="GeNVS-Lite: Novel View Completion")
    parser.add_argument("--input-ply", type=str, help="Input partial PLY file")
    parser.add_argument("--images", type=str, required=True, help="Front images directory")
    parser.add_argument("--output", type=str, default="genvs_output", help="Output directory")
    parser.add_argument("--num-views", type=int, default=20, help="Number of novel views")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--zero123", action="store_true", 
                        help="Use Zero123 for multi-view synthesis (better for asymmetric objects)")
    parser.add_argument("--colmap-dir", type=str, help="COLMAP dir for blind spot detection")
    
    args = parser.parse_args()
    
    # Config
    config = GeNVSConfig()
    config.DEVICE = args.device
    config.novel_view.NUM_VIEWS_BACK = args.num_views
    
    # Run
    genvs = GeNVSLite(config, use_zero123=args.zero123)
    
    try:
        # If Zero123 mode and colmap_dir provided, use blind spot detection
        if args.zero123 and args.colmap_dir:
            from genvs.blind_spot_detector import BlindSpotDetector
            detector = BlindSpotDetector()
            angles = detector.get_zero123_angles(args.colmap_dir, args.num_views)
            print(f"[GeNVS] Detected blind spots: {angles}")
            # TODO: Pass angles to Zero123 generation
        
        genvs.complete_3dgs(
            input_ply=Path(args.input_ply) if args.input_ply else None,
            front_images_dir=Path(args.images),
            output_dir=Path(args.output)
        )
    finally:
        genvs.cleanup()


if __name__ == "__main__":
    main()
