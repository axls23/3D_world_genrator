"""
Depth Pre-computation for VRAM Optimization

This script pre-computes depth maps for all images using configurable models:
- zoedepth: Metric depth (best quality, ~200ms/frame)
- depth_anything: Relative depth (fast, ~30ms/frame)
- midas: Relative depth (lightweight, ~100ms/frame)

Usage:
    python precompute_depths.py <images_dir> --model depth_anything
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def precompute_depths(
    images_dir: Path, 
    output_dir: Path = None,
    model: str = "zoedepth"
) -> int:
    """
    Pre-compute depth maps for all images in a directory.
    
    Args:
        images_dir: Directory containing JPG images
        output_dir: Where to save depth NPY files (defaults to images_dir)
        model: Depth model to use ("zoedepth", "depth_anything", "midas")
        
    Returns:
        Number of depth maps computed
    """
    # Import unified depth estimator from hypersplat
    from hypersplat.pipeline.wrappers.depth import DepthEstimator
    
    if output_dir is None:
        output_dir = images_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(images_dir.glob("*.png"))
    
    if not image_files:
        logger.warning(f"No images found in {images_dir}")
        return 0
        
    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Using depth model: {model}")
    
    # Load depth model ONCE
    depth_estimator = DepthEstimator(model=model, device="cuda")
    
    # Process all images
    count = 0
    import cv2
    
    for i, img_path in enumerate(image_files):
        depth_path = output_dir / f"{img_path.stem}.npy"
        
        # Skip if already cached
        if depth_path.exists():
            logger.debug(f"Skipping {img_path.name} (cached)")
            count += 1
            continue
            
        try:
            # Load image
            image_rgb = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
            
            # Estimate depth
            depth = depth_estimator(image_rgb)
            
            # Save to disk
            np.save(depth_path, depth)
            count += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            logger.warning(f"Failed to process {img_path.name}: {e}")
    
    logger.info(f"Computed {count} depth maps using {model}")
    
    # Unload model to free VRAM
    depth_estimator.unload()
    
    # Verify VRAM freed
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"VRAM after unload: {allocated:.2f} GB")
    
    return count


def main():
    parser = argparse.ArgumentParser(description="Pre-compute depth maps")
    parser.add_argument("images_dir", type=Path, help="Directory with images")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--model", type=str, default="zoedepth",
        choices=["zoedepth", "depth_anything", "midas"],
        help="Depth model: zoedepth (quality), depth_anything (fast), midas (light)"
    )
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all models")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.benchmark:
        from depth_estimator import DepthEstimator
        import cv2
        
        # Load first image for benchmark
        images = sorted(args.images_dir.glob("*.jpg"))
        if not images:
            images = sorted(args.images_dir.glob("*.png"))
        if images:
            image = cv2.imread(str(images[0]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            DepthEstimator.benchmark(image)
        return
    
    count = precompute_depths(args.images_dir, args.output_dir, args.model)
    print(f"Done! Computed {count} depth maps using {args.model}.")


if __name__ == "__main__":
    main()

