"""
Verify MiDaS Depth Estimation Quality
Generates side-by-side visualization of input images and estimated depth maps.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from ensemble.midas_wrapper import MiDaSDepth


def visualize_depth(image_path: str, output_dir: str, device: str = "cuda"):
    """Generate and save depth visualization for an image."""
    
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Estimate depth
    midas = MiDaSDepth(device=device)
    depth = midas.estimate(img_rgb)
    midas.unload()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    im = axes[1].imshow(depth, cmap="plasma")
    axes[1].set_title("MiDaS Depth (Relative)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Save
    output_path = Path(output_dir) / f"depth_{Path(image_path).stem}.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[Depth] Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Verify MiDaS Depth Estimation")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="depth_verify", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    visualize_depth(args.image, args.output, args.device)


if __name__ == "__main__":
    main()
