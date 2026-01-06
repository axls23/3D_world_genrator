
import argparse
import os
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pre-process images with YOLO segmentation for 4DGS.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--classes", type=int, nargs="+", default=[2], help="COCO class IDs to segment (2=car, 3=motorcycle, 5=bus, 7=truck)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--img_suffix", type=str, default=".png", help="Image suffix")
    parser.add_argument("--debug", action="store_true", help="Save debug visualization")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    
    # Check if we should use downscaled images by default if images doesn't exist
    if not images_dir.exists():
        # Try to find images_4 or images_2
        potentials = sorted(list(data_dir.glob("images_*")))
        if potentials:
            images_dir = potentials[0]
            print(f"Using images dir: {images_dir}")
        else:
            print(f"Error: No images directory found in {data_dir}")
            return

    masks_dir = data_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    
    vis_dir = data_dir / "masks_vis"
    if args.debug:
        vis_dir.mkdir(exist_ok=True)

    print(f"Loading YOLOv11 segmentation model...")
    # Use standard yolo11n-seg or yolov8n-seg if 11 not available? 
    # Ultralytics auto-downloads data.
    try:
        model = YOLO("yolo11n-seg.pt")
    except Exception:
        print("YOLO11 not found, trying YOLOv8...")
        model = YOLO("yolov8n-seg.pt")
        
    image_paths = sorted(list(images_dir.glob(f"*{args.img_suffix}")) + list(images_dir.glob("*.jpg")))
    
    print(f"Processing {len(image_paths)} images from {images_dir}...")
    
    for img_path in tqdm(image_paths):
        try:
            # Predict (Force CPU to avoid torchvision NMS cuda mismatch)
            results = model.predict(str(img_path), conf=args.conf, verbose=False, classes=args.classes, device="cpu")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Create empty mask
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        final_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Aggregate masks
        res = results[0]
        if res.masks is not None:
            # masks.data is [N, H, W] tensor (scaled usually? No, YOLO returns resized or original)
            # Ultralytics handles scaling if using standard predict.
            # masks.xy is polygon. masks.data is bitmaps.
            
            # Use bitmaps
            # Note: masks.data might be lower resolution.
            # We used retinanet/masks logic.
            # Safer to iterate segments?
            
            masks_tensor = res.masks.data # [N, h, w] -> might be smaller than image
            
            # Resize masks to full image size if needed
            if masks_tensor.shape[1:] != (H, W):
                masks_tensor = torch.nn.functional.interpolate(
                    masks_tensor.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(1)
            
            masks_np = masks_tensor.cpu().numpy()
            
            # Combine all detected instances
            combined = np.any(masks_np > 0.5, axis=0).astype(np.uint8) * 255
            final_mask = combined
            
        # Save
        mask_path = masks_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(mask_path), final_mask)
        
        if args.debug:
            # Save overlay
            vis = img.copy()
            vis[final_mask > 0] = vis[final_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            cv2.imwrite(str(vis_dir / img_path.name), vis)

    print(f"Done. Masks saved to {masks_dir}")

if __name__ == "__main__":
    main()
