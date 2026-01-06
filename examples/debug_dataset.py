
import sys
import os
from pathlib import Path
import cv2
import torch

# Add gsplat root to path to allow imports from examples
current_dir = Path(__file__).parent.absolute()
root_dir = current_dir.parent
sys.path.append(str(root_dir))
sys.path.append(str(current_dir)) # For dynamic_trainer import if in same dir

from examples.dynamic_trainer import Config, DynamicDataset
from examples.datasets.colmap import Dataset, Parser

def test_dataset():
    data_dir = "data/acezero"
    factor = 4
    
    print(f"Testing dataset with dir={data_dir}, factor={factor}")
    
    parser = Parser(
        data_dir=data_dir,
        factor=factor,
        normalize=True,
    )
    
    base_trainset = Dataset(parser, split="train")
    
    # Enable masks
    trainset = DynamicDataset(base_trainset, load_masks=True)
    
    print(f"Dataset length: {len(trainset)}")
    
    for i in range(min(5, len(trainset))):
        data = trainset[i]
        
        img = data["image"]
        mask = data.get("mask", None)
        
        print(f"Item {i}: Image {img.shape}, Mask {mask.shape if mask is not None else 'None'}")
        
        if mask is not None:
            # Check dimensions
            # Image: [3, H, W]
            # Mask: [1, H, W]
            if img.shape[1:] != mask.shape[1:]:
                print(f"MISMATCH! {img.shape} vs {mask.shape}")
            else:
                print("Shapes match.")
                
            # Check values
            print(f"Mask range: {mask.min()} - {mask.max()}")

if __name__ == "__main__":
    test_dataset()
