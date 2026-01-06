
"""
GeNVS-Lite: Quality Evaluation
Computes semantic consistency using CLIP between front and generated back views.
"""

import sys
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import argparse
import numpy as np

def calculate_clip_score(image_path1, image_path2, model, processor, device):
    """Calculate cosine similarity between two images."""
    try:
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)
        
        inputs = processor(images=[image1, image2], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        
        # Normalize
        features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        similarity = (features[0] @ features[1].T).item()
        return similarity
    except Exception as e:
        print(f"Error processing {image_path1} or {image_path2}: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate GeNVS Generation Quality")
    parser.add_argument("--front", type=str, required=True, help="Path to reference front image")
    parser.add_argument("--back", type=str, required=True, help="Path to generated back image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    print(f"[Eval] Loading CLIP model on {args.device}...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"[Eval] Error loading CLIP: {e}")
        return

    score = calculate_clip_score(args.front, args.back, model, processor, args.device)
    
    print(f"\nResults:")
    print(f"----------------------------------------")
    print(f"Semantic Consistency Score (CLIP): {score:.4f}")
    print(f"----------------------------------------")
    print(f"Interpretation:")
    print(f" > 0.85: Excellent (Very consistent)")
    print(f" > 0.75: Good (Consistent theme)")
    print(f" < 0.60: Poor (Likely hallucination)")

if __name__ == "__main__":
    main()
