#!/usr/bin/env python3
"""
Quick test script to verify 3DGUT training setup
"""
import sys
import os

# Add gsplat to Python path
sys.path.insert(0, r'C:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat')

print("="*80)
print("3DGUT Training Setup Verification")
print("="*80)

# Check imports
print("\n1. Checking imports...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ PyTorch import failed: {e}")

try:
    import gsplat
    print(f"   ✓ gsplat {gsplat.__version__}")
except Exception as e:
    print(f"   ✗ gsplat import failed: {e}")

try:
    from examples.datasets.colmap import Dataset
    print(f"   ✓ COLMAP dataset loader")
except Exception as e:
    print(f"   ✗ COLMAP dataset import failed: {e}")

# Check dataset
print("\n2. Checking dataset...")
dataset_path = r"C:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat\TEST_OUTPUT_RUN2"
if os.path.exists(dataset_path):
    print(f"   ✓ Dataset path exists: {dataset_path}")
    
    # Check for required files
    sparse_path = os.path.join(dataset_path, "sparse", "0")
    images_path = os.path.join(dataset_path, "filtered_images")
    
    if os.path.exists(sparse_path):
        print(f"   ✓ Sparse reconstruction found")
        files = os.listdir(sparse_path)
        print(f"     Files: {', '.join(files[:5])}")
    else:
        print(f"   ✗ Sparse reconstruction not found")
    
    if os.path.exists(images_path):
        image_count = len([f for f in os.listdir(images_path) if f.endswith('.jpg')])
        print(f"   ✓ Images found: {image_count} images")
    else:
        print(f"   ✗ Images directory not found")
else:
    print(f"   ✗ Dataset path not found: {dataset_path}")

# Check results directory
print("\n3. Checking results directory...")
results_path = r"C:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat\RESULTS_3DGUT_TEST"
if os.path.exists(results_path):
    print(f"   ✓ Results directory exists")
    for subdir in ['ckpts', 'stats', 'renders', 'ply', 'tb']:
        subdir_path = os.path.join(results_path, subdir)
        if os.path.exists(subdir_path):
            file_count = len(os.listdir(subdir_path))
            print(f"     - {subdir}/: {file_count} files")
else:
    print(f"   ✗ Results directory not found")

print("\n" + "="*80)
print("Verification Complete")
print("="*80)

# Try to load dataset
print("\n4. Attempting to load dataset...")
try:
    from examples.datasets.colmap import Parser, Dataset
    
    parser = Parser(
        data_dir=dataset_path,
        factor=2,
        normalize=True,
        test_every=8
    )
    dataset = Dataset(parser, split="train")
    
    print(f"   ✓ Dataset loaded successfully!")
    print(f"   - Number of training images: {len(dataset)}")
    print(f"   - Image resolution: {dataset.parser.image_names[0] if dataset.parser.image_names else 'N/A'}")
    
except Exception as e:
    print(f"   ✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Setup verification complete!")









