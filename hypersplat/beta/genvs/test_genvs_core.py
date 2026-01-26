"""
GeNVS-Core Component Test Suite
Tests all components: Encoder, FeatureVolume, Rendering, UNet, Pipeline

Run: python scripts/genvs_core/test_genvs_core.py --data-dir demo/test_run_genvs_gpu_max_seed2089
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Fix Windows console encoding
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test 1: Verify all modules import correctly."""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    try:
        from hypersplat.beta.genvs import GeometryEncoder, FrustumFeatureVolume, NeuralVolumeRenderer, DiffusionUNet
        from hypersplat.beta.genvs.pipeline import GeNVSPipeline
        print("[PASS] All genvs_core modules imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_encoder(device='cuda'):
    """Test 2: GeometryEncoder forward pass."""
    print("\n" + "="*60)
    print("TEST 2: GeometryEncoder")
    print("="*60)
    
    from hypersplat.beta.genvs import GeometryEncoder
    
    try:
        encoder = GeometryEncoder(c_feat=16, depth_planes=64).to(device)
        print(f"  * Created encoder on {device}")
        
        # Test input
        B, C, H, W = 1, 3, 256, 256
        x = torch.randn(B, C, H, W, device=device)
        print(f"  * Input shape: {x.shape}")
        
        # Forward
        t0 = time.time()
        with torch.no_grad():
            out = encoder(x)
        t1 = time.time()
        
        print(f"  * Output shape: {out.shape}")
        print(f"  * Expected: [1, 16, 64, 256, 256]")
        print(f"  * Time: {(t1-t0)*1000:.1f}ms")
        
        expected = (B, 16, 64, H, W)
        if out.shape == expected:
            print("[PASS] Encoder output shape correct!")
            return True
        else:
            print(f"[FAIL] Shape mismatch: got {out.shape}, expected {expected}")
            return False
    except Exception as e:
        print(f"[FAIL] Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_volume(device='cuda'):
    """Test 3: FrustumFeatureVolume."""
    print("\n" + "="*60)
    print("TEST 3: FrustumFeatureVolume")
    print("="*60)
    
    from hypersplat.beta.genvs import FrustumFeatureVolume
    
    try:
        volume = FrustumFeatureVolume(c_feat=16, depth_planes=64).to(device)
        print(f"  * Created volume on {device}")
        print(f"  * Depth bins range: [{volume.depth_bins.min():.2f}, {volume.depth_bins.max():.2f}]")
        
        # Test passthrough
        features = torch.randn(1, 16, 64, 128, 128, device=device)
        out = volume(features)
        
        assert out.shape == features.shape, "Passthrough shape mismatch"
        print("[PASS] FrustumFeatureVolume working correctly!")
        return True
    except Exception as e:
        print(f"[FAIL] FeatureVolume test failed: {e}")
        return False

def test_renderer(device='cuda'):
    """Test 4: NeuralVolumeRenderer."""
    print("\n" + "="*60)
    print("TEST 4: NeuralVolumeRenderer")
    print("="*60)
    
    from hypersplat.beta.genvs import NeuralVolumeRenderer
    
    try:
        # Corrected init arg
        renderer = NeuralVolumeRenderer(c_feat=16).to(device)
        print(f"  * Created renderer on {device}")
        
        B, num_src = 2, 1
        C, D, H, W = 16, 64, 128, 128
        
        # Mock inputs
        volume = torch.randn(B, num_src, C, D, H, W, device=device)
        source_poses = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, num_src, 4, 4).to(device)
        source_Ks = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, num_src, 3, 3).to(device)
        target_poses = torch.eye(4).unsqueeze(0).expand(B, 4, 4).to(device)
        target_Ks = torch.eye(3).unsqueeze(0).expand(B, 3, 3).to(device)
        
        print(f"  * Volume shape: {volume.shape}")
        
        t0 = time.time()
        with torch.no_grad():
            out, depth = renderer(volume, source_poses, source_Ks, target_poses, target_Ks)
        t1 = time.time()
        
        print(f"  * Output shape: {out.shape}")
        print(f"  * Depth shape: {depth.shape}")
        print(f"  * Expected: [2, 16, 128, 128]")
        print(f"  * Time: {(t1-t0)*1000:.1f}ms")
        
        if out.shape == (B, 16, H, W):
            print("[PASS] Renderer output shape correct!")
            return True
        else:
            print(f"[FAIL] Shape mismatch")
            return False
    except Exception as e:
        print(f"[FAIL] Renderer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unet(device='cuda'):
    """Test 5: DiffusionUNet."""
    print("\n" + "="*60)
    print("TEST 5: DiffusionUNet")
    print("="*60)
    
    from hypersplat.beta.genvs import DiffusionUNet
    
    try:
        # Corrected init args for UNet
        # in_channels = 3 (RGB) + 16 (Feature) = 19
        unet = DiffusionUNet(in_channels=19, out_channels=3, model_channels=64).to(device)
        print(f"  * Created UNet on {device}")
        print(f"  * Params: {sum(p.numel() for p in unet.parameters()) / 1e6:.2f}M")
        
        B, H, W = 1, 128, 128
        x_t = torch.randn(B, 3, H, W, device=device)
        t = torch.tensor([500], device=device).long()
        cond = torch.randn(B, 16, H, W, device=device)
        
        t0 = time.time()
        with torch.no_grad():
            pass # No forward pass in this test func yet? Ah the previous file had it.
            # Adding forward test logic back
            out = unet(x_t, t, cond)

        t1 = time.time()
        
        print(f"  * Input x_t: {x_t.shape}")
        print(f"  * Input cond: {cond.shape}")
        print(f"  * Output: {out.shape}")
        print(f"  * Time: {(t1-t0)*1000:.1f}ms")
        
        if out.shape == x_t.shape:
            print("[PASS] UNet output shape correct!")
            return True
        else:
            print(f"[FAIL] Shape mismatch")
            return False
    except Exception as e:
        print(f"[FAIL] UNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_init(device='cuda'):
    """Test 6: Full Pipeline initialization."""
    print("\n" + "="*60)
    print("TEST 6: Pipeline Initialization")
    print("="*60)
    
    from hypersplat.beta.genvs.pipeline import GeNVSPipeline
    
    try:
        t0 = time.time()
        pipeline = GeNVSPipeline(device=device)
        t1 = time.time()
        
        print(f"  * Pipeline initialized in {(t1-t0)*1000:.1f}ms")
        print(f"  * Encoder: {type(pipeline.encoder).__name__}")
        print(f"  * Volume: {type(pipeline.volume_struct).__name__}")
        print(f"  * Renderer: {type(pipeline.renderer).__name__}")
        print(f"  * UNet: {type(pipeline.unet).__name__}")
        
        print("[PASS] Pipeline initialized successfully!")
        return True
    except Exception as e:
        print(f"[FAIL] Pipeline init failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_inference(device='cuda'):
    """Test 7: Full Pipeline inference (small batch)."""
    print("\n" + "="*60)
    print("TEST 7: Pipeline Inference (Mini Batch)")
    print("="*60)
    
    from hypersplat.beta.genvs.pipeline import GeNVSPipeline
    
    try:
        pipeline = GeNVSPipeline(device=device)
        
        # Create test inputs
        B = 2  # 2 target views
        source_image = torch.randn(1, 3, 256, 256, device=device)
        source_pose = torch.eye(4, device=device).unsqueeze(0)
        source_K = torch.tensor([
            [256, 0, 128],
            [0, 256, 128],
            [0, 0, 1]
        ], dtype=torch.float32, device=device).unsqueeze(0)
        
        target_poses = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
        target_Ks = source_K.expand(B, 3, 3).clone()
        
        print(f"  * Source image: {source_image.shape}")
        print(f"  * Target poses: {target_poses.shape}")
        
        t0 = time.time()
        with torch.no_grad():
            result = pipeline.sample_batch(
                source_image, 
                source_pose, 
                source_K,
                target_poses, 
                target_Ks,
                num_steps=5,  # Very few for speed
                batch_size=2
            )
        t1 = time.time()
        
        # Expect 128x128 because the renderer upsamples to target_res=128 by default
        print(f"  * Output shape: {result.shape}")
        print(f"  * Expected: [2, 3, 128, 128]")
        print(f"  * Time: {(t1-t0):.2f}s")
        
        if result.shape == (B, 3, 128, 128):
            print("[PASS] Pipeline inference successful!")
            return True
        else:
            print(f"[FAIL] Shape mismatch")
            return False
    except Exception as e:
        print(f"[FAIL] Pipeline inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image(data_dir: Path, device='cuda'):
    """Test 8: Pipeline with real image from dataset."""
    print("\n" + "="*60)
    print("TEST 8: Pipeline with Real Image")
    print("="*60)
    
    from hypersplat.beta.genvs.pipeline import GeNVSPipeline
    
    images_dir = data_dir / "images"
    if not images_dir.exists():
        print(f"  [WARN] Images dir not found: {images_dir}")
        return None
    
    # Find first image
    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not images:
        print(" [WARN] No images found")
        return None
    
    try:
        # Load image
        img_path = images[0]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256))
        img_np = np.array(img)
        
        # Convert to tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor = (img_tensor - 0.5) * 2  # Normalize to [-1, 1]
        
        print(f"  * Loaded: {img_path.name}")
        print(f"  * Original size: {Image.open(img_path).size}")
        print(f"  * Tensor shape: {img_tensor.shape}")
        
        # Run through encoder only (lighter test)
        from hypersplat.beta.genvs import GeometryEncoder
        encoder = GeometryEncoder().to(device)
        
        t0 = time.time()
        with torch.no_grad():
            features = encoder(img_tensor)
        t1 = time.time()
        
        print(f"  * Encoder output: {features.shape}")
        print(f"  * Time: {(t1-t0)*1000:.1f}ms")
        print(f"  * Feature stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")
        
        print("[PASS] Real image encoding successful!")
        return True
    except Exception as e:
        print(f"[FAIL] Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="GeNVS-Core Component Tests")
    parser.add_argument("--data-dir", type=str, 
                        default="demo/test_run_genvs_gpu_max_seed2089",
                        help="Data directory with images folder")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests only (skip inference)")
    args = parser.parse_args()
    
    # Resolve data dir
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    
    print("="*60)
    print("    GeNVS-Core Component Test Suite")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Data Dir: {data_dir}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    results = {}
    
    # Run tests
    results["imports"] = test_imports()
    results["encoder"] = test_encoder(args.device)
    results["feature_volume"] = test_feature_volume(args.device)
    results["renderer"] = test_renderer(args.device)
    results["unet"] = test_unet(args.device)
    results["pipeline_init"] = test_pipeline_init(args.device)
    
    if not args.quick:
        results["pipeline_inference"] = test_pipeline_inference(args.device)
        results["real_image"] = test_with_real_image(data_dir, args.device)
    
    # Summary
    print("\n" + "="*60)
    print("    TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        if result is True:
            status = "[PASS] PASS"
            passed += 1
        elif result is False:
            status = "[FAIL] FAIL"
            failed += 1
        else:
            status = "[WARN] SKIP"
            skipped += 1
        print(f"  {name:25s} {status}")
    
    print("-"*60)
    print(f"Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
