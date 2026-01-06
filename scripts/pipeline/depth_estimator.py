"""
Unified Depth Estimation Module

Supports multiple backends:
- ZoeDepth (metric, high quality, ~200ms/frame)
- Depth Anything v2 (relative, fast, ~30ms/frame)
- MiDaS (relative, lightweight, ~100ms/frame)

Usage:
    depth_estimator = DepthEstimator(model="depth_anything")
    depth_map = depth_estimator(image)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Literal
import cv2


class DepthEstimator:
    """
    Unified depth estimation with multiple backends.
    
    Models:
        - zoedepth: Metric depth in meters (default, same as ACE-Zero)
        - depth_anything: Relative depth, very fast (30+ FPS)
        - midas: Relative depth, lightweight
    """
    
    MODELS = ["zoedepth", "depth_anything", "midas"]
    
    def __init__(
        self, 
        model: Literal["zoedepth", "depth_anything", "midas"] = "zoedepth",
        device: str = "cuda"
    ):
        self.model_name = model
        self.device = device
        self.model = None
        self.transform = None
        
    def load(self):
        """Lazy load the depth model."""
        if self.model is not None:
            return
            
        print(f"[DepthEstimator] Loading {self.model_name}...")
        
        if self.model_name == "zoedepth":
            self._load_zoedepth()
        elif self.model_name == "depth_anything":
            self._load_depth_anything()
        elif self.model_name == "midas":
            self._load_midas()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
    def _load_zoedepth(self):
        """Load ZoeDepth (metric depth)."""
        try:
            repo = "isl-org/ZoeDepth"
            self.model = torch.hub.load(repo, "ZoeD_NK", pretrained=True, trust_repo=True)
            self.model = self.model.to(self.device).eval()
            print("[DepthEstimator] ZoeDepth loaded (metric depth)")
        except Exception as e:
            print(f"[DepthEstimator] ZoeDepth failed: {e}")
            print("[DepthEstimator] Falling back to MiDaS")
            self._load_midas()
            
    def _load_depth_anything(self):
        """Load Depth Anything v2 (fast, relative depth)."""
        try:
            # Try depth_anything_v2 package first
            from depth_anything_v2.dpt import DepthAnythingV2
            
            # Load small model for speed
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            }
            
            self.model = DepthAnythingV2(**model_configs['vits'])
            
            # Download weights if needed
            weights_path = Path.home() / ".cache" / "depth_anything_v2" / "depth_anything_v2_vits.pth"
            if not weights_path.exists():
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
                print(f"[DepthEstimator] Downloading Depth Anything weights...")
                torch.hub.download_url_to_file(url, str(weights_path))
            
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            self.model = self.model.to(self.device).eval()
            print("[DepthEstimator] Depth Anything v2 loaded (fast)")
            
        except ImportError:
            print("[DepthEstimator] depth_anything_v2 not installed, using torch.hub")
            try:
                # Fallback: use transformers
                from transformers import pipeline
                self.model = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
                print("[DepthEstimator] Depth Anything v2 loaded via transformers")
            except Exception as e:
                print(f"[DepthEstimator] Depth Anything failed: {e}")
                print("[DepthEstimator] Falling back to MiDaS")
                self._load_midas()
                
    def _load_midas(self):
        """Load MiDaS (lightweight, relative depth)."""
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
            self.model = self.model.to(self.device).eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.dpt_transform
            
            print("[DepthEstimator] MiDaS loaded (lightweight)")
            self.model_name = "midas"  # Mark as midas
        except Exception as e:
            print(f"[DepthEstimator] MiDaS failed: {e}")
            self.model = "fallback"
            
    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Args:
            image: RGB image [H, W, 3] uint8
            
        Returns:
            depth: Depth map [H, W] float32
        """
        self.load()
        
        if self.model == "fallback":
            return self._fallback_depth(image)
            
        H, W = image.shape[:2]
        
        if self.model_name == "zoedepth":
            from PIL import Image
            pil_img = Image.fromarray(image)
            depth = self.model.infer_pil(pil_img)
            depth = np.array(depth)
            
        elif self.model_name == "depth_anything":
            if hasattr(self.model, 'infer_image'):
                # Native depth_anything_v2
                depth = self.model.infer_image(image)
            else:
                # Transformers pipeline
                from PIL import Image
                pil_img = Image.fromarray(image)
                result = self.model(pil_img)
                depth = np.array(result['depth'])
                
        elif self.model_name == "midas":
            input_batch = self.transform(image).to(self.device)
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth = prediction.cpu().numpy()
            # MiDaS outputs inverse depth, convert
            depth = 1.0 / (depth + 1e-6)
            
        return depth.astype(np.float32)
    
    def _fallback_depth(self, image: np.ndarray) -> np.ndarray:
        """Gradient-based fallback when no model available."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Simple gradient magnitude as pseudo-depth
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        depth = 1.0 / (np.sqrt(gx**2 + gy**2) + 1e-6)
        depth = cv2.GaussianBlur(depth, (21, 21), 0)
        return depth
        
    def unload(self):
        """Free VRAM."""
        if self.model is not None and self.model != "fallback":
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            print(f"[DepthEstimator] {self.model_name} unloaded")
            
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.estimate(image)
        
    @staticmethod
    def benchmark(image: np.ndarray, device: str = "cuda"):
        """Benchmark all depth models."""
        import time
        
        results = {}
        for model_name in DepthEstimator.MODELS:
            try:
                est = DepthEstimator(model=model_name, device=device)
                
                # Warmup
                _ = est(image)
                
                # Benchmark
                times = []
                for _ in range(10):
                    start = time.time()
                    _ = est(image)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times) * 1000  # ms
                fps = 1000 / avg_time
                results[model_name] = {"ms": avg_time, "fps": fps}
                
                est.unload()
            except Exception as e:
                results[model_name] = {"error": str(e)}
                
        print("\n=== Depth Model Benchmark ===")
        for name, res in results.items():
            if "error" in res:
                print(f"  {name}: FAILED - {res['error']}")
            else:
                print(f"  {name}: {res['ms']:.1f}ms ({res['fps']:.1f} FPS)")
                
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Depth Estimation")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="zoedepth", 
                        choices=DepthEstimator.MODELS,
                        help="Depth model to use")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all models")
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if args.benchmark:
        DepthEstimator.benchmark(image)
    else:
        est = DepthEstimator(model=args.model)
        depth = est(image)
        print(f"Depth shape: {depth.shape}")
        print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
        
        # Save
        output_path = Path(args.image).stem + "_depth.npy"
        np.save(output_path, depth)
        print(f"Saved to {output_path}")
