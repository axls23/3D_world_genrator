"""
Zero123-XL Wrapper for GeNVS-Vector Pipeline

Multi-view synthesis from a single image using Zero123-XL (fp16).
Generates novel views at specified azimuth/elevation angles.

VRAM: ~4GB in fp16 mode
Used to replace symmetry-based back view generation for asymmetric objects.
"""

import gc
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image


class Zero123Wrapper:
    """
    Zero123-XL wrapper for novel view synthesis.
    
    Generates multi-view images from a single reference view.
    Uses fp16 precision to fit in 6GB VRAM when run sequentially.
    
    Usage:
        wrapper = Zero123Wrapper(device="cuda")
        novel_views = wrapper.generate(
            image,
            azimuths=[180, 90, 270],  # Back, Left, Right
            elevations=[0, 0, 0]
        )
        wrapper.unload()  # Free VRAM for next stage
    """
    
    MODEL_ID = "sudo-ai/zero123plus-v1.2"
    
    def __init__(
        self,
        device: str = "cuda",
        use_fp16: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.device = device
        self.use_fp16 = use_fp16
        self.cache_dir = cache_dir
        self.pipe = None
        self._loaded = False
        
    def _load_model(self):
        """Lazy load Zero123 model."""
        if self._loaded:
            return
            
        print("[Zero123] Loading Zero123Plus-v1.2 (fp16)...")
        try:
            from diffusers import DiffusionPipeline
            
            dtype = torch.float16 if self.use_fp16 else torch.float32
            
            self.pipe = DiffusionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.pipe.to(self.device)
            
            # Enable memory optimizations
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            
            self._loaded = True
            print("[Zero123] Model loaded successfully")
            
        except ImportError as e:
            print(f"[Zero123] diffusers not installed: {e}")
            print("[Zero123] Install with: pip install diffusers transformers accelerate")
            self._loaded = False
            
        except Exception as e:
            print(f"[Zero123] Failed to load model: {e}")
            self._loaded = False
    
    def generate(
        self,
        image: Union[np.ndarray, Image.Image],
        azimuths: List[float] = [180],  # Back view by default
        elevations: List[float] = [0],
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0
    ) -> List[np.ndarray]:
        """
        Generate novel views at specified angles.
        
        Args:
            image: Reference image (front view)
            azimuths: List of azimuth angles in degrees (0=front, 180=back)
            elevations: List of elevation angles in degrees
            num_inference_steps: Diffusion steps (more = better quality, slower)
            guidance_scale: CFG scale (higher = more consistent, less diverse)
            
        Returns:
            List of generated images as numpy arrays
        """
        self._load_model()
        
        if not self._loaded:
            print("[Zero123] Model not available, returning empty list")
            return []
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Resize to expected input size
        image = image.resize((256, 256), Image.LANCZOS)
        
        novel_views = []
        
        for azimuth, elevation in zip(azimuths, elevations):
            print(f"[Zero123] Generating view at azimuth={azimuth}°, elevation={elevation}°")
            
            try:
                with torch.no_grad():
                    result = self.pipe(
                        image,
                        elevation=elevation,
                        azimuth=azimuth,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    ).images[0]
                
                # Convert to numpy
                novel_view = np.array(result)
                novel_views.append(novel_view)
                
            except Exception as e:
                print(f"[Zero123] Generation failed for az={azimuth}: {e}")
                continue
        
        return novel_views
    
    def generate_multiview(
        self,
        image: Union[np.ndarray, Image.Image],
        num_views: int = 6,
        elevation: float = 0
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generate multiple views evenly distributed around the object.
        
        Args:
            image: Reference image (front view)
            num_views: Number of views to generate (evenly spaced)
            elevation: Fixed elevation angle
            
        Returns:
            List of (image, azimuth, elevation) tuples
        """
        # Evenly space azimuths, avoid 0 (front) since we already have it
        azimuths = [360 * i / num_views for i in range(1, num_views)]
        elevations = [elevation] * len(azimuths)
        
        views = self.generate(image, azimuths, elevations)
        
        return list(zip(views, azimuths, [elevation] * len(views)))
    
    def unload(self):
        """Unload model to free VRAM for next pipeline stage."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._loaded = False
            
        # Aggressive cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("[Zero123] Unloaded, VRAM freed")


def flush_vram():
    """Utility to aggressively flush VRAM between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Report VRAM usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Zero123 Novel View Synthesis")
    parser.add_argument("--image", type=str, required=True, help="Input image")
    parser.add_argument("--output_dir", type=str, default="novel_views", help="Output directory")
    parser.add_argument("--num_views", type=int, default=6, help="Number of views")
    args = parser.parse_args()
    
    # Load image
    from PIL import Image
    img = Image.open(args.image).convert("RGB")
    
    # Generate views
    wrapper = Zero123Wrapper(device="cuda")
    results = wrapper.generate_multiview(img, num_views=args.num_views)
    wrapper.unload()
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (view, az, el) in enumerate(results):
        out_path = output_dir / f"view_az{int(az)}_el{int(el)}.png"
        Image.fromarray(view).save(out_path)
        print(f"Saved: {out_path}")
    
    print(f"Generated {len(results)} views")
