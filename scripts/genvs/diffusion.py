"""
Lightweight Diffusion Wrapper
Uses SD-Turbo or LCM for fast image generation
"""

import torch
import numpy as np
from typing import Optional, Union
from pathlib import Path


class LightweightDiffusion:
    """
    VRAM-efficient diffusion for conditioned image generation.
    Uses SD-Turbo (4 steps) or ControlNet-Depth.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/sd-turbo",
        controlnet_id: Optional[str] = None,
        device: str = "cuda",
        use_fp16: bool = True
    ):
        """
        Args:
            model_id: Diffusion model ID
            controlnet_id: Optional ControlNet model
            device: "cuda" or "cpu"
            use_fp16: Use half precision for memory savings
        """
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        self.pipe = None
        self.controlnet = None
        
    def load(self):
        """Load diffusion pipeline."""
        if self.pipe is not None:
            return
            
        print(f"[Diffusion] Loading {self.model_id}...")
        
        try:
            from diffusers import AutoPipelineForImage2Image, ControlNetModel
            from diffusers import StableDiffusionControlNetPipeline
            
            if self.controlnet_id:
                # Load with ControlNet
                print(f"[Diffusion] Loading ControlNet: {self.controlnet_id}")
                self.controlnet = ControlNetModel.from_pretrained(
                    self.controlnet_id,
                    torch_dtype=self.dtype
                )
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=self.controlnet,
                    torch_dtype=self.dtype
                ).to(self.device)
            else:
                # Load img2img pipeline
                self.pipe = AutoPipelineForImage2Image.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None
                ).to(self.device)
            
            # Memory optimizations
            self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
                
            print("[Diffusion] Loaded successfully")
            
        except ImportError as e:
            print(f"[Diffusion] diffusers not available: {e}")
            print("[Diffusion] Using fallback mode")
            self.pipe = "fallback"
            
    def unload(self):
        """Free VRAM."""
        if self.pipe is not None and self.pipe != "fallback":
            del self.pipe
            del self.controlnet
            self.pipe = None
            self.controlnet = None
            torch.cuda.empty_cache()
            print("[Diffusion] Unloaded")
    
    def generate_fallback(
        self,
        condition: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fallback without diffusion - apply color transfer from reference.
        
        Args:
            condition: Condition map [H, W, C]
            reference: Reference image [H, W, 3]
            
        Returns:
            output: Generated image [H, W, 3]
        """
        H, W = condition.shape[:2]
        
        if reference is None:
            # Just convert condition to RGB
            if condition.ndim == 2:
                output = np.stack([condition, condition, condition], axis=-1)
            else:
                output = condition[:, :, :3]
        else:
            # Simple color transfer: blend reference with condition structure
            import cv2
            
            # Resize reference to match
            ref = cv2.resize(reference, (W, H))
            
            # Use condition as structure, reference as color source
            gray_cond = np.mean(condition, axis=-1, keepdims=True) if condition.ndim == 3 else condition[:, :, np.newaxis]
            gray_cond = np.tile(gray_cond, (1, 1, 3))
            
            # Blend
            output = 0.3 * ref.astype(np.float32) + 0.7 * gray_cond * 255
            output = np.clip(output, 0, 255).astype(np.uint8)
            
        return output
    
    @torch.no_grad()
    def generate(
        self,
        condition: Union[np.ndarray, torch.Tensor],
        reference: Optional[np.ndarray] = None,
        prompt: str = "high quality photo, same style",
        num_steps: int = 4,
        guidance_scale: float = 0.0,
        strength: float = 0.8
    ) -> np.ndarray:
        """
        Generate image conditioned on prior.
        
        Args:
            condition: Condition map [H, W, C] or [1, C, H, W]
            reference: Optional reference image for style
            prompt: Text prompt for generation
            num_steps: Diffusion steps (4 for SD-Turbo)
            guidance_scale: CFG scale (0 for Turbo)
            strength: Denoising strength
            
        Returns:
            output: Generated image [H, W, 3] uint8
        """
        self.load()
        
        # Fallback mode
        if self.pipe == "fallback":
            if isinstance(condition, torch.Tensor):
                condition = condition.squeeze().permute(1, 2, 0).cpu().numpy()
            return self.generate_fallback(condition, reference)
        
        # Prepare condition image
        if isinstance(condition, np.ndarray):
            # Convert to PIL
            from PIL import Image
            
            if condition.ndim == 2:
                cond_img = (condition * 255).astype(np.uint8)
                cond_img = Image.fromarray(cond_img).convert("RGB")
            elif condition.shape[-1] == 1:
                cond_img = (condition[:, :, 0] * 255).astype(np.uint8)
                cond_img = Image.fromarray(cond_img).convert("RGB")
            else:
                if condition.max() <= 1.0:
                    condition = (condition * 255).astype(np.uint8)
                cond_img = Image.fromarray(condition[:, :, :3])
        else:
            # Tensor to PIL
            from PIL import Image
            arr = condition.squeeze().permute(1, 2, 0).cpu().numpy()
            arr = (arr * 255).astype(np.uint8)
            cond_img = Image.fromarray(arr[:, :, :3])
        
        # Reference image
        if reference is not None:
            from PIL import Image
            if reference.max() <= 1.0:
                reference = (reference * 255).astype(np.uint8)
            ref_img = Image.fromarray(reference)
            input_img = ref_img.resize(cond_img.size)
        else:
            input_img = cond_img
        
        # Generate
        if self.controlnet is not None:
            # ControlNet path
            result = self.pipe(
                prompt=prompt,
                image=cond_img,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        else:
            # Img2Img path
            result = self.pipe(
                prompt=prompt,
                image=input_img,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            ).images[0]
        
        # Convert to numpy
        output = np.array(result)
        
        return output
    
    def __call__(
        self,
        condition: Union[np.ndarray, torch.Tensor],
        reference: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        return self.generate(condition, reference, **kwargs)


# Standalone test
if __name__ == "__main__":
    diff = LightweightDiffusion()
    
    # Test with fallback
    condition = np.random.rand(256, 256, 3).astype(np.float32)
    reference = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    output = diff(condition, reference)
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
