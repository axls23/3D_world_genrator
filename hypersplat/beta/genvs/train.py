
import sys
import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hypersplat.beta.genvs.dataset import GenVSDataset
from hypersplat.beta.genvs import GeometryEncoder, FrustumFeatureVolume, NeuralVolumeRenderer, DiffusionUNet

class GeNVSTrainer:
    def __init__(self, args):
        self.args = args
# ... (skipping unchanged lines is hard with replace_file_content large blocks)
# Actually I'll use separate replacements for import and unpacking.

        self.device = args.device
        
        # 1. Pipeline Components
        print("[Trainer] Initializing components...")
        self.encoder = GeometryEncoder(c_feat=16).to(self.device)
        self.volume = FrustumFeatureVolume(c_feat=16).to(self.device)
        self.renderer = NeuralVolumeRenderer(c_feat=16).to(self.device)
        self.unet = DiffusionUNet(in_channels=19, out_channels=3, model_channels=64).to(self.device)
        
        # 2. Optimizer
        params = list(self.encoder.parameters()) + \
                 list(self.volume.parameters()) + \
                 list(self.renderer.parameters()) + \
                 list(self.unet.parameters())
                 
        self.optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        
        # 3. Dataset
        print("[Trainer] Loading dataset...")
        self.dataset = GenVSDataset(args.data_dir, image_size=args.image_size)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        # 4. Checkpoints
        self.ckpt_dir = Path(args.output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. Noise Schedule (Linear Beta)
        self.num_timesteps = 1000
        self.beta = torch.linspace(1e-4, 0.02, self.num_timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 6. Smart Optimizations
        self.enable_amp = args.amp
        self.scaler = torch.amp.GradScaler('cuda') if self.enable_amp else None
        
        if args.compile:
             print("[Trainer] Compiling model with torch.compile...")
             # Compile UNet as it's the heaviest part
             self.unet = torch.compile(self.unet)

    def save_checkpoint(self, step):
        path = self.ckpt_dir / f"step_{step}.pt"
        torch.save({
            'step': step,
            'encoder': self.encoder.state_dict(),
            'volume': self.volume.state_dict(),
            'renderer': self.renderer.state_dict(),
            # Save original model if compiled
            'unet': self.unet._orig_mod.state_dict() if hasattr(self.unet, '_orig_mod') else self.unet.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"[Trainer] Saved checkpoint to {path}")

    def train(self):
        print(f"[Trainer] Starting training for {self.args.num_steps} steps...")
        if self.enable_amp:
            print("[Trainer] AMP Enabled (Mixed Precision)")
        if self.args.eco_mode:
            print(f"[Trainer] Eco Mode Enabled (Sleep {self.args.eco_sleep}s/step)")

        self.encoder.train()
        self.volume.train()
        self.renderer.train()
        self.unet.train()
        
        step = 0
        pbar = tqdm(total=self.args.num_steps)
        
        while step < self.args.num_steps:
            for batch in self.dataloader:
                if step >= self.args.num_steps:
                    break
                
                # Move batch to device
                src_img = batch['source_image'].to(self.device)
                src_pose = batch['source_pose'].to(self.device)
                src_K = batch['source_K'].to(self.device)
                
                tgt_img = batch['target_image'].to(self.device) # [-1, 1] GT
                tgt_pose = batch['target_pose'].to(self.device)
                tgt_K = batch['target_K'].to(self.device)
                
                B = src_img.shape[0]

                # Optimization: Zero grad first
                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=self.enable_amp):
                    # -------------------------
                    # 1. Geometry Pass (Conditioning)
                    # -------------------------
                    # Encoder -> Feature Volume
                    volume = self.encoder(src_img) # [B, C, D, H, W] 
                    
                    src_pose_expanded = src_pose.unsqueeze(1) # [B, 1, 4, 4]
                    src_K_expanded = src_K.unsqueeze(1) # [B, 1, 3, 3]
                    
                    volume_expanded = volume.unsqueeze(1) # [B, 1, C, D, H, W]
                    
                    cond_feat, _ = self.renderer(volume_expanded, src_pose_expanded, src_K_expanded, tgt_pose, tgt_K)
                    # Output: [B, C, H, W]
                    
                    # -------------------------
                    # 2. Diffusion Pass
                    # -------------------------
                    t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
                    
                    noise = torch.randn_like(tgt_img)
                    alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
                    
                    x_t = torch.sqrt(alpha_bar_t) * tgt_img + torch.sqrt(1 - alpha_bar_t) * noise
                    
                    noise_pred = self.unet(x_t, t, cond_feat)
                    
                    loss = nn.functional.mse_loss(noise_pred, noise)
                
                # Backward Pass
                if self.enable_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Logging
                step += 1
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                
                if step % self.args.save_interval == 0:
                    self.save_checkpoint(step)
                
                # Eco Mode: Cool-down sleep
                if self.args.eco_mode:
                    time.sleep(self.args.eco_sleep)
                
                # Logging
                step += 1
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                
                if step % self.args.save_interval == 0:
                    self.save_checkpoint(step)
        
        self.save_checkpoint(step)
        print("[Trainer] Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ACE output directory")
    parser.add_argument("--output_dir", type=str, default="results/genvs_train")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1) # Low batch size for VRAM safety
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=128) # 128 for training speed/stability initially
    parser.add_argument("--device", type=str, default="cuda")
    
    # Smart Optimizations
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (Graph Optimization)")
    parser.add_argument("--eco_mode", action="store_true", help="Enable Eco Mode (Sleep between steps)")
    parser.add_argument("--eco_sleep", type=float, default=0.1, help="Sleep duration in seconds for Eco Mode")
    
    args = parser.parse_args()
    
    trainer = GeNVSTrainer(args)
    trainer.train()
