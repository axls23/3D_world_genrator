
import sys
import os
import argparse
import time
import json
import csv
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hypersplat.beta.genvs.dataset import GenVSDataset
from hypersplat.beta.genvs import GeometryEncoder, FrustumFeatureVolume, NeuralVolumeRenderer, DiffusionUNet, DiT_S_2


# --------------------------------------------------------------
# Phase Definitions (from High-Level Training Goals)
# --------------------------------------------------------------
PHASE_GEOMETRY  = "Phase 1: Geometry Convergence"
PHASE_DIFFUSION = "Phase 2: Diffusion Denoising"
PHASE_QUALITY   = "Phase 3: Visual Quality Validation"

PHASE_THRESHOLDS = {
    # Phase transitions based on loss values
    "geometry_converged": 0.30,   # Loss below this → geometry is learning
    "diffusion_converged": 0.10,  # Loss below this → UNet is denoising
    "trained": 0.05,              # Loss below this → model is trained
}


class MetricsTracker:
    """Tracks and persists training metrics for dashboard visualization."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_dir = output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffers
        self.loss_history = []           # (step, loss)
        self.loss_ema_history = []       # (step, ema_loss) 
        self.lr_history = []             # (step, lr)
        self.phase_transitions = []     # (step, phase_name)
        self.val_psnr_history = []      # (step, psnr)
        self.val_ssim_history = []      # (step, ssim)
        self.geometry_std_history = []  # (step, feature_std)
        
        # EMA for smooth loss curve
        self._ema_loss = None
        self._ema_alpha = 0.05  # Smoothing factor
        
        # CSV file for persistence
        self.csv_path = self.metrics_dir / "training_log.csv"
        self._init_csv()
    
    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'loss', 'ema_loss', 'lr', 'phase',
                    'val_psnr', 'val_ssim', 'geometry_feat_std'
                ])
    
    def log_step(self, step, loss, lr, phase):
        """Log a single training step."""
        # Update EMA
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss
        
        self.loss_history.append((step, loss))
        self.loss_ema_history.append((step, self._ema_loss))
        self.lr_history.append((step, lr))
    
    def log_phase_transition(self, step, phase_name):
        self.phase_transitions.append((step, phase_name))
        
    def log_validation(self, step, psnr=None, ssim=None, feat_std=None):
        if psnr is not None:
            self.val_psnr_history.append((step, psnr))
        if ssim is not None:
            self.val_ssim_history.append((step, ssim))
        if feat_std is not None:
            self.geometry_std_history.append((step, feat_std))
    
    def save_csv_row(self, step, loss, lr, phase, psnr=None, ssim=None, feat_std=None):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, f"{loss:.6f}", f"{self._ema_loss:.6f}", f"{lr:.8f}", phase,
                f"{psnr:.2f}" if psnr else "", 
                f"{ssim:.4f}" if ssim else "",
                f"{feat_std:.4f}" if feat_std else ""
            ])
    
    def plot_dashboard(self, step, args, current_phase):
        """Generate a comprehensive training dashboard as PNG."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("[Metrics] matplotlib not found, skipping dashboard plot")
            return
        
        fig = plt.figure(figsize=(18, 12), facecolor='#1a1a2e')
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Style
        text_color = '#e0e0e0'
        grid_color = '#333355'
        accent_colors = {
            PHASE_GEOMETRY:  '#ff6b6b',
            PHASE_DIFFUSION: '#4ecdc4',
            PHASE_QUALITY:   '#45b7d1',
        }
        
        plt.rcParams.update({
            'text.color': text_color,
            'axes.labelcolor': text_color,
            'xtick.color': text_color,
            'ytick.color': text_color,
        })
        
        # ── 1. Loss Curve (Top-Left, spans 2 cols) ──
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor('#16213e')
        
        if self.loss_history:
            steps_l, losses = zip(*self.loss_history)
            # Subsample for plotting if too many points
            if len(steps_l) > 2000:
                idx = np.linspace(0, len(steps_l)-1, 2000, dtype=int)
                steps_l = [steps_l[i] for i in idx]
                losses = [losses[i] for i in idx]
            ax1.plot(steps_l, losses, color='#ffffff22', linewidth=0.5, label='Raw Loss')
        
        if self.loss_ema_history:
            steps_e, ema = zip(*self.loss_ema_history)
            if len(steps_e) > 2000:
                idx = np.linspace(0, len(steps_e)-1, 2000, dtype=int)
                steps_e = [steps_e[i] for i in idx]
                ema = [ema[i] for i in idx]
            ax1.plot(steps_e, ema, color='#4ecdc4', linewidth=2, label='EMA Loss')
        
        # Phase transition markers
        for (ps, pname) in self.phase_transitions:
            color = accent_colors.get(pname, '#ffffff')
            ax1.axvline(x=ps, color=color, linestyle='--', alpha=0.7, linewidth=1)
            ax1.text(ps, ax1.get_ylim()[1]*0.95, pname.split(":")[0], 
                    color=color, fontsize=7, rotation=90, va='top')
        
        # Threshold lines
        for label, val in PHASE_THRESHOLDS.items():
            ax1.axhline(y=val, color='#ff6b6b33', linestyle=':', linewidth=1)
            ax1.text(step * 0.01, val, f'{label}: {val}', 
                    color='#ff6b6b88', fontsize=7, va='bottom')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold', color='#4ecdc4')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, color=grid_color, alpha=0.3)
        ax1.set_yscale('log')
        
        # ── 2. Hyperparameters Panel (Top-Right) ──
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor('#16213e')
        ax2.axis('off')
        
        current_loss = self.loss_history[-1][1] if self.loss_history else float('nan')
        current_ema = self._ema_loss if self._ema_loss else float('nan')
        current_lr = self.lr_history[-1][1] if self.lr_history else args.lr
        
        info_text = (
            f"╔══════════════════════════════╗\n"
            f"║   GeNVS Training Dashboard   ║\n"
            f"╠══════════════════════════════╣\n"
            f"║ Step:        {step:>14,} ║\n"
            f"║ Phase:       {current_phase.split(':')[0]:>14} ║\n"
            f"║ Loss (raw):  {current_loss:>14.6f} ║\n"
            f"║ Loss (EMA):  {current_ema:>14.6f} ║\n"
            f"╠══════════════════════════════╣\n"
            f"║        Hyperparameters       ║\n"
            f"╠══════════════════════════════╣\n"
            f"║ LR:          {current_lr:>14.2e} ║\n"
            f"║ Image Size:  {args.image_size:>14} ║\n"
            f"║ Batch Size:  {args.batch_size:>14} ║\n"
            f"║ UNet Ch:     {64:>14} ║\n"
            f"║ Timesteps:   {1000:>14} ║\n"
            f"║ AMP:         {'Enabled' if args.amp else 'Disabled':>14} ║\n"
            f"║ Dataset:     {len(self.loss_history):>10} itr ║\n"
            f"╚══════════════════════════════╝"
        )
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
                fontfamily='monospace', fontsize=9, color='#4ecdc4',
                verticalalignment='top')
        
        # ── 3. Validation PSNR (Middle-Left) ──
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#16213e')
        if self.val_psnr_history:
            steps_p, psnrs = zip(*self.val_psnr_history)
            ax3.plot(steps_p, psnrs, 'o-', color='#45b7d1', linewidth=2, markersize=6)
            ax3.fill_between(steps_p, psnrs, alpha=0.15, color='#45b7d1')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('PSNR (dB)')
        ax3.set_title('Validation PSNR', fontsize=11, fontweight='bold', color='#45b7d1')
        ax3.grid(True, color=grid_color, alpha=0.3)
        
        # ── 4. Geometry Feature Std (Middle-Center) ──
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#16213e')
        if self.geometry_std_history:
            steps_g, stds = zip(*self.geometry_std_history)
            ax4.plot(steps_g, stds, 's-', color='#ff6b6b', linewidth=2, markersize=6)
            ax4.fill_between(steps_g, stds, alpha=0.15, color='#ff6b6b')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Feature Std')
        ax4.set_title('Geometry Feature Variance', fontsize=11, fontweight='bold', color='#ff6b6b')
        ax4.grid(True, color=grid_color, alpha=0.3)
        
        # ── 5. Learning Rate Schedule (Middle-Right) ──
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_facecolor('#16213e')
        if self.lr_history:
            steps_lr, lrs = zip(*self.lr_history)
            # Subsample
            if len(steps_lr) > 500:
                idx = np.linspace(0, len(steps_lr)-1, 500, dtype=int)
                steps_lr = [steps_lr[i] for i in idx]
                lrs = [lrs[i] for i in idx]
            ax5.plot(steps_lr, lrs, color='#ffd93d', linewidth=2)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Learning Rate')
        ax5.set_title('LR Schedule (Cosine)', fontsize=11, fontweight='bold', color='#ffd93d')
        ax5.grid(True, color=grid_color, alpha=0.3)
        ax5.ticklabel_format(axis='y', style='scientific', scilimits=(-4,-4))
        
        # ── 6. Phase Progress Bar (Bottom, spans all) ──
        ax6 = fig.add_subplot(gs[2, :])
        ax6.set_facecolor('#16213e')
        ax6.axis('off')
        
        total_steps = args.num_steps
        progress = step / total_steps
        
        # Draw progress bar
        bar_y = 0.6
        bar_height = 0.25
        # Background
        ax6.barh(bar_y, 1.0, height=bar_height, color='#333355', edgecolor='#555577')
        # Progress
        phase_color = accent_colors.get(current_phase, '#4ecdc4')
        ax6.barh(bar_y, progress, height=bar_height, color=phase_color, alpha=0.8)
        
        # Phase markers
        p1_end = min(2000 / total_steps, 1.0)
        p2_end = min(10000 / total_steps, 1.0)
        ax6.axvline(x=p1_end, ymin=0.3, ymax=0.9, color='#ff6b6b', linewidth=2, linestyle='--')
        ax6.axvline(x=p2_end, ymin=0.3, ymax=0.9, color='#4ecdc4', linewidth=2, linestyle='--')
        
        ax6.text(p1_end, 0.25, 'P1->P2\n2K', ha='center', fontsize=8, color='#ff6b6b')
        ax6.text(p2_end, 0.25, 'P2->P3\n10K', ha='center', fontsize=8, color='#4ecdc4')
        ax6.text(progress, bar_y + bar_height + 0.05, 
                f'Step {step:,}/{total_steps:,} ({progress*100:.1f}%)',
                ha='center', fontsize=11, fontweight='bold', color=text_color)
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1.2)
        ax6.set_title(f'Training Progress — {current_phase}', 
                      fontsize=13, fontweight='bold', color=phase_color, pad=15)
        
        # Save
        dashboard_path = self.metrics_dir / "dashboard.png"
        fig.savefig(dashboard_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        
        # Also save a timestamped copy every N checkpoints
        if step % (args.save_interval * 5) == 0:
            fig_copy_path = self.metrics_dir / f"dashboard_step_{step:06d}.png"
            import shutil
            shutil.copy(dashboard_path, fig_copy_path)
        
        print(f"[Metrics] Dashboard saved to {dashboard_path}")


class GeNVSTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # 1. Pipeline Components
        print("[Trainer] Initializing components...")
        self.encoder = GeometryEncoder(c_feat=16).to(self.device)
        self.volume = FrustumFeatureVolume(c_feat=16).to(self.device)
        self.renderer = NeuralVolumeRenderer(c_feat=16).to(self.device)
        self.unet = DiT_S_2(in_channels=19, out_channels=3, input_size=args.image_size).to(self.device)
        
        # 2. Optimizer with Cosine LR Scheduling
        params = list(self.encoder.parameters()) + \
                 list(self.volume.parameters()) + \
                 list(self.renderer.parameters()) + \
                 list(self.unet.parameters())
                 
        self.optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_steps, eta_min=args.lr * 0.01
        )
        
        # 3. Dataset
        print("[Trainer] Loading dataset...")
        self.dataset = GenVSDataset(args.data_dir, image_size=args.image_size)
        self.dataloader = DataLoader(
            self.dataset, batch_size=args.batch_size, 
            shuffle=True, num_workers=0
        )
        
        # 4. Checkpoints & Validation
        self.output_dir = Path(args.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir = self.output_dir / "validation_samples"
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
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
             self.unet = torch.compile(self.unet)
        
        # 7. Metrics & Phase Tracking
        self.metrics = MetricsTracker(self.output_dir)
        self.current_phase = PHASE_GEOMETRY
        self._phase_logged = set()
        
        # 8. Cache fixed validation pair for visual tracking
        self._val_ref_idx = 0
        self._val_tgt_idx = min(len(self.dataset) - 1, len(self.dataset) // 2)
        
        # 9. Loss window for phase detection
        self._loss_window = deque(maxlen=200)
        
        # 10. Resume from checkpoint
        self.start_step = 0
        if args.resume:
            self.start_step = self._load_checkpoint(args.resume)
        
        # Print config summary
        self._print_config()
    
    def _print_config(self):
        n_params_enc = sum(p.numel() for p in self.encoder.parameters())
        n_params_vol = sum(p.numel() for p in self.volume.parameters())
        n_params_ren = sum(p.numel() for p in self.renderer.parameters())
        unet_mod = self.unet._orig_mod if hasattr(self.unet, '_orig_mod') else self.unet
        n_params_unet = sum(p.numel() for p in unet_mod.parameters())
        n_total = n_params_enc + n_params_vol + n_params_ren + n_params_unet
        
        print(f"\n{'='*60}")
        print(f"  GeNVS Trainer - 3-Phase Training Strategy")
        print(f"{'='*60}")
        print(f"  Dataset:          {len(self.dataset)} frames")
        print(f"  Image Size:       {self.args.image_size}x{self.args.image_size}")
        print(f"  Total Steps:      {self.args.num_steps:,}")
        print(f"  Learning Rate:    {self.args.lr:.1e} -> {self.args.lr * 0.01:.1e} (cosine)")
        print(f"  Batch Size:       {self.args.batch_size}")
        print(f"  AMP:              {'Enabled' if self.enable_amp else 'Disabled'}")
        print(f"  Save Interval:    Every {self.args.save_interval} steps")
        print(f"{'-'*60}")
        print(f"  Encoder:          {n_params_enc:>12,} params")
        print(f"  FrustumVolume:    {n_params_vol:>12,} params")
        print(f"  Renderer:         {n_params_ren:>12,} params")
        print(f"  DiT-S/2:          {n_params_unet:>12,} params")
        print(f"  TOTAL:            {n_total:>12,} params")
        print(f"{'-'*60}")
        print(f"  Phase 1 (Geometry):   Steps 0 -> 2,000   (loss target: < {PHASE_THRESHOLDS['geometry_converged']})")
        print(f"  Phase 2 (Diffusion):  Steps 2,000 -> 10,000 (loss target: < {PHASE_THRESHOLDS['diffusion_converged']})")
        print(f"  Phase 3 (Quality):    Steps 10,000+       (loss target: < {PHASE_THRESHOLDS['trained']})")
        if self.start_step > 0:
            print(f"  Resuming from:    Step {self.start_step}")
        print(f"{'='*60}\n")

    def _load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        print(f"[Trainer] Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.volume.load_state_dict(checkpoint['volume'])
        self.renderer.load_state_dict(checkpoint['renderer'])
        
        unet_state = checkpoint['unet']
        unet_mod = self.unet._orig_mod if hasattr(self.unet, '_orig_mod') else self.unet
        unet_mod.load_state_dict(unet_state)
        
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        step = checkpoint.get('step', 0)
        print(f"[Trainer] Resumed at step {step}")
        return step

    def _detect_phase(self, step):
        """Auto-detect current training phase based on step and loss."""
        if len(self._loss_window) < 50:
            return self.current_phase
        
        avg_loss = sum(self._loss_window) / len(self._loss_window)
        
        new_phase = self.current_phase
        if step < 2000 or avg_loss > PHASE_THRESHOLDS['geometry_converged']:
            new_phase = PHASE_GEOMETRY
        elif step < 10000 or avg_loss > PHASE_THRESHOLDS['diffusion_converged']:
            new_phase = PHASE_DIFFUSION
        else:
            new_phase = PHASE_QUALITY
        
        if new_phase != self.current_phase and new_phase not in self._phase_logged:
            self.current_phase = new_phase
            self._phase_logged.add(new_phase)
            self.metrics.log_phase_transition(step, new_phase)
            print(f"\n{'='*50}")
            print(f"  PHASE TRANSITION -> {new_phase}")
            print(f"  Step: {step:,}  |  Avg Loss: {avg_loss:.4f}")
            print(f"{'='*50}\n")
        
        self.current_phase = new_phase
        return new_phase

    def save_checkpoint(self, step):
        path = self.ckpt_dir / f"step_{step}.pt"
        unet_mod = self.unet._orig_mod if hasattr(self.unet, '_orig_mod') else self.unet
        torch.save({
            'step': step,
            'encoder': self.encoder.state_dict(),
            'volume': self.volume.state_dict(),
            'renderer': self.renderer.state_dict(),
            'unet': unet_mod.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'model_channels': 64,
                'c_feat': 16,
                'image_size': self.args.image_size,
                'phase': self.current_phase,
            }
        }, path)
        print(f"[Trainer] Saved checkpoint to {path}")

    def validate_and_visualize(self, step):
        """
        Run a quick DDIM sample on a fixed source→target pair and save the output.
        Computes PSNR and geometry feature statistics for dashboard tracking.
        """
        from PIL import Image
        
        self.encoder.eval()
        self.volume.eval()
        self.renderer.eval()
        unet = self.unet._orig_mod if hasattr(self.unet, '_orig_mod') else self.unet
        unet.eval()
        
        psnr_val = None
        feat_std = None
        
        try:
            with torch.no_grad():
                # Load fixed validation views
                src_view = self.dataset.load_view(self._val_ref_idx)
                tgt_view = self.dataset.load_view(self._val_tgt_idx)
                
                if src_view is None or tgt_view is None:
                    print("[Trainer] Validation skipped: could not load views")
                    return
                
                src_img = src_view['image'].unsqueeze(0).to(self.device)
                src_pose = src_view['pose'].unsqueeze(0).unsqueeze(0).to(self.device)
                src_K = src_view['K'].unsqueeze(0).unsqueeze(0).to(self.device)
                tgt_pose = tgt_view['pose'].unsqueeze(0).to(self.device)
                tgt_K = tgt_view['K'].unsqueeze(0).to(self.device)
                
                # --- Geometry Stage ---
                volume_feat = self.encoder(src_img)
                volume_expanded = volume_feat.unsqueeze(1)
                cond_feat, depth = self.renderer(volume_expanded, src_pose, src_K, tgt_pose, tgt_K)
                
                # Track geometry feature statistics (Phase 1 indicator)
                feat_std = cond_feat.std().item()
                self.metrics.log_validation(step, feat_std=feat_std)
                
                # --- Save depth map ---
                if depth is not None:
                    depth_np = depth[0, 0].cpu().numpy()
                    depth_norm = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255).astype(np.uint8)
                    depth_img = Image.fromarray(depth_norm, mode='L')
                    depth_img.save(self.val_dir / f"depth_step_{step:06d}.png")
                
                # --- DDIM Sample (20 steps) ---
                H, W = cond_feat.shape[2:]
                x_t = torch.randn(1, 3, H, W, device=self.device)
                ddim_steps = 20
                timesteps = torch.linspace(999, 0, ddim_steps, device=self.device).long()
                
                for t_idx, t in enumerate(timesteps):
                    t_batch = t.unsqueeze(0)
                    noise_pred = unet(x_t, t_batch, cond_feat)
                    
                    ab_t = self.alpha_bar[t]
                    ab_prev = self.alpha_bar[timesteps[min(t_idx + 1, ddim_steps - 1)]]
                    
                    pred_x0 = (x_t - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
                    dir_xt = torch.sqrt(1 - ab_prev) * noise_pred
                    x_t = torch.sqrt(ab_prev) * pred_x0 + dir_xt
                
                # Save generated image
                gen_np = x_t[0].clamp(-1, 1).cpu().permute(1, 2, 0).numpy()
                gen_uint8 = ((gen_np + 1.0) * 127.5).astype(np.uint8)
                Image.fromarray(gen_uint8).save(self.val_dir / f"sample_step_{step:06d}.png")
                
                # Compute PSNR against ground truth
                gt_np = tgt_view['image'].permute(1, 2, 0).numpy()  # [-1, 1]
                gt_uint8 = ((gt_np + 1.0) * 127.5).astype(np.uint8)
                
                # Save ground truth (once)
                gt_path = self.val_dir / "ground_truth_target.png"
                if not gt_path.exists():
                    Image.fromarray(gt_uint8).save(gt_path)
                
                # PSNR: 10 * log10(MAX^2 / MSE)
                mse = np.mean((gen_uint8.astype(float) - gt_uint8.astype(float)) ** 2)
                if mse > 0:
                    psnr_val = 10.0 * np.log10(255.0**2 / mse)
                else:
                    psnr_val = 50.0  # Perfect match
                
                self.metrics.log_validation(step, psnr=psnr_val)
                
                print(f"[Trainer] Validation: PSNR={psnr_val:.2f}dB | Feat.Std={feat_std:.4f} | Saved to {self.val_dir}")
        
        except Exception as e:
            print(f"[Trainer] Validation failed (non-fatal): {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.encoder.train()
            self.volume.train()
            self.renderer.train()
            unet.train()
        
        return psnr_val, feat_std

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
        
        step = self.start_step
        pbar = tqdm(total=self.args.num_steps - step, initial=0)
        
        # Log initial phase
        self.metrics.log_phase_transition(step, self.current_phase)
        
        while step < self.args.num_steps:
            for batch in self.dataloader:
                if step >= self.args.num_steps:
                    break
                
                # Move batch to device
                src_imgs = batch['source_images'].to(self.device)
                src_poses = batch['source_poses'].to(self.device)
                src_Ks = batch['source_Ks'].to(self.device)
                
                tgt_img = batch['target_image'].to(self.device)
                tgt_pose = batch['target_pose'].to(self.device)
                tgt_K = batch['target_K'].to(self.device)
                
                B, N_src, C_img, H_img, W_img = src_imgs.shape

                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=self.enable_amp):
                    # ── 1. Geometry Pass (Conditioning) ──
                    src_imgs_flat = src_imgs.view(B * N_src, C_img, H_img, W_img)
                    volume_flat = self.encoder(src_imgs_flat)
                    
                    C_feat = volume_flat.shape[1]
                    D_planes = volume_flat.shape[2]
                    H_vol = volume_flat.shape[3]
                    W_vol = volume_flat.shape[4]
                    volume = volume_flat.view(B, N_src, C_feat, D_planes, H_vol, W_vol)
                    
                    cond_feat, _ = self.renderer(volume, src_poses, src_Ks, tgt_pose, tgt_K)
                    
                    # ── 2. Diffusion Pass ──
                    t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
                    
                    noise = torch.randn_like(tgt_img)
                    alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
                    
                    x_t = torch.sqrt(alpha_bar_t) * tgt_img + torch.sqrt(1 - alpha_bar_t) * noise
                    
                    noise_pred = self.unet(x_t, t, cond_feat)
                    
                    loss = nn.functional.mse_loss(noise_pred, noise)
                
                # Backward
                if self.enable_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # LR Schedule Step
                self.scheduler.step()
                
                # ── Tracking ──
                step += 1
                loss_val = loss.item()
                current_lr = self.scheduler.get_last_lr()[0]
                self._loss_window.append(loss_val)
                
                # Detect phase
                phase = self._detect_phase(step)
                
                # Log metrics
                self.metrics.log_step(step, loss_val, current_lr, phase)
                
                # Log to CSV every 50 steps
                if step % 50 == 0:
                    self.metrics.save_csv_row(step, loss_val, current_lr, phase)
                
                pbar.update(1)
                pbar.set_description(
                    f"{phase.split(':')[0]} | Loss: {loss_val:.4f} | "
                    f"EMA: {self.metrics._ema_loss:.4f} | LR: {current_lr:.2e}"
                )
                
                # ── Checkpoint + Validation + Dashboard ──
                if step % self.args.save_interval == 0:
                    self.save_checkpoint(step)
                    psnr, feat_std = self.validate_and_visualize(step)
                    self.metrics.plot_dashboard(step, self.args, self.current_phase)
                
                # Eco Mode
                if self.args.eco_mode:
                    time.sleep(self.args.eco_sleep)
        
        # Final checkpoint + dashboard
        self.save_checkpoint(step)
        self.validate_and_visualize(step)
        self.metrics.plot_dashboard(step, self.args, self.current_phase)
        
        pbar.close()
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Final Step: {step:,}")
        print(f"  Final Phase: {self.current_phase}")
        print(f"  Final EMA Loss: {self.metrics._ema_loss:.6f}")
        print(f"  Checkpoints: {self.ckpt_dir}")
        print(f"  Validation Samples: {self.val_dir}")
        print(f"  Dashboard: {self.metrics.metrics_dir / 'dashboard.png'}")
        print(f"  Metrics CSV: {self.metrics.csv_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeNVS 3-Phase Trainer")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ACE output directory")
    parser.add_argument("--output_dir", type=str, default="results/genvs_train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1-2 for VRAM safety)")
    parser.add_argument("--num_steps", type=int, default=10000, help="Total training steps (recommended: 10K-20K)")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint + validation interval")
    parser.add_argument("--image_size", type=int, default=128, help="Training resolution (128 for speed, 256 for quality)")
    parser.add_argument("--device", type=str, default="cuda")
    
    # Smart Optimizations
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (Graph Optimization)")
    parser.add_argument("--eco_mode", action="store_true", help="Enable Eco Mode (Sleep between steps)")
    parser.add_argument("--eco_sleep", type=float, default=0.1, help="Sleep duration in seconds for Eco Mode")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    trainer = GeNVSTrainer(args)
    trainer.train()
