
import os
import sys
import time
import json
import torch
import torch.nn as nn
import viser
import numpy as np
from pathlib import Path
from typing import Optional, Dict

# Setup Logging
logging_path = Path(__file__).parent / "dashboard.log"
import logging
logging.basicConfig(
    filename=str(logging_path),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup path for local packages
# Try multiple strategies to find root
SEARCH_ROOT = Path(__file__).resolve().parent
while SEARCH_ROOT.name and SEARCH_ROOT.name != "3D_world_genrator":
    if (SEARCH_ROOT / "examples").exists():
        break
    SEARCH_ROOT = SEARCH_ROOT.parent

PROJECT_ROOT = SEARCH_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

logger.info(f"Dashboard starting. Project Root: {PROJECT_ROOT}")

try:
    from gsplat_viewer import GsplatViewer
    from gsplat.rendering import rasterization
    from nerfview import CameraState, RenderTabState
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class DashboardRunner:
    """A minimal version of the Runner that only handles Rendering for the Dashboard."""
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.splats = nn.ParameterDict()
        self.cfg = type('obj', (object,), {
            'sh_degree': 3,
            'packed': False,
            'sparse_grad': False,
            'camera_model': 'pinhole',
            'antialiased': False
        })
        self.last_ckpt_path = None

    def load_checkpoint(self, ckpt_path: Path):
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            splat_data = ckpt["splats"]
            
            # Re-create parameters if number of gaussians changed
            N = splat_data["means"].shape[0]
            if not self.splats or self.splats["means"].shape[0] != N:
                self.splats = nn.ParameterDict({
                    k: nn.Parameter(v.to(self.device)) for k, v in splat_data.items()
                })
            else:
                self.splats.load_state_dict(splat_data)
            
            # Infer SH degree
            if "shN" in splat_data:
                n_coeffs = splat_data["shN"].shape[1] + 1
                self.cfg.sh_degree = int(np.sqrt(n_coeffs) - 1)
            else:
                self.cfg.sh_degree = 0
                
            self.last_ckpt_path = ckpt_path
            return True
        except Exception as e:
            print(f"[Dashboard] Load failed: {e}")
            return False

    def render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState):
        """The callback used by GsplatViewer to render the scene."""
        if not self.splats:
            return np.zeros((camera_state.height, camera_state.width, 3), dtype=np.uint8)

        # Build matrices
        viewmat = torch.from_numpy(camera_state.viewmat).float().to(self.device)
        K = torch.from_numpy(camera_state.K).float().to(self.device)
        
        # Pull parameters
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        
        # Colors (SH Support)
        if "sh0" in self.splats:
             colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)
        else:
             colors = torch.sigmoid(self.splats["colors"])

        # Map Render Modes
        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        # Use gsplat rasterization for high-quality "Simple Trainer" style view
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=camera_state.width,
            height=camera_state.height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            packed=False,
        )
        
        if render_tab_state.render_mode == "rgb":
            canvas = torch.clamp(render_colors[0, ..., 0:3], 0.0, 1.0)
        else:
            # Handle depth/alpha visualization
            renders = render_colors[0]
            if render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
                renders = (renders - render_tab_state.near_plane) / (render_tab_state.far_plane - render_tab_state.near_plane)
            canvas = torch.clamp(renders, 0.0, 1.0)
            if canvas.shape[-1] == 1:
                canvas = canvas.repeat(1, 1, 3)

        return (canvas.detach().cpu().numpy() * 255).astype(np.uint8)

def main():
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--dir", required=True)
        parser.add_argument("--port", type=int, default=8082)
        args = parser.parse_args()
        
        results_dir = Path(args.dir)
        ckpt_dir = results_dir / "ckpts"
        
        # Wait for results_dir to exist (Director might be starting ACE-Zero first)
        logger.info(f"Dashboard waiting for results directory: {results_dir}")
        while not results_dir.exists():
            time.sleep(5)
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        runner = DashboardRunner(device=device)
        
        server = viser.ViserServer(port=args.port)
        server.gui.add_markdown("### 🏆 GeNVS Premium Dashboard")
        
        # Progress Panel
        episode_gui = server.gui.add_text("Episode", initial_value="1", disabled=True)
        step_gui = server.gui.add_text("Step", initial_value="0", disabled=True)
        reward_gui = server.gui.add_text("Total Reward", initial_value="0.0", disabled=True)
        psnr_gui = server.gui.add_text("PSNR", initial_value="0.0", disabled=True)
        
        # Initialize GsplatViewer with our render_fn
        viewer = GsplatViewer(
            server=server,
            render_fn=runner.render_fn,
            output_dir=results_dir,
            mode="rendering"
        )
        
        logger.info(f"Dashboard active at http://localhost:{args.port}")
        print(f"Dashboard active at http://localhost:{args.port}")
        
        while True:
            # Load State
            state_path = results_dir / "state.json"
            if state_path.exists():
                try:
                    with open(state_path, 'r') as f:
                        state_data = json.load(f)
                        episode_gui.value = str(state_data.get("episode", 1))
                        step_gui.value = str(state_data.get("step", 0))
                        reward_gui.value = f"{state_data.get('total_reward', 0.0):.2f}"
                except: pass

            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
                if ckpts:
                    latest = ckpts[-1]
                    if latest != runner.last_ckpt_path:
                        if runner.load_checkpoint(latest):
                            logger.info(f"Reloaded Model: {latest.name}")
                            viewer.rerender()
                            
                            # Update stats
                            stats_files = sorted((results_dir / "stats").glob("*.json"), key=lambda p: p.stat().st_mtime)
                            if stats_files:
                                 try:
                                     with open(stats_files[-1], 'r') as f:
                                         psnr_gui.value = f"{json.load(f).get('psnr', 0.0):.2f}"
                                 except: pass
                                 
            time.sleep(2)
    except Exception as e:
        import traceback
        error_msg = f"Dashboard CRASHED: {e}\n{traceback.format_exc()}"
        print(error_msg)
        logger.error(error_msg)

if __name__ == "__main__":
    main()
