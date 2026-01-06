import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import viser
from plyfile import PlyData

# Add project root and examples to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
examples_dir = project_root / "examples"
sys.path.append(str(project_root))
sys.path.append(str(examples_dir))

from gsplat.rendering import rasterization
# Try importing gsplat_viewer from examples
try:
    from gsplat_viewer import GsplatViewer, GsplatRenderTabState
except ImportError:
    # If running from a different context, try relative import or just fail gracefully
    sys.path.append(str(Path.cwd() / "gsplat" / "examples"))
    from gsplat_viewer import GsplatViewer, GsplatRenderTabState

from nerfview import CameraState, RenderTabState, apply_float_colormap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ply(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Loads a PLY file and returns the Gaussian Splat parameters as tensors."""
    plydata = PlyData.read(path)
    
    xyz = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']), axis=1)
    opacities = plydata['vertex']['opacity']
    
    scale_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.stack([plydata['vertex'][n] for n in scale_names], axis=1)
    
    rot_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.stack([plydata['vertex'][n] for n in rot_names], axis=1)
    
    # SH features
    features_dc = np.stack([plydata['vertex'][n] for n in ['f_dc_0', 'f_dc_1', 'f_dc_2']], axis=1)
    features_dc = features_dc.reshape(-1, 1, 3)
    
    extra_f_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.stack([plydata['vertex'][n] for n in extra_f_names], axis=1)
    # Reshape features_extra to [N, (degrees+1)^2 - 1, 3]
    # Total SH coeffs = (degrees+1)^2 * 3
    # We have 3 DC coeffs, so remaining are ((degrees+1)^2 - 1) * 3
    num_extra_features = len(extra_f_names)
    num_extra_coeffs = num_extra_features // 3
    features_extra = features_extra.reshape(-1, num_extra_coeffs, 3)
    
    features = np.concatenate((features_dc, features_extra), axis=1)
    
    return {
        "means": torch.from_numpy(xyz).float().to(device),
        "scales": torch.from_numpy(scales).float().to(device),
        "quats": torch.from_numpy(rots).float().to(device),
        "opacities": torch.from_numpy(opacities).float().to(device),
        "colors": torch.from_numpy(features).float().to(device)
    }

def main():
    parser = argparse.ArgumentParser(description="Visualize Gaussian Splat Chunks")
    parser.add_argument("--input_path", required=True, help="Path to a PLY file or a directory containing chunks/manifest.json")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viewer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_path = Path(args.input_path)
    chunks_data = []
    
    if input_path.is_file() and input_path.suffix == '.ply':
        logger.info(f"Loading single PLY file: {input_path}")
        chunks_data.append(load_ply(str(input_path), device))
    elif input_path.is_dir():
        # Check for manifest
        manifest_path = input_path / "manifest.json"
        if manifest_path.exists():
            logger.info(f"Loading chunks from manifest: {manifest_path}")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            for chunk_id, chunk_info in manifest["chunks"].items():
                chunk_file = input_path / chunk_info["file"]
                if chunk_file.exists():
                    logger.info(f"Loading chunk {chunk_id}...")
                    chunks_data.append(load_ply(str(chunk_file), device))
        else:
            # Load all .ply files in dir
            logger.info(f"Loading all PLY files from directory: {input_path}")
            for f in input_path.glob("*.ply"):
                logger.info(f"Loading {f.name}...")
                chunks_data.append(load_ply(str(f), device))
    
    if not chunks_data:
        logger.error("No data loaded!")
        return

    # Concatenate all chunks for rendering
    # In a more advanced viewer, we might want to toggle them, but for now let's merge
    means = torch.cat([c["means"] for c in chunks_data], dim=0)
    scales = torch.cat([c["scales"] for c in chunks_data], dim=0)
    quats = torch.cat([c["quats"] for c in chunks_data], dim=0)
    opacities = torch.cat([c["opacities"] for c in chunks_data], dim=0)
    colors = torch.cat([c["colors"] for c in chunks_data], dim=0)
    
    # Normalize quats
    quats = F.normalize(quats, p=2, dim=-1)
    # Exp scales (usually stored as log scale in checkpoints, but PLY might be raw scale? 
    # Standard 3DGS PLY stores log scales usually. Let's assume log scales as per simple_viewer logic)
    scales = torch.exp(scales)
    # Sigmoid opacities
    opacities = torch.sigmoid(opacities)
    
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)
    logger.info(f"Loaded {len(means)} Gaussians. SH degree: {sh_degree}")

    # Viewer setup
    server = viser.ViserServer(port=args.port, verbose=False)
    
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
            
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat[None],
            K[None],
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
        )

        if render_tab_state.render_mode == "rgb":
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            
        return renders

    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path("."),
        mode="rendering",
    )
    
    logger.info(f"Viewer running at http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")

if __name__ == "__main__":
    main()
