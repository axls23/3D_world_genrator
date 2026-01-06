"""
PLY to Checkpoint Converter for gsplat Viewer

Converts a pruned/cleaned PLY file back into a checkpoint format
that can be viewed with simple_viewer.py.

Usage:
    python ply_to_ckpt.py --input cleaned.ply --template original_ckpt.pt --output cleaned_ckpt.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData


def load_ply_to_splats(ply_path: str) -> dict:
    """Load PLY file and convert to gsplat splats format."""
    print(f"Loading PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Extract positions
    x = np.array(vertex['x'], dtype=np.float32)
    y = np.array(vertex['y'], dtype=np.float32)
    z = np.array(vertex['z'], dtype=np.float32)
    means = np.stack([x, y, z], axis=1)
    
    # Extract scales (log-encoded in PLY)
    scale_0 = np.array(vertex['scale_0'], dtype=np.float32)
    scale_1 = np.array(vertex['scale_1'], dtype=np.float32)
    scale_2 = np.array(vertex['scale_2'], dtype=np.float32)
    scales = np.stack([scale_0, scale_1, scale_2], axis=1)
    
    # Extract rotations (quaternions)
    rot_0 = np.array(vertex['rot_0'], dtype=np.float32)
    rot_1 = np.array(vertex['rot_1'], dtype=np.float32)
    rot_2 = np.array(vertex['rot_2'], dtype=np.float32)
    rot_3 = np.array(vertex['rot_3'], dtype=np.float32)
    quats = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1)
    
    # Extract opacities (stored as f_dc_0 in some formats, or opacity)
    if 'opacity' in vertex.data.dtype.names:
        opacities = np.array(vertex['opacity'], dtype=np.float32)  # Shape: (N,)
    else:
        opacities = np.zeros(len(x), dtype=np.float32)
    
    # Extract SH coefficients (f_dc_* and f_rest_*)
    sh_dc = []
    for i in range(3):
        if f'f_dc_{i}' in vertex.data.dtype.names:
            sh_dc.append(np.array(vertex[f'f_dc_{i}'], dtype=np.float32))
    
    sh_rest = []
    rest_idx = 0
    while f'f_rest_{rest_idx}' in vertex.data.dtype.names:
        sh_rest.append(np.array(vertex[f'f_rest_{rest_idx}'], dtype=np.float32))
        rest_idx += 1
    
    # Combine SH coefficients
    if sh_dc:
        sh_dc = np.stack(sh_dc, axis=1)
    else:
        sh_dc = np.zeros((len(x), 3), dtype=np.float32)
    
    if sh_rest:
        sh_rest = np.stack(sh_rest, axis=1)
        # Reshape to (N, K, 3) format expected by gsplat
        sh_rest = sh_rest.reshape(len(x), -1, 3) if sh_rest.shape[1] % 3 == 0 else sh_rest
    
    # Build splats dict
    splats = {
        'means': torch.from_numpy(means),
        'scales': torch.from_numpy(scales),
        'quats': torch.from_numpy(quats),
        'opacities': torch.from_numpy(opacities),
    }
    
    # Add SH coefficients
    if sh_dc is not None:
        splats['sh0'] = torch.from_numpy(sh_dc.reshape(-1, 1, 3))
    if sh_rest is not None and len(sh_rest) > 0:
        try:
            splats['shN'] = torch.from_numpy(sh_rest.reshape(len(x), -1, 3))
        except:
            pass
    
    print(f"Loaded {len(means)} Gaussians from PLY")
    return splats


def ply_to_checkpoint(ply_path: str, template_ckpt: str, output_path: str):
    """
    Convert PLY to checkpoint using a template checkpoint for metadata.
    
    Args:
        ply_path: Path to the cleaned/pruned PLY file
        template_ckpt: Original checkpoint to use as template (for step, etc.)
        output_path: Where to save the new checkpoint
    """
    # Load template checkpoint
    print(f"Loading template checkpoint: {template_ckpt}")
    template = torch.load(template_ckpt, map_location='cpu', weights_only=False)
    
    # Load PLY and convert to splats
    new_splats = load_ply_to_splats(ply_path)
    
    # Create new checkpoint
    new_ckpt = {
        'step': template.get('step', 0),
        'splats': new_splats,
    }
    
    # Copy other metadata if present
    for key in ['optimizers', 'schedulers', 'cfg']:
        if key in template:
            new_ckpt[key] = template[key]
    
    # Save
    print(f"Saving checkpoint to: {output_path}")
    torch.save(new_ckpt, output_path)
    print("Done!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert PLY to gsplat checkpoint")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Input PLY file (cleaned/pruned)")
    parser.add_argument("--template", "-t", type=str, required=True,
                        help="Template checkpoint (.pt) for metadata")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output checkpoint path (.pt)")
    args = parser.parse_args()
    
    ply_to_checkpoint(args.input, args.template, args.output)


if __name__ == "__main__":
    main()
