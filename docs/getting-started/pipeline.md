# Pipeline Guide

Modern video-to-3DGS pipeline using ACE-Zero for COLMAP-free pose estimation.

## Quick Start

```bash
# Full pipeline with ACE-Zero (default)
python scripts/pipeline/automated_intelligent_pipeline.py video.mp4 --output_dir output/

# With GeNVS novel view augmentation
python scripts/pipeline/automated_intelligent_pipeline.py video.mp4 --genvs --genvs-views 20
```

---

## Pipeline Levels

| Level | Mode | Description |
|-------|------|-------------|
| **L2** | `ace_zero` | ACE-Zero → 3DGS MCMC (default, recommended) |
| L1 | `simple` | Direct training on pre-processed data |
| L3 | `chunked` | Legacy COLMAP-based chunked pipeline |

---

## Level 2: ACE-Zero Pipeline (Default)

### Flow
```
Video → Frame Extraction → ACE-Zero Pose Estimation → [GeNVS] → 3DGS MCMC → [DQN Pruner]
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fps` | 1.5 | Frame extraction rate |
| `--max_steps` | 7000 | Training iterations |
| `--data_factor` | 2 | Image downsampling (1=full, 2=half) |
| `--with_ut` | enabled | Enable 3DGUT features |
| `--genvs` | disabled | Enable novel view synthesis |
| `--genvs-views` | 20 | Number of synthetic views |

### Example Commands

```bash
# Standard quality
python scripts/pipeline/automated_intelligent_pipeline.py video.mp4 \
    --output_dir output/ --max_steps 7000

# High quality with augmentation
python scripts/pipeline/automated_intelligent_pipeline.py video.mp4 \
    --output_dir output/ --max_steps 15000 --genvs --genvs-views 30

# Fast preview
python scripts/pipeline/automated_intelligent_pipeline.py video.mp4 \
    --output_dir output/ --max_steps 3000 --fps 2
```

---

## GeNVS-Lite (Novel View Synthesis)

Augments training data with synthetic back-views using diffusion models.

### How It Works
1. **Symmetry Prior**: Mirrors front views as initialization
2. **Depth Estimation**: ZoeDepth for consistent geometry
3. **Diffusion Refinement**: SD-Turbo img2img enhancement
4. **COLMAP Injection**: Adds synthetic poses to dataset

### When to Use
- Sparse input (< 50 frames)
- Single-sided captures
- Object-centric scenes

### Limitations
- Asymmetric objects (trees, natural scenes)
- Novel views lack depth supervision

---

## DQN Pruner (Post-Processing)

Removes floater Gaussians using reinforcement learning.

```bash
python scripts/pipeline/dqn_floater_pruner.py \
    --ply_path output/results/final.ply \
    --output_path output/results/pruned.ply
```

Typical reduction: 15-25% Gaussians removed.

---

## Output Structure

```
output/
├── acezero/           # Pose estimation results
│   ├── images/        # Extracted frames
│   ├── images_2/      # Downsampled (data_factor=2)
│   └── sparse/        # COLMAP-format poses
├── results/           # Training outputs
│   ├── ckpts/         # Checkpoints (.pt)
│   ├── renders/       # Evaluation renders
│   └── final.ply      # 3D model
└── chunks/            # (L3 only) Per-chunk results
```

---

## Demo Server

Web interface for the pipeline:

```bash
cd demo && python demo_server.py
# Open http://localhost:8080
```

API endpoints:
- `POST /api/upload` - Upload video
- `POST /api/train` - Start training
- `GET /api/status` - Training progress
- `POST /api/view` - Start 3D viewer

---

## Troubleshooting

### ACE-Zero fails
- Check WSL GPU access: `nvidia-smi` in WSL
- Verify conda environment: `conda activate ace0`

### Out of VRAM
- Increase `--data_factor` (2 → 4)
- Reduce `--max_steps`
- Disable `--genvs`

### Poor reconstruction
- Increase `--fps` for more frames
- Enable `--genvs` for sparse captures
- Check video quality (motion blur, low light)
