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
| L3 | `chunked` | COLMAP-based chunked pipeline — support level may be limited, see note below |

> **Note on Level 3 (chunked):** `hypersplat/pipeline/wrapper.py`'s `_run_chunked_pipeline` invokes `automated_intelligent_pipeline.py` with `--use_colmap` and `--min_chunk_duration` flags. Depending on which fixes have landed on your checkout, `manager.py`'s CLI may or may not currently accept these flags — treat chunked mode as experimental/may be limited until you've confirmed `--use_colmap`/`--min_chunk_duration` are recognized by `python scripts/pipeline/automated_intelligent_pipeline.py --help` on your branch.

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
| `--with_ut` | disabled | Enable 3DGUT (Unscented Transform) features. This is an `action="store_true"` flag, so it is off unless passed. It is also force-disabled automatically when `--pose-opt` is active (the default), since UT + joint pose optimization are currently incompatible in gsplat — see `hypersplat/pipeline/manager.py`. |
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
python scripts/post/dqn_pruner/prune.py \
    -i output/results/final.ply \
    -o output/results/pruned.ply
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
# Open http://localhost:8081
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
