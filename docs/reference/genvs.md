# GeNVS-Lite

Novel view synthesis module for augmenting sparse 3DGS training data.

## Overview

GeNVS-Lite generates synthetic back-views to fill gaps in single-sided captures.

```
Front Views → Symmetry Mirror → Diffusion Refinement → Augmented Dataset
```

## Usage

### Integrated Pipeline

```bash
python scripts/pipeline/automated_intelligent_pipeline.py video.mp4 \
    --genvs \
    --genvs-views 20
```

### Standalone

```bash
# Generate novel views
python scripts/genvs/run_completion.py \
    --images path/to/frames \
    --output novel_views \
    --num-views 20

# Inject into COLMAP dataset
python scripts/genvs/inject_back_views.py \
    --colmap-dir acezero_output \
    --novel-views novel_views \
    --output-dir augmented_dataset
```

## Components

| File | Purpose |
|------|---------|
| `run_completion.py` | Main runner |
| `diffusion.py` | SD-Turbo/ControlNet wrapper |
| `inject_back_views.py` | COLMAP augmentation |

### Ensemble Priors (`ensemble/`)

| File | Purpose |
|------|---------|
| `zoedepth_wrapper.py` | Metric depth estimation |
| `midas_wrapper.py` | Relative depth (fallback) |
| `normal_wrapper.py` | Surface normals |
| `edge_wrapper.py` | Canny edge detection |
| `symmetry.py` | Bilateral symmetry prior |
| `fusion.py` | Multi-prior fusion |

## Results

| Metric | Original | With GeNVS |
|--------|----------|------------|
| Training frames | 21 | 41 (+95%) |
| Final loss | ~0.020 | ~0.015 |
| VRAM usage | 1.3 GB | 1.3 GB |

## Limitations

- **Asymmetric objects**: Symmetry fails for trees, natural scenes
- **Depth supervision**: Novel views lack point_indices
- **Heuristic poses**: Mirrored, not ground truth
