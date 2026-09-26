# GeNVS (Generative Novel View Synthesis)

Novel view synthesis module for augmenting sparse 3DGS training data, based on
"Generative Novel View Synthesis with 3D-Aware Diffusion Models" (ICCV 2023).

## Overview

GeNVS lifts a 2D image into a 3D-aware feature volume, renders that volume
from a novel camera pose, and uses a conditional diffusion model to turn the
rendered feature image into a photorealistic RGB frame. In this project it is
used to synthesize additional (e.g. back-facing) views to fill coverage gaps
in sparse 3DGS captures, and/or to co-train with 3DGS in a feedback loop.

```
Source Views → Geometry Encoder → Frustum Feature Volume → Neural Renderer → Diffusion (U-Net/DiT) → Novel View
```

## Module Layout

The real implementation lives in `hypersplat/beta/genvs/`:

| File | Purpose |
|------|---------|
| `run_completion.py` | COLMAP model I/O (read/write cameras, images, points3D) plus `inject_back_views()` and `inject_autoregressive_back_views()` — functions that inject synthesized novel views back into a COLMAP-format dataset. Also hosts the `GeNVSLite` class. |
| `pipeline.py` | `GeNVSPipeline` — the main orchestrator that wires together the encoder, feature volume, renderer, and diffusion backbone into a single forward pass. |
| `encoder.py` | `GeometryEncoder` — ResNet-34 + DeepLabV3+ head that lifts a 2D image into a feature representation for the 3D frustum volume. |
| `feature_volume.py` | `FrustumFeatureVolume` — frustum-aligned voxel grid data structure defining the 3D coordinate system used by the renderer. |
| `rendering.py` | `NeuralVolumeRenderer` — differentiable ray-marching renderer that samples the frustum feature volume (stratified sampling, trilinear interpolation, MLP decoder) to produce a rendered feature image. |
| `unet_2d.py` | `DiffusionUNet` — 2D U-Net (ADM/DDPM++ style) diffusion backbone conditioned on the rendered feature image via channel concatenation. |
| `dit_2d.py` | Diffusion Transformer (DiT) building blocks (timestep embedding, `modulate`, `DiT_S_2`, etc.) — an alternative/experimental backbone to the U-Net. |
| `train.py` | 3-phase GeNVS trainer entry point (`argparse` CLI, `description="GeNVS 3-Phase Trainer"`). |
| `inference.py` | Standalone CLI for running inference with a trained GeNVS checkpoint. |
| `refine_loop.py` | Recursive GeNVS ↔ 3DGS co-training feedback loop: trains GeNVS, generates novel views, augments the dataset, trains 3DGS, renders 3DGS at novel poses, and augments again for the next cycle. |
| `render_from_3dgs.py` | Standalone 3DGS renderer used by `refine_loop.py` to produce geometrically-consistent feedback images at specified camera poses from a trained 3DGS checkpoint. |
| `dataset.py` | `GenVSDataset` — dataset loader used by training/inference. |

`inject_back_views` and `inject_autoregressive_back_views` are **functions inside `run_completion.py`**, not standalone scripts — there is no `scripts/genvs/inject_back_views.py` or `scripts/genvs/diffusion.py` in the current codebase (those paths were removed in a prior refactor).

## Backward-Compat Shim

`scripts/genvs_core/` is a thin shim package (`__init__.py`, `pipeline.py`, `run_completion.py`, `train.py`) that re-exports the classes above (`GeometryEncoder`, `FrustumFeatureVolume`, `NeuralVolumeRenderer`, `DiffusionUNet`, etc.) so older CLI invocations and imports that reference `scripts.genvs_core.*` continue to work. New code should import directly from `hypersplat.beta.genvs`.

## Usage

### Integrated Pipeline

```bash
python scripts/pipeline/automated_intelligent_pipeline.py video.mp4 \
    --genvs \
    --genvs-views 20
```

This is wired through `hypersplat/pipeline/manager.py`'s `--genvs`/`--genvs-views`/`--genvs_ckpt`/`--genvs_interval`/`--genvs-autoregressive` flags.

### Standalone

Refer to each module's own `argparse` CLI (`train.py`, `inference.py`, `render_from_3dgs.py`) for exact flags — there is currently no single unified standalone entry point equivalent to the old `scripts/genvs/run_completion.py --images ... --output ...` invocation described in earlier drafts of this doc.

## Limitations

- **Asymmetric objects**: symmetry/geometry priors are weaker for trees, natural scenes.
- **Depth supervision**: synthesized novel views lack point-cloud correspondences.
- **Heuristic/approximate poses**: injected views are not ground-truth camera poses.
