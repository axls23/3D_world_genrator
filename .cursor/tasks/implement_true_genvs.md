# Task: Implement True GeNVS (3D-Aware Diffusion)

## Status
- [ ] Research & Architecture Design <!-- id: 0 -->
- [ ] Core Components Implementation <!-- id: 1 -->
    - [x] `encoder.py` (ResNet-34 + DeepLabV3+ Lifting) <!-- id: 2 -->
    - [x] `feature_volume.py` (Frustum Volume Management) <!-- id: 3 -->
    - [x] `neural_renderer.py` (Feature Ray Marching) <!-- id: 4 -->
    - [x] `unet_2d.py` (ADM/DDPM++ with 19-channel input) <!-- id: 5 -->
- [x] Pipeline Integration <!-- id: 6 -->
    - [x] `pipeline.py` (Geometric Stage + Generative Stage Loop) <!-- id: 7 -->
    - [x] Integration into `automated_intelligent_pipeline.py` <!-- id: 8 -->

## Context
The user provided a "Technical Deep Dive" PDF clarifying that GeNVS uses a **hybrid architecture**: a 2D Diffusion Model conditioned on a rendered 3D Feature Volume. It does **NOT** use a 3D U-Net. 

## Requirements
1.  **Encoder**: ResNet-34 + DeepLabV3+ (ASPP) -> Lift to Frustum Volume (`16 feat x 64 depth`).
2.  **Representation**: Frustum-aligned Voxel Grid (not Euclidean).
3.  **Renderer**: Volumetric Ray Marching producing a **16-channel Feature Image**.
4.  **Diffusion**: **2D U-Net** (ADM 90M params) taking `3 (RGB) + 16 (Cond)` = 19 input channels.
5.  **Adherence**: Strict following of the dimensions and logic in the provided PDF.

## Technical Approach
*   **Geometry Stage**: Lift Image -> Frustum Grid -> Render Feature Map.
*   **Generative Stage**: Concatenate Noisy RGB + Feature Map -> 2D U-Net -> Denoised RGB.
