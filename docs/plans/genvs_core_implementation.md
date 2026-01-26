# GeNVS Core Implementation Plan

**Objective**: Operationalize the "True 3D-Aware" Generative Novel View Synthesis (GeNVS) module, upgrading from the Lite version to the full architecture described in the "GeNVS Technical Deep Dive".

## 1. Current Status (Verified)
The core architectural components have been implemented and validated via `scripts/genvs_core/test_genvs_core.py`:
- **GeometryEncoder**: ResNet-34 + DeepLabV3+ + Lifting (Verified)
- **FrustumFeatureVolume**: 3D Voxel Grid creation (Verified)
- **NeuralVolumeRenderer**: Ray-marching based feature rendering (Verified & Fixed Batching)
- **DiffusionUNet**: 3D-Conditioned 2D U-Net (Verified & Fixed Forward Pass)

## 2. Architecture Alignment (Reference: Technical Deep Dive)
Based on the ICCV 2023 paper and code structure:
- **Pipeline**: `Image -> Encoder -> Volume -> Renderer -> Feature Image -> Concatenation -> Diffusion Denoising -> Output Image`.
- **Key Mechanism**: 3D Geometry acts as a strong structural guidance for the Diffusion model, ensuring generated views are consistent (not just hallucinations).

## 3. Implementation Steps

### Phase 1: Training Infrastructure (Priority)
Since no pretrained weights exist for the core module, a training pipeline is required.
- [ ] **Data Loader**: Implement `GenVSDataset` to load (Source, Target, Pose, K) tuples from standard datasets (e.g., RealEstate10k or simple 3DGS datasets like Mip-NeRF 360).
- [ ] **Training Script**: Create `scripts/genvs_core/train.py`.
    - **Stage 1 (Optional)**: Geometry Pretraining (Train Encoder+Renderer with L2 loss on low-res targets).
    - **Stage 2 (Main)**: End-to-End Diffusion Training (Train UNet + Encoder+Renderer with VLB/MSE noise loss).
- [ ] **Config System**: Extend `config.py` to handle GeNVS Core hyperparameters (LR, batch size, steps).

### Phase 2: Inference & Integration
Once trained weights are available:
- [ ] **Inference Script**: Create `scripts/genvs_core/run_inference.py` to generate N views from 1 input.
- [ ] **3DGS Integration**: Update `automated_intelligent_pipeline.py` to use `genvs_core` if flag `--genvs-mode core` is set.
- [ ] **Optimization**: Ensure FP16/Half-precision inference fits in 6GB VRAM (RTX 4050).

### Phase 3: Advanced Features (From Deep Dive)
- [ ] **Cross-View Attention**: If specified in Deep Dive, upgrade UNet to attend to source features (currently Concatenation).
- [ ] **Refinement**: Post-process with slight MCMC or 3DGS optimization.

## 4. Next Actions
1.  Review "Technical Deep Dive" for specific layer choices (ResNet bottleneck vs others) - *Done (Inferred via code)*.
2.  Implement `train.py` skeleton.
3.  Define Dataset format.

**User Approval Required**:
- Shall we proceed with creating the **Training Script**?
- Do you have a preferred dataset for training (e.g. your own capture data)?
