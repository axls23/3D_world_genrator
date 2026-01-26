"""
GeNVS-Core: True 3D-Aware Generative Novel View Synthesis
Reference: 'Generative Novel View Synthesis with 3D-Aware Diffusion Models' (ICCV 2023)

This package contains the faithful implementation of the paper's architecture:
1. Encoder: ResNet-34 + DeepLabV3+ Lifting (2D -> 3D Frustum Grid)
2. Representation: Frustum-Aligned Voxel Volume
3. Renderer: Neural Volume Renderer (Ray Marching) -> Feature Image
4. Backbone: 2D U-Net (ADM) conditioned on rendered features
"""

from .encoder import GeometryEncoder
from .feature_volume import FrustumFeatureVolume
from .rendering import NeuralVolumeRenderer
from .unet_2d import DiffusionUNet
