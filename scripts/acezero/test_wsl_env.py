import torch
import numpy as np
try:
    import dsacstar
    print("dsacstar import success")
except ImportError as e:
    print(f"dsacstar import failed: {e}")

try:
    from ace_network import Regressor
    print("Regressor import success")
except ImportError as e:
    print(f"Regressor import failed: {e}")

try:
    from dataset import CamLocDataset
    print("CamLocDataset import success")
except ImportError as e:
    print(f"CamLocDataset import failed: {e}")

print("---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
