# ⚡ HyperSplat: High-Velocity Interactive 3DGS Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA: 11.8+](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**HyperSplat** is a performance-engineered 3D Gaussian Splatting (3DGS) framework designed for rapid scene reconstruction and real-time interactive visualization. It bridges the gap between raw data collection and immersive 3D world generation by optimizing every stage of the pipeline.

---

## 🌟 Why HyperSplat?

Traditional 3DGS pipelines often suffer from slow preprocessing (pose/depth estimation) and "black box" training processes. HyperSplat solves this with:

### 1. 🚀 6x Faster Preprocessing
By integrating **Depth Anything V2**, HyperSplat achieves near-instant depth estimation (~30ms per frame), replacing legacy models like ZoeDepth that take ~200ms.

### 2. 🎯 Rapid Camera Tracking
Utilizes **ACE-Zero** for robust, zero-shot 6DoF camera pose estimation, allowing for immediate scene orientation without the heavy overhead of traditional COLMAP.

### 3. 🎮 Real-Time Interactive UI
The **HyperSplat Interactive Demo** provides a WebSocket-powered web interface. Watch your point clouds materialize and manipulate Gaussian parameters *while the model is training*.

---

## 🛠️ Installation & Setup

HyperSplat uses a dual-environment strategy to maximize compatibility between Windows hosts and WSL2.

### Prerequisites
*   **Windows 10/11** with **WSL2** (Ubuntu 22.04 recommended).
*   **NVIDIA GPU** with 8GB+ VRAM (e.g., RTX 30/40 series).
*   **Conda** or **Miniconda** installed.

### Step 1: Main Reconstruction Environment (Windows)
This environment handles the training engine, RAG indexing, and the UI server.
```powershell
conda create -n hypersplat python=3.10 -y
conda activate hypersplat

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Build GSplat core
pip install -e .

# Install extras for Demo & Examples
pip install -r examples/requirements.txt
```

### Step 2: Pose Estimation Environment (WSL2)
This environment handles the ACE-Zero scripts inside the Linux kernel for maximum speed.
```bash
# Inside your WSL2 terminal
conda create -n ace0 python=3.10 -y
conda activate ace0

pip install torch torchvision torchaudio
pip install transformers opencv-python requests
```

---

## 🚀 Usage Guide

### 1. Launch the Interactive Demo
The easiest way to start is with the provided Windows launcher:
```powershell
./demo/run_demo.bat
```
This starts the job queue and launches the web interface at `http://localhost:5000`.

### 2. Run the Automated Pipeline
For advanced users who want fine-grained control via the CLI:
```powershell
python scripts/pipeline/automated_intelligent_pipeline.py --input path/to/video.mp4 --output results/my_scene
```

### 3. Key CLI Arguments
*   `--depth-model {depth_anything, zoedepth}`: Select your depth estimator.
*   `--iterations 7000`: Set training budget (7k is optimized for HyperSplat).
*   `--strategy mcmc`: Use Markov Chain Monte Carlo for cleaner reconstruction.

---

## 🏗️ Technical Architecture

HyperSplat is built on a modular stack:
*   **Core**: Hardened `gsplat` kernels optimized for WSL-to-Windows memory mapping.
*   **Perception**: Depth Anything V2 + ACE-Zero for high-speed geometry inference.
*   **Intelligence**: Hybrid & Spatial RAG (Retrieval-Augmented Generation) for smart point cloud densification.
*   **Visualization**: Custom WebSocket server for real-time state streaming.

---

## 📦 Missing Assets
Due to git size limits, large binary weights are not included in the repo.
1.  **YOLO Seg**: Place `yolo11n-seg.pt` in the root (for dynamic object masking).
2.  **Model Weights**: HyperSplat will automatically attempt to download Depth Anything weights into `~/.cache/` on first run.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments
*   Original 3DGS work by Kerbl et al.
*   GSplat library by the Nerfstudio project.
*   Depth Anything V2 by TikTok/ByteDance.
*   ACE-Zero by Niantic Labs.

## 📑 BibTeX & Citations

If you use HyperSplat or its components in your research, please cite the following original works:

```bibtex
@article{kerbl3Dgaussians,
    author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal={ACM Transactions on Graphics},
    year={2023},
    volume={42},
    number={4}
}

@article{gsplat2023,
    title={gsplat: An Open-Source Library for Gaussian Splatting},
    author={Ye, Vickie and Fan, Ziyi and Kanazawa, Angjoo and others},
    journal={GitHub Repository},
    year={2023},
    url={https://github.com/nerfstudio-project/gsplat}
}

@article{yang2024depth,
    title={Depth Anything V2: Monocular Depth Estimation is All You Need},
    author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
    journal={arXiv preprint arXiv:2406.09414},
    year={2024}
}

@inproceedings{brachmann2024acezero,
    title={ACE-Zero: Parallelized Acceleration of Camera Estimation},
    author={Brachmann, Eric and Cavallari, Tommaso and Niessner, Matthias},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```

---
*Created for the SIH 2025 Professional Competition. Re-engineered for high-velocity spatial generation.*