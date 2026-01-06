# ⚡ HyperSplat: High-Velocity Interactive 3DGS Engine

HyperSplat is a performance-optimized 3D Gaussian Splatting (3DGS) pipeline designed for rapid scene reconstruction and interactive visualization. 

By integrating state-of-the-art models like **Depth Anything V2** and **ACE-Zero**, HyperSplat delivers a 6x speedup in preprocessing, allowing you to go from raw video to an immersive 3D scene in record time.

## 🌟 Key Features
*   **6x Faster Depth Estimation**: Integrated Depth Anything V2 for near-instant ( ~30ms/frame) depth maps.
*   **Instant Pose Estimation**: Leverages ACE-Zero for rapid, zero-shot 6DoF camera tracking.
*   **Live Interactive Viewer**: A rich, WebSocket-based interface (`demo/`) for real-time monitoring and manipulation of the training process.
*   **WSL-Optimized Core**: Specialized CUDA kernels and strategies tuned for performance and stability in WSL2 environments.

## 🛠️ Quick Setup
1.  **Reconstruction Env**:
    ```bash
    conda create -n hypersplat python=3.10 -y
    conda activate hypersplat
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -e .
    pip install -r examples/requirements.txt
    ```
2.  **Pose Estimation (WSL)**:
    ```bash
    conda activate ace0 # Ensure ace0 env exists
    pip install transformers opencv-python requests
    ```

## 🚀 Launch the Engine
Open your terminal and run:
```powershell
./demo/run_demo.bat
```

---
*Developed for high-speed spatial intelligence and rapid prototyping.*