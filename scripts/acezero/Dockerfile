# ============================================================
# ACE-Zero Docker Image — Lean Build
# Uses base PyTorch image directly (2.5.1 + CUDA 11.8)
# No redundant conda env — pip-install missing deps only
# Code is mounted at /workspace via docker-compose volume
# ============================================================
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel

WORKDIR /workspace

# System deps (OpenGL for pyrender, ffmpeg for video)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget \
        libgl1-mesa-glx libglib2.0-0 \
        libegl1-mesa libgles2-mesa \
        xvfb mesa-utils \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install ACE-Zero Python deps into base env
# (base image already has: torch 2.5.1, torchvision, numpy, pillow, etc.)
RUN pip install --no-cache-dir \
    scipy \
    scikit-image \
    opencv-python-headless \
    matplotlib \
    trimesh \
    pyrender \
    roma \
    timm>=0.6.7 \
    pycolmap>=0.4.0 \
    pyyaml \
    joblib \
    tqdm

# dsacstar C++ CUDA extension requires GPU at compile time.
# We defer its build to container startup via entrypoint.sh.
ENV CONDA_PREFIX=/opt/conda

# Xvfb for headless OpenGL (pyrender needs it)
ENV DISPLAY=:99
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default working directory = mounted project scripts
WORKDIR /workspace/scripts/acezero

# Entrypoint builds dsacstar on first run, then sleeps
ENTRYPOINT ["/entrypoint.sh"]