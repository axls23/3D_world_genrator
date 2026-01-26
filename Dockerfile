# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
    PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda clean --all --yes

# Set working directory
WORKDIR /app

# Install PyTorch and core dependencies directly (simpler than yml)
RUN /opt/conda/bin/pip install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
RUN /opt/conda/bin/pip install --no-cache-dir \
    numpy ninja jaxtyping rich typing_extensions \
    fastapi uvicorn python-multipart aiofiles pydantic websockets \
    opencv-python-headless tqdm scipy boto3 plyfile einops matplotlib \
    tyro viser ultralytics filterpy gdown scikit-learn scikit-image \
    kornia lpips open3d trimesh transformers accelerate timm

# Copy project files
COPY . .

# Install gsplat package (with --no-build-isolation to use installed torch)
RUN /opt/conda/bin/pip install -v -e . --no-build-isolation

# Expose ports
EXPOSE 8080
EXPOSE 8092

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["uvicorn", "hypersplat.services.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
