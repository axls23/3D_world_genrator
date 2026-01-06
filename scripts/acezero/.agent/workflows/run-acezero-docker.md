---
description: How to run ACE-Zero via Docker for reliable pose estimation
---

# ACE-Zero Docker Workflow

This workflow allows you to run ACE-Zero in a Docker container, bypassing all Windows-specific compatibility issues (EGL/PyRender).

## Prerequisites
- Docker Desktop installed with WSL2 backend
- NVIDIA Container Toolkit installed
- GPU with CUDA support

## Steps

### 1. Build the Docker image (one-time setup, ~15 mins)
```powershell
cd C:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat\scripts\acezero
docker-compose build
```

### 2. Start the container
```powershell
docker-compose up -d
```

### 3. Enter the container
```powershell
docker exec -it acezero bash
```

### 4. Run ACE-Zero inside the container
```bash
# Activate the environment
conda activate ace0

# Run pose estimation on your video frames
# Frames should be in /workspace/data/output_3dgs/acezero_output/images/
python ace_zero.py /workspace/data/output_3dgs/acezero_output/images/*.jpg /workspace/data/output_3dgs/acezero_output/acezero_output --iterations_max 5
```

### 5. Exit the container
```bash
exit
```

### 6. Stop the container (when done)
```powershell
docker-compose down
```

## Notes
- The container has GPU access via nvidia-docker
- EGL/Xvfb is pre-configured for visualization
- Your local data is mounted at `/workspace/data`
- Your Downloads folder is mounted at `/workspace/downloads` (read-only)
