#!/bin/bash
# ACE-Zero container entrypoint
# Starts Xvfb for headless OpenGL and keeps container alive for docker exec

echo "[entrypoint] ACE-Zero container starting..."

# Verify GPU access
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
    echo "[entrypoint] GPU access verified."
else
    echo "[entrypoint] WARNING: No GPU detected. ACE-Zero will fail."
fi

# Verify Python + PyTorch
python -c "import torch; print(f'[entrypoint] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Start Xvfb for headless OpenGL (pyrender needs it)
Xvfb :99 -screen 0 1024x768x24 &>/dev/null &

echo "[entrypoint] Container ready. Sleeping..."
exec sleep infinity
