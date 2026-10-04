#!/usr/bin/env bash
set -euo pipefail
WT=/mnt/Shared/3D_WORLD_GENRATION/3D_world_genrator/.claude/worktrees/agent-a65818da08341b288
cd "$WT"
export PYTHONPATH="$WT" TORCH_EXTENSIONS_DIR="$WT/.torch_ext" HF_HOME=/mnt/Shared/3D_WORLD_GENRATION/hf_cache
exec flock /mnt/Shared/3D_WORLD_GENRATION/3D_world_genrator/experiments/.gpu.lock "$WT/.venv_3dgrut/bin/python" experiments/difix_noise_probe.py
