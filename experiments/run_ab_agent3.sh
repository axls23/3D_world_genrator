#!/usr/bin/env bash
# Agent 3 A/B runner: worktree code, private CUDA JIT dir, shared HF cache and GPU lock.
set -euo pipefail
WT=/mnt/Shared/3D_WORLD_GENRATION/3D_world_genrator/.claude/worktrees/agent-a65818da08341b288
MAIN=/mnt/Shared/3D_WORLD_GENRATION/3D_world_genrator
cd "$WT"
export PYTHONPATH="$WT"
export TORCH_EXTENSIONS_DIR="$WT/.torch_ext"
export HF_HOME=/mnt/Shared/3D_WORLD_GENRATION/hf_cache
# The ace0 env is found relative to the repo (parents[4]): a worktree sits two levels deeper
export HYPERSPLAT_ACEZERO_PYTHON=/mnt/Shared/3D_WORLD_GENRATION/.conda_envs/ace0/bin/python
exec "$WT/.venv_3dgrut/bin/python" "$WT/scripts/experiments/ab_test.py" "$@" --lock "$MAIN/experiments/.gpu.lock"
