#!/bin/bash
source ~/.bashrc
conda activate ace0 || source ~/miniconda3/etc/profile.d/conda.sh && conda activate ace0
cd /mnt/c/Users/sxhil_25660/Documents/GitHub/SIH_2025_Professional/gsplat/scripts/acezero
export MKL_THREADING_LAYER=GNU
echo "Starting training..."
python -u train_ace.py "/tmp/debug_img/*.jpg" /tmp/debug_out/seed.pt --use_pose_seed 0 --iterations 100 --use_heuristic_focal_length True
echo "Starting registration..."
python -u register_mapping.py "/tmp/debug_img/*.jpg" /tmp/debug_out/seed.pt --session final --render_visualization False
