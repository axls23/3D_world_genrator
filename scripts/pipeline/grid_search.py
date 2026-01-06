import argparse
import subprocess
import shutil
from pathlib import Path
import itertools
import sys
import time

# Configuration
DEFAULT_VIDEO = "demo/videos/garden.mp4"
BASE_OUTPUT_DIR = Path("data/grid_search")
ACE_SHARED_DIR = BASE_OUTPUT_DIR / "shared_ace"

# Hyperparameter Grid
GRID = {
    "opacity_reg": [0.01, 0.05, 0.1],
    "scale_reg": [0.01, 0.1]
}

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="3DGS Hyperparameter Grid Search")
    parser.add_argument("video_path", default=DEFAULT_VIDEO, nargs="?")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    # 1. Run Baseline (ACE-Zero Only) if not exists
    print(f"\n{'='*50}\nSTEP 1: Shared ACE-Zero Pose Estimation\n{'='*50}")
    ace_output_path = ACE_SHARED_DIR / "acezero_output"
    
    if not ace_output_path.exists() or not (ace_output_path / "poses_final.txt").exists():
        print("Running ACE-Zero to generate shared poses...")
        cmd = [
            sys.executable, "gsplat/scripts/pipeline/automated_intelligent_pipeline.py",
            str(video_path),
            "--output_dir", str(ACE_SHARED_DIR),
            "--max_steps", "1", # Dummy training, we only care about ACE
            "--no-prune"
        ]
        run_command(cmd)
    else:
        print("Shared ACE-Zero output already exists. Skipping.")

    # 2. Run Grid
    keys = list(GRID.keys())
    values = list(GRID.values())
    combinations = list(itertools.product(*values))

    print(f"\n{'='*50}\nSTEP 2: Grid Search ({len(combinations)} trials)\n{'='*50}")
    
    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        trial_name = f"trial_{i}_" + "_".join([f"{k}{v}" for k, v in params.items()])
        trial_dir = BASE_OUTPUT_DIR / trial_name
        
        print(f"\n--- Running Trial {i+1}/{len(combinations)}: {trial_name} ---")
        print(f"Params: {params}")

        # Setup Trial Directory with ACE Copy
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True)
        
        # Copy ACE output (use symlink if possible on Windows? Copy is safer)
        print("Copying shared ACE output...")
        shutil.copytree(ace_output_path, trial_dir / "acezero_output")

        # Run Pipeline with Skip ACE
        cmd = [
            sys.executable, "gsplat/scripts/pipeline/automated_intelligent_pipeline.py",
            str(video_path),
            "--output_dir", str(trial_dir),
            "--skip-ace",
            "--max_steps", "7000",
            "--prune",
            "--with_eval3d" # Metrics
        ]
        
        # Append hyperparameters
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])

        start_time = time.time()
        try:
            run_command(cmd)
            elapsed = time.time() - start_time
            results.append({"name": trial_name, "status": "Success", "time": elapsed, "params": params})
        except Exception as e:
            print(f"Trial failed: {e}")
            results.append({"name": trial_name, "status": "Failed", "error": str(e), "params": params})

    # 3. Report
    print(f"\n{'='*50}\nGRID SEARCH COMPLETE\n{'='*50}")
    print(f"{'Trial':<40} | {'Status':<10} | {'Time (s)':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<40} | {r['status']:<10} | {r.get('time', 0):<10.1f}")

if __name__ == "__main__":
    main()
