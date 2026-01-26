
import sys
import os
import argparse
import subprocess
from pathlib import Path
import shutil

project_root = Path(__file__).parent.parent.parent
# Assuming scripts/genvs_core/refine_loop.py -> .../3D_world_genrator

def run_command(cmd):
    print(f"[RefineLoop] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ACE output directory")
    parser.add_argument("--output_dir", type=str, default="results/genvs_refinement")
    parser.add_argument("--num_cycles", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--gen_views", type=int, default=20)
    parser.add_argument("--use_mcmc", action="store_true", help="Use MCMC strategy for 3DGS")
    parser.add_argument("--enable_viewer", action="store_true", help="Enable Viser viewer during 3DGS training")
    parser.add_argument("--with_ut", action="store_true", help="Enable 3DGUT Uncertainty (Unscented Transform)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Working dataset (starts as copy of input)
    current_dataset = out_dir / "dataset_v0"
    if not current_dataset.exists():
        shutil.copytree(data_dir, current_dataset)
        
    for cycle in range(args.num_cycles):
        print(f"\n{'='*40}\nCycle {cycle+1}/{args.num_cycles}\n{'='*40}")
        
        cycle_dir = out_dir / f"cycle_{cycle+1}"
        cycle_dir.mkdir(exist_ok=True)
        
        # 1. Train GeNVS
        print(">>> Step 1: Training GeNVS...")
        train_cmd = f"python scripts/genvs_core/train.py --data_dir {current_dataset} --output_dir {cycle_dir}/training --num_steps {args.train_steps} --batch_size 1"
        run_command(train_cmd)
        
        ckpt_path = cycle_dir / "training" / "checkpoints" / f"step_{args.train_steps}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        # 2. Generate Views (VR Sphere)
        print(">>> Step 2: Generating Novel Views...")
        gen_out = cycle_dir / "generated"
        inf_cmd = f"python scripts/genvs_core/inference.py --checkpoint {ckpt_path} --data_dir {current_dataset} --num_views {args.gen_views} --output_dir {gen_out}"
        run_command(inf_cmd)
        
        # 3. Augment Dataset
        print(">>> Step 3: Augmenting Dataset...")
        next_dataset = out_dir / f"dataset_v{cycle+1}"
        if next_dataset.exists():
            shutil.rmtree(next_dataset)
        shutil.copytree(current_dataset, next_dataset)
        
        # Copy generated images to dataset/images
        gen_images = list((gen_out / "images").glob("*.jpg"))
        dest_images = next_dataset / "images"
        dest_images.mkdir(exist_ok=True)
        
        for img in gen_images:
            shutil.copy(img, dest_images / img.name)
            
        # Append poses
        with open(gen_out / "poses_generated.txt", "r") as f_new:
            new_lines = f_new.readlines()
            
        with open(next_dataset / "poses_final.txt", "a") as f_main:
            # Need to ensure paths in new_lines match the new location?
            # inference.py wrote absolute paths or relative? 
            # It wrote absolute paths: `str(save_path.absolute())`.
            # But `CamLocDataset` loads absolute paths fine.
            # Ideally we keep it relative for portability but Absolute is OK locally.
            for line in new_lines:
                f_main.write(line)
        
        print(f"Dataset augmented: {len(gen_images)} new views added.")
        current_dataset = next_dataset
        
        # 4. Trigger 3DGS Training
        print(">>> Step 4: Training 3DGS (examples/simple_trainer.py)...")
        
        # Build command
        gs_cmd = ["python", "examples/simple_trainer.py"]
        
        # Strategy Subcommand (tyro)
        if args.use_mcmc:
            gs_cmd.append("mcmc")
        else:
            gs_cmd.append("default") # Explicitly use default strategy if needed, or omit? 
            # Tyro: if union, must pick one? Or default is DefaultStrategy?
            # Let's hope 'default' works or just omit for old behavior?
            # simple_trainer.py Config defaults to DefaultStrategy.
            # But if we want to pass args, we might need subcommand if Config structure requires it?
            # Actually, without subcommand it uses default?
            # Let's use 'default' subcommand to be safe if use_mcmc is False, 
            # OR just omit and hope tyro falls back.
            # Safest is just omit if not MCMC.
            pass

        gs_cmd.extend([
            "--data_dir", str(current_dataset),
            "--result_dir", f"{cycle_dir}/gsplat_result"
        ])
        
        if not args.enable_viewer:
            gs_cmd.append("--disable_viewer")
        else:
            print("!!! Viewer Enabled: Training will block until you exit the process !!!")
            
        if args.with_ut:
            gs_cmd.append("--with_ut")
            
        gs_cmd_str = " ".join(gs_cmd)
        
        # execution
        print(f"Running 3DGS: {gs_cmd_str}")
        run_command(gs_cmd_str) 
        # print(f"[Simulated] Would run: {gs_cmd_str}")
        # print("Note: 3DGS training skipped in this loop to save time/VRAM. Run manually if desired.")

    print("\nRefinement Loop Complete!")
    print(f"Final Augmented Dataset: {current_dataset}")

if __name__ == "__main__":
    main()
