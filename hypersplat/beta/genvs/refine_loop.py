"""
refine_loop.py — Recursive GeNVS ↔ 3DGS Co-Training Feedback Loop

Each cycle:
  1. Train GeNVS on current dataset (learns to hallucinate textures)
  2. Generate novel views with GeNVS (fill coverage gaps)
  3. Augment dataset with GeNVS views
  4. Train 3DGS on augmented dataset (geometric filter)
  5. Render 3DGS at novel poses (geometrically consistent feedback)
  6. Augment dataset with 3DGS renders for next GeNVS cycle

The key insight: GeNVS provides texture hallucination, 3DGS provides geometric
consistency. By ping-ponging between them, each system bootstraps the other.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
import shutil
import json

project_root = Path(__file__).parent.parent.parent.parent

def run_command(cmd, description=""):
    print(f"\n{'='*60}")
    print(f"[RefineLoop] {description}")
    print(f"[RefineLoop] Running: {cmd}")
    print(f"{'='*60}")
    if os.name == 'nt':
        ret = os.system(f'"{cmd}"')
    else:
        ret = os.system(cmd)
    if ret != 0:
        print(f"[RefineLoop] WARNING: Command exited with code {ret}")
        return False
    return True

def find_latest_ckpt(ckpt_dir: Path, pattern="ckpt_*_rank0.pt"):
    """Find the latest checkpoint in a directory."""
    candidates = sorted(ckpt_dir.glob(pattern))
    if candidates:
        return candidates[-1]
    # Also try step_*.pt pattern (GeNVS format)
    candidates = sorted(ckpt_dir.glob("step_*.pt"))
    if candidates:
        return candidates[-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Recursive GeNVS ↔ 3DGS Co-Training Loop")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ACE output directory")
    parser.add_argument("--output_dir", type=str, default="results/genvs_refinement")
    parser.add_argument("--num_cycles", type=int, default=2, help="Number of refinement cycles (2 = laptop-friendly)")
    
    # GeNVS settings
    parser.add_argument("--genvs_train_steps", type=int, default=500, help="GeNVS training steps per cycle")
    parser.add_argument("--genvs_gen_views", type=int, default=10, help="Novel views to generate with GeNVS")
    
    # 3DGS settings
    parser.add_argument("--gs_max_steps", type=int, default=7000, help="3DGS training steps (7K = laptop-friendly)")
    parser.add_argument("--gs_final_steps", type=int, default=15000, help="3DGS steps for final cycle")
    parser.add_argument("--use_mcmc", action="store_true", help="Use MCMC strategy for 3DGS")
    parser.add_argument("--disable_viewer", action="store_true", default=True, help="Disable Viser viewer")
    
    # Feedback settings
    parser.add_argument("--feedback_views", type=int, default=10, help="3DGS renders to feed back to GeNVS")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Working dataset (starts as copy of input)
    current_dataset = out_dir / "dataset_v0"
    if not current_dataset.exists():
        print(f"[RefineLoop] Copying initial dataset to {current_dataset}")
        shutil.copytree(data_dir, current_dataset)
    
    genvs_ckpt = None  # Will be set after first training
    
    # Save run config
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for cycle in range(args.num_cycles):
        is_final = (cycle == args.num_cycles - 1)
        
        print(f"\n{'#'*60}")
        print(f"  CYCLE {cycle+1}/{args.num_cycles}  {'(FINAL)' if is_final else ''}")
        print(f"{'#'*60}")
        
        cycle_dir = out_dir / f"cycle_{cycle+1}"
        cycle_dir.mkdir(exist_ok=True)
        
        # ─────────────────────────────────────────────────────────
        # STEP 1: Train GeNVS
        # ─────────────────────────────────────────────────────────
        genvs_train_dir = cycle_dir / "genvs_training"
        
        resume_flag = ""
        if genvs_ckpt is not None and genvs_ckpt.exists():
            resume_flag = f" --resume {genvs_ckpt}"
            print(f"[RefineLoop] Resuming GeNVS from {genvs_ckpt}")
        
        train_cmd = (
            f'"{sys.executable}" hypersplat/beta/genvs/train.py '
            f'--data_dir "{current_dataset}" '
            f'--output_dir "{genvs_train_dir}" '
            f'--num_steps {args.genvs_train_steps} '
            f'--batch_size 1'
            f'{resume_flag}'
        )
        success = run_command(train_cmd, f"Step 1: Training GeNVS ({args.genvs_train_steps} steps)")
        
        # Find the latest checkpoint
        ckpt_dir = genvs_train_dir / "checkpoints"
        if ckpt_dir.exists():
            genvs_ckpt = find_latest_ckpt(ckpt_dir)
            # Storage Optimization: Delete all older checkpoints
            if genvs_ckpt is not None:
                for ckpt_file in ckpt_dir.glob("*.pt"):
                    if ckpt_file != genvs_ckpt:
                        try:
                            ckpt_file.unlink()
                        except OSError:
                            pass
        
        if genvs_ckpt is None or not genvs_ckpt.exists():
            print(f"[RefineLoop] WARNING: No GeNVS checkpoint found. Skipping generation.")
            continue
            
        # ─────────────────────────────────────────────────────────
        # STEP 2 & 3: Convergence Gating & Dataset Augmentation
        # ─────────────────────────────────────────────────────────
        next_dataset = out_dir / f"dataset_v{cycle+1}"
        if next_dataset.exists():
            shutil.rmtree(next_dataset)
            
        def link_if_possible(src, dst):
            if src.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
                
        shutil.copytree(current_dataset, next_dataset, copy_function=link_if_possible)

        # Check if GeNVS has converged enough to generate useful views
        genvs_converged = False
        metrics_csv = genvs_train_dir / "metrics" / "training_log.csv"
        if metrics_csv.exists():
            with open(metrics_csv, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(",")
                    try:
                        ema_loss = float(last_line[2])
                        if ema_loss <= 0.1: # Threshold for useful diffusion
                            genvs_converged = True
                        else:
                            print(f"[RefineLoop] GeNVS EMA Loss {ema_loss:.4f} > 0.1. Skipping generation as it would be pure noise.")
                    except ValueError:
                        pass
        
        if genvs_converged:
            gen_out = cycle_dir / "genvs_generated"
            
            inf_cmd = (
                f'"{sys.executable}" hypersplat/beta/genvs/inference.py '
                f'--checkpoint "{genvs_ckpt}" '
                f'--data_dir "{current_dataset}" '
                f'--num_views {args.genvs_gen_views} '
                f'--output_dir "{gen_out}"'
            )
            run_command(inf_cmd, f"Step 2: Generating {args.genvs_gen_views} novel views with GeNVS")
            
            # Copy generated images
            gen_images_dir = gen_out / "images"
            if gen_images_dir.exists():
                dest_images = next_dataset / "acezero_output" / "images"
                dest_images.mkdir(parents=True, exist_ok=True)
                
                gen_images = list(gen_images_dir.glob("*.jpg")) + list(gen_images_dir.glob("*.png"))
                for img in gen_images:
                    shutil.copy(img, dest_images / img.name)
                
                # Append poses
                gen_pose_file = gen_out / "poses_generated.txt"
                if gen_pose_file.exists():
                    main_pose_file = next_dataset / "acezero_output" / "poses_final.txt"
                    with open(gen_pose_file, "r") as f_new:
                        new_lines = f_new.readlines()
                    with open(main_pose_file, "a") as f_main:
                        for line in new_lines:
                            f_main.write(line)
                            
                print(f"[RefineLoop] Dataset augmented: {len(gen_images)} GeNVS views added.")
        
        # ─────────────────────────────────────────────────────────
        # STEP 4: Train 3DGS on augmented dataset
        # ─────────────────────────────────────────────────────────
        gs_steps = args.gs_final_steps if is_final else args.gs_max_steps
        gs_result_dir = cycle_dir / "gsplat_result"
        
        strategy = "mcmc" if args.use_mcmc else "default"
        viewer_flag = "--disable_viewer" if args.disable_viewer else ""
        
        gs_cmd = (
            f'"{sys.executable}" examples/simple_trainer.py {strategy} '
            f'--data_dir "{next_dataset}" '
            f'--result_dir "{gs_result_dir}" '
            f'--max_steps {gs_steps} '
            f'--save_steps {gs_steps} '
            f'{viewer_flag}'
        )
        run_command(gs_cmd, f"Step 4: Training 3DGS ({gs_steps} steps)")
        
        # ─────────────────────────────────────────────────────────
        # STEP 5: Render 3DGS feedback views (THE FEEDBACK LOOP)
        # ─────────────────────────────────────────────────────────
        if not is_final:
            gs_ckpt_dir = gs_result_dir / "ckpts"
            gs_ckpt = find_latest_ckpt(gs_ckpt_dir, "ckpt_*_rank0.pt")
            
            # Storage Optimization: Delete all older checkpoints
            if gs_ckpt is not None and gs_ckpt.exists():
                for ckpt_file in gs_ckpt_dir.glob("*.pt"):
                    if ckpt_file != gs_ckpt:
                        try:
                            ckpt_file.unlink()
                        except OSError:
                            pass
                            
                feedback_dir = cycle_dir / "3dgs_feedback"
                
                render_cmd = (
                    f'"{sys.executable}" hypersplat/beta/genvs/render_from_3dgs.py '
                    f'--ckpt "{gs_ckpt}" '
                    f'--data_dir "{next_dataset}" '
                    f'--output_dir "{feedback_dir}" '
                    f'--num_views {args.feedback_views}'
                )
                run_command(render_cmd, f"Step 5: Rendering {args.feedback_views} 3DGS feedback views")
                
                # ─────────────────────────────────────────────────
                # STEP 6: Augment dataset with 3DGS renders
                # ─────────────────────────────────────────────────
                feedback_images_dir = feedback_dir / "images"
                if feedback_images_dir.exists():
                    dest_images = next_dataset / "acezero_output" / "images"
                    
                    fb_images = list(feedback_images_dir.glob("*.jpg")) + list(feedback_images_dir.glob("*.png"))
                    for img in fb_images:
                        shutil.copy(img, dest_images / img.name)
                    
                    # Append 3DGS-rendered poses
                    fb_pose_file = feedback_dir / "poses_3dgs_rendered.txt"
                    if fb_pose_file.exists():
                        main_pose_file = next_dataset / "acezero_output" / "poses_final.txt"
                        with open(fb_pose_file, "r") as f_new:
                            new_lines = f_new.readlines()
                        with open(main_pose_file, "a") as f_main:
                            for line in new_lines:
                                f_main.write(line)
                    
                    print(f"[RefineLoop] Feedback injected: {len(fb_images)} 3DGS renders added to dataset.")
                else:
                    print(f"[RefineLoop] WARNING: No feedback images found at {feedback_images_dir}")
            else:
                print(f"[RefineLoop] WARNING: No 3DGS checkpoint found. Skipping feedback.")
        
        current_dataset = next_dataset
        
        # Log cycle summary
        summary = {
            "cycle": cycle + 1,
            "dataset": str(current_dataset),
            "genvs_ckpt": str(genvs_ckpt) if genvs_ckpt else None,
            "gs_steps": gs_steps,
            "is_final": is_final,
        }
        with open(cycle_dir / "cycle_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  REFINEMENT LOOP COMPLETE")
    print(f"  Final Dataset: {current_dataset}")
    print(f"  Cycles: {args.num_cycles}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
