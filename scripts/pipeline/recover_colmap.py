import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add scripts to path to find ace_zero_wrapper
sys.path.append(r"c:\Users\sxhil_25660\3D_world_genrator\scripts\pipeline")

from ace_zero_wrapper import ACEZeroPoseEstimator

def main():
    # Define paths
    output_dir = Path(r"c:\Users\sxhil_25660\3D_world_genrator\demo\test_run_genvs_gpu_max_seed2089")
    
    
    # Initialize wrapper with existing output dir
    wrapper = ACEZeroPoseEstimator(output_dir)
    wrapper._setup_directories()
    
    # CORRECT: Move the nested file to where the wrapper expects it
    nested_pose_file = output_dir / "acezero_output" / "acezero_output" / "poses_final.txt"
    expected_pose_file = output_dir / "acezero_output" / "poses_final.txt"
    
    if nested_pose_file.exists():
        logger.info(f"Moving nested pose file from {nested_pose_file} to {expected_pose_file}")
        import shutil
        shutil.copy2(nested_pose_file, expected_pose_file)
    else:
        logger.warning(f"Nested pose file NOT found at {nested_pose_file}. Hoping it is already in place.")

    # 3. Parse output (populates wrapper.poses)
    if wrapper._parse_acezero_output():
        logger.info(f"Successfully parsed ACE-Zero output. Loaded {len(wrapper.poses)} poses.")
        logger.info(f"Target sparse directory: {wrapper.sparse_dir}")
        wrapper.sparse_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
        
        # 4. Write COLMAP files (images.bin, cameras.bin)
        wrapper.write_colmap_format()
        
        # 5. Generate Point Cloud (points3D.ply)
        # This calls 'generate_points_wsl.py', which expects 'poses_final.txt' at the standard location
        logger.info("Generating initial point cloud...")
        wrapper.generate_initial_points()
        
        # 6. Downsample images
        logger.info("Creating downsampled images...")
        wrapper.create_downsampled_images(factors=[2, 4])
        
        logger.info("Manual recovery complete. Ready for GeNVS/3DGS.")
    else:
        logger.error("Failed to parse ACE-Zero poses!")

if __name__ == "__main__":
    main()
