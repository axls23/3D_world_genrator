#!/usr/bin/env python3
"""
SHIM: Redirects legacy script calls to the new HyperSplat package.
"""
import sys
from pathlib import Path

# Add project root to sys.path to allow `import hypersplat`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Expose the class so `from scripts.pipeline.ace_zero_wrapper import ACEZeroPoseEstimator` still works
    from hypersplat.pipeline.wrappers.perception import ACEZeroPoseEstimator
    
    # If run as a script for testing
    if __name__ == "__main__":
        # Mimic exact behavior of original argparse block
        import argparse
        parser = argparse.ArgumentParser(description="ACE-Zero Pose Estimator Wrapper (Shim)")
        parser.add_argument("video_path", help="Path to input video file")
        parser.add_argument("--output_dir", default="./data/acezero", help="Output directory")
        parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate")
        # Add minimal required args for basic run
        parser.add_argument("--quality_mode", type=str, default="balanced")
        parser.add_argument("--min_confidence", type=int, default=1000)
        
        args, unknown = parser.parse_known_args()
        
        estimator = ACEZeroPoseEstimator(
            args.output_dir,
            quality_mode=args.quality_mode,
            min_confidence=args.min_confidence
        )
        estimator.process_video(args.video_path, fps=args.fps)
        
except ImportError as e:
    print(f"CRITICAL ERROR: HyperSplat package not found. {e}", file=sys.stderr)
    sys.exit(1)
