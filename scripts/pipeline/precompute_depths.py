#!/usr/bin/env python3
"""
SHIM: Redirects legacy script calls to the new HyperSplat package.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from hypersplat.pipeline.steps.depth import precompute_depths, main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"CRITICAL ERROR: HyperSplat package not found. {e}", file=sys.stderr)
    sys.exit(1)
