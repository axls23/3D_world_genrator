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
    from hypersplat.pipeline.manager import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"CRITICAL ERROR: HyperSplat package not found. {e}", file=sys.stderr)
    print(f"PYTHONPATH: {sys.path}", file=sys.stderr)
    sys.exit(1)
