import sys
from pathlib import Path

# Add hypersplat to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BETA_PATH = PROJECT_ROOT / "hypersplat" / "beta"

if str(BETA_PATH) not in sys.path:
    sys.path.insert(0, str(BETA_PATH))

try:
    from genvs.pipeline import GeNVSPipeline
except ImportError:
    # Handle direct import if beta is already in path as a package
    from hypersplat.beta.genvs.pipeline import GeNVSPipeline
