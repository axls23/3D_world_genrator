"""
SHIM: Redirects imports to hypersplat.pipeline.wrapper
"""
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from hypersplat.pipeline.wrapper import TrainingManager, PipelineMode
except ImportError as e:
    raise ImportError(f"Could not import hypersplat.pipeline.wrapper: {e}")
