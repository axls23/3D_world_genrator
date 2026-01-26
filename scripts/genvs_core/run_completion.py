import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BETA_PATH = PROJECT_ROOT / "hypersplat" / "beta"

if str(BETA_PATH) not in sys.path:
    sys.path.insert(0, str(BETA_PATH))

from genvs.run_completion import GeNVSLite, read_model, mirror_camera_pose, compute_scene_center, qvec2rotmat, inject_back_views
