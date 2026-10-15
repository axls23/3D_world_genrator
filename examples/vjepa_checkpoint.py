"""V-JEPA 2 checkpoint path resolution and the startup check for latent_model="vjepa".

Kept free of CUDA / gsplat imports so it can be unit tested on CPU
(tests/pipeline/test_vjepa_checkpoint.py).
"""

import os
import zipfile
from typing import Optional

VJEPA_CHECKPOINT_DEFAULT = "/mnt/Shared/3D_WORLD_GENRATION/hf_cache/vjepa2/vitl.pt"
VJEPA_CHECKPOINT_ENV = "HYPERSPLAT_VJEPA_CKPT"
VJEPA_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"


def default_vjepa_checkpoint() -> str:
    """Config default: $HYPERSPLAT_VJEPA_CKPT if set, else the shared cache path.

    Read when the Config is built, so an explicit --vjepa_checkpoint still wins."""
    return os.environ.get(VJEPA_CHECKPOINT_ENV) or VJEPA_CHECKPOINT_DEFAULT


def missing_vjepa_checkpoint(latent_loss: bool, latent_model: str, checkpoint: Optional[str]) -> Optional[str]:
    """Error message when the run needs the V-JEPA checkpoint and it is missing or unreadable, else None."""
    if not latent_loss or latent_model != "vjepa":
        return None
    if not checkpoint:
        problem = "Not found: (vjepa_checkpoint is unset)"
    elif not os.path.isfile(checkpoint):
        problem = f"Not found: {checkpoint}"
    # torch.save files are zip archives; a truncated download has no central directory and would
    # otherwise fail later, in torch.load
    elif not zipfile.is_zipfile(checkpoint):
        problem = f"Not a torch checkpoint (truncated download?): {checkpoint}"
    else:
        return None
    return (
        "[V-JEPA] latent_loss with latent_model=vjepa needs the V-JEPA 2 ViT-L checkpoint (vitl.pt).\n"
        f"  {problem}\n"
        f"  Download:  {VJEPA_CHECKPOINT_URL}\n"
        f"             then pass --vjepa_checkpoint PATH or set {VJEPA_CHECKPOINT_ENV}=PATH\n"
        "  Opt out:   latent_model=feature, the plain frozen-ViT feature loss\n"
        "             (trainer: --latent_model feature; pipeline: --trainer-arg latent_model=feature),\n"
        "             or no latent loss (demo: run without --jepa)"
    )
