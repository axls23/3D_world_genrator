"""Frozen joint-embedding (DINOv2 / JEPA-family) features for latent-space losses.

    render (B, 3, H, W) --resize to 14-px multiple--> frozen ViT --patch tokens (B, N, C)
    target (B, 3, H, W) --same, no grad-------------> frozen ViT --patch tokens (B, N, C)
    loss = 1 - cos(tokens_render, tokens_target)

Unlike ``DepthEstimator`` the forward pass is differentiable w.r.t. the *input image* (weights
are frozen, but there is no ``no_grad``), so the loss back-propagates into the rasterizer.
``target_features`` is the no-grad twin for ground-truth frames and can be cached per frame.
"""

import logging
import threading
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from hypersplat.stream.hf_lock import hf_load

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/dinov2-small"
DA_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DA_LAYERS = (3, 6, 9, 12)  # Depth Anything V2-S backbone out_indices
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
PATCH = 14


class LatentEncoder:
    """Frozen ViT features (transformers). Loaded lazily; weights come from HF_HOME.

    ``backend="dinov2"`` is plain DINOv2; ``backend="depth_anything"`` uses the DINOv2 backbone
    of Depth Anything V2 (depth-tuned features, already cached, and ``model`` may be the
    ``DepthAnythingForDepthEstimation`` the trainer loaded so no second copy is held).

    ``layer`` picks the hidden state (-1 = last). Shallower layers keep more spatial detail,
    deeper ones are more invariant -- the knob to turn if the latent loss falls but PSNR doesn't.
    For ``depth_anything`` the layer must be one of ``DA_LAYERS`` (or -1).
    """

    def __init__(self, model_id: Optional[str] = None, input_side: int = 224, layer: int = -1,
                 device: str = "cuda", half: bool = True, backend: str = "dinov2", model=None):
        if backend not in ("dinov2", "depth_anything"):
            raise ValueError(f"unknown latent backend {backend!r}")
        if backend == "depth_anything" and layer != -1 and layer not in DA_LAYERS:
            raise ValueError(f"depth_anything layer must be -1 or one of {DA_LAYERS}, got {layer}")
        self.backend = backend
        self.model_id = model_id or (DA_MODEL if backend == "depth_anything" else DEFAULT_MODEL)
        self.input_side = input_side  # long side fed to the ViT (rounded to a multiple of 14)
        self.layer = layer
        self.device = device
        self.half = half and device.startswith("cuda")
        self.model = model  # shared Depth Anything model (backend="depth_anything"), or None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.model is None:
                dtype = torch.float16 if self.half else torch.float32
                if self.backend == "depth_anything":
                    from transformers import DepthAnythingForDepthEstimation as Cls
                else:
                    from transformers import Dinov2Model as Cls
                with hf_load():
                    # _fast_init=False: fast init swaps torch.nn.init out process-wide while it loads
                    self.model = Cls.from_pretrained(self.model_id, _fast_init=False).to(self.device, dtype).eval()
                logger.info(f"[Latent] {self.model_id} loaded on {self.device} (layer {self.layer})")
            self.model.requires_grad_(False)  # harmless for a shared model: depth runs no_grad
        return self

    def grid(self, h: int, w: int) -> Tuple[int, int]:
        """Patch grid (gh, gw) the ViT sees for an (h, w) image; ``features`` returns gh*gw tokens."""
        s = self.input_side / max(h, w)
        ih = max(PATCH, int(round(h * s / PATCH)) * PATCH)
        iw = max(PATCH, int(round(w * s / PATCH)) * PATCH)
        return ih // PATCH, iw // PATCH

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float in [0, 1] -> ImageNet-normalised, resized to a 14-px multiple."""
        _, _, h, w = img.shape
        gh, gw = self.grid(h, w)
        ih, iw = gh * PATCH, gw * PATCH
        x = F.interpolate(img, size=(ih, iw), mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor(_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def features(self, img: torch.Tensor) -> torch.Tensor:
        """Differentiable patch tokens (B, N, C), L2-normalised per token. CLS token dropped."""
        self.load()
        x = self._prep(img)
        p = next(self.model.parameters())
        x = x.to(p.dtype)
        if self.backend == "depth_anything":
            # backbone feature_maps are the layer-normed tokens at DA_LAYERS (CLS included)
            maps = self.model.backbone.forward_with_filtered_kwargs(x).feature_maps
            tokens = maps[-1] if self.layer == -1 else maps[DA_LAYERS.index(self.layer)]
        else:
            out = self.model(pixel_values=x, output_hidden_states=self.layer != -1)
            tokens = out.last_hidden_state if self.layer == -1 else out.hidden_states[self.layer]
        return F.normalize(tokens[:, 1:].float(), dim=-1)

    @torch.no_grad()
    def target_features(self, img: torch.Tensor) -> torch.Tensor:
        """No-grad twin of ``features`` for ground-truth frames (cacheable)."""
        return self.features(img)


def latent_cosine_loss(pred: torch.Tensor, target: torch.Tensor,
                       weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean ``1 - cos`` over patch tokens; inputs are already L2-normalised (B, N, C).

    ``weight`` (B, N) lets a caller down-weight patches (e.g. depth-prior confidence)."""
    d = 1.0 - (pred * target.detach()).sum(-1)
    if weight is not None:
        return (d * weight).sum() / weight.sum().clamp_min(1e-6)
    return d.mean()


def patch_error_weight(render: torch.Tensor, target: torch.Tensor, grid: Tuple[int, int],
                       floor: float = 0.1) -> torch.Tensor:
    """Per-patch weight (B, N) from the photometric error between render and target.

    ``|render - target|`` (mean over channels) is average-pooled onto the ViT patch grid, so each
    token's weight is how far its image region is from the photo. Each image's weights are scaled
    to mean 1, then ``floor`` is added so matched patches still contribute a little. Detached:
    this only chooses where the latent loss pulls, it is not itself optimised.

    ``render`` / ``target``: (B, 3, H, W) in [0, 1] at the same resolution."""
    gh, gw = grid
    err = (render.detach() - target.detach()).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
    pooled = F.adaptive_avg_pool2d(err.float(), (gh, gw)).flatten(1)  # (B, N)
    w = pooled / pooled.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return w + floor


def latent_lambda(step: int, lam: float, warmup: int, anneal_end: int) -> float:
    """Constant ``lam`` after ``warmup``, then linearly annealed to 0 at ``anneal_end``.

    ``anneal_end <= 0`` holds ``lam`` (arm B); a positive value anneals it (arm C)."""
    if step < warmup:
        return 0.0
    if anneal_end <= 0:
        return lam
    if step >= anneal_end:
        return 0.0
    return lam * (1.0 - (step - warmup) / max(1, anneal_end - warmup))
