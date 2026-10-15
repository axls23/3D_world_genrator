"""V-JEPA 2 predictive latent loss with secondary (depth / DINO) tokens hooked into the predictor.

    render --> frozen V-JEPA encoder --> context tokens (random subset, differentiable)
                                                 |
                              predictor  <-- SecondaryHook (forward hook on predictor_embed):
                                  |              + proj(secondary tokens of the photo)
                                  v
                          predicted latents at held-out positions
                                  |  1 - cos, optional per-patch error weight
    photo  --> frozen V-JEPA encoder (no grad) --> target latents at the same held-out positions

Secondary tokens come from the photo (Depth Anything V2 or DINOv2 via ``LatentEncoder``), plus one
channel of the render's alpha (3DGUT coverage) pooled to the patch grid, so they are conditioning,
not a second copy of the target. ``proj`` is zero-initialised: at step 0 the hook
adds nothing and the pretrained predictor is unchanged. Only ``proj`` is trained; encoder and
predictor stay frozen, and gradients reach the render through them.

Alpha hole term: a patch the predictor gets badly wrong (high 1 - cos against the photo) while the
render barely covers it (low alpha) is a hole the photo says should be filled. ``alpha_hole`` is
mean(relative latent error * (1 - alpha)) over such patches, differentiable in alpha only, so it
raises opacity where geometry is missing. Patches outside ``valid`` (e.g. sky, no depth) are skipped.
"""

import logging
import math
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypersplat.stream.hf_lock import hf_load
from hypersplat.stream.latent import latent_cosine_loss, patch_error_weight
from hypersplat.stream.vjepa.masks import apply_masks

logger = logging.getLogger(__name__)

VJEPA_PATCH = 16
VJEPA_TUBELET = 2
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

class LatentPacer:
    """Paces the V-JEPA term against the trainer's own step rate.

    ``budget`` is the fraction of a trainer step the V-JEPA call may add on average. The call runs
    every ``interval`` steps, with ``interval = ceil(call_cost / (budget * step_time))``, so the
    V-JEPA share of wall time stays near ``budget``. On run steps the term is multiplied by
    ``interval``, so its expected contribution over time matches running it every step.
    ``budget <= 0`` runs it every step. Times are EMAs; the step time excludes V-JEPA time.
    """

    def __init__(self, budget: float, ema: float = 0.9):
        self.budget = budget
        self.ema = ema
        self.base_s: Optional[float] = None  # trainer step time, V-JEPA time removed
        self.cost_s: Optional[float] = None  # one V-JEPA call
        self._last_wall: Optional[float] = None
        self._last_cost = 0.0

    def _mix(self, old, new):
        return new if old is None else self.ema * old + (1.0 - self.ema) * new

    def tick(self, now: float) -> None:
        """Call once at the top of each training step with ``time.time()``."""
        if self._last_wall is not None:
            self.base_s = self._mix(self.base_s, max((now - self._last_wall) - self._last_cost, 1e-6))
        self._last_wall = now
        self._last_cost = 0.0

    def interval(self) -> int:
        if self.budget <= 0 or self.base_s is None or self.cost_s is None:
            return 1
        return max(1, math.ceil(self.cost_s / (self.budget * self.base_s)))

    def should_run(self, step: int) -> bool:
        return step % self.interval() == 0

    def record(self, cost_s: float) -> None:
        """Call with the wall time of a V-JEPA call that just ran."""
        self._last_cost = cost_s
        self.cost_s = self._mix(self.cost_s, cost_s)


class SecondaryHook:
    """Adds ``proj(secondary)`` to the predictor's context tokens via a forward hook.

    The hook sits on ``predictor.predictor_embed``, the Linear that maps encoder context tokens into
    predictor width, so the secondary signal enters before the predictor blocks. Call ``set`` with
    the full-grid secondary tokens and the context indices before each predictor call, and ``clear``
    after it. With nothing set, the hook returns its input unchanged.
    """

    def __init__(self, predictor: nn.Module, sec_dim: int):
        out_dim = predictor.predictor_embed.out_features
        self.proj = nn.Linear(sec_dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self._sec: Optional[torch.Tensor] = None
        self._masks_x = None
        self._handle = predictor.predictor_embed.register_forward_hook(self._hook)

    def set(self, sec_tokens: torch.Tensor, masks_x) -> None:
        self._sec = sec_tokens
        self._masks_x = masks_x if isinstance(masks_x, list) else [masks_x]

    def clear(self) -> None:
        self._sec = None
        self._masks_x = None

    def _hook(self, module, inputs, output):
        if self._sec is None:
            return output
        # Gather the secondary tokens with the same context indices the predictor used, in the same
        # order: apply_masks stacks one block per mask, matching the predictor's (len(masks_x) * B) batch.
        s = self.proj(self._sec.float())
        s = apply_masks(s, self._masks_x).to(output.dtype)
        return output + s

    def remove(self) -> None:
        self._handle.remove()


class VJEPAOut(NamedTuple):
    latent: torch.Tensor  # 1 - cos on held-out patches (scalar)
    alpha_hole: Optional[torch.Tensor]  # hole term (scalar, grad into alpha), None without alpha
    hole_frac: Optional[float]  # share of held-out patches counted as holes, for logging


class VJEPALatentLoss(nn.Module):
    """Predictive latent loss; see module docstring.

    ``encoder`` / ``predictor`` are V-JEPA 2 modules (vendored in ``hypersplat.stream.vjepa``).
    ``secondary`` is a ``LatentEncoder`` (``features``, ``grid``); ``sec_dim`` is its token width Cs.
    """

    def __init__(self, encoder: nn.Module, predictor: nn.Module, secondary, sec_dim: int, side: int = 256,
                 target_frac: float = 0.5, mask_index: int = 0):
        super().__init__()
        if not 0.0 < target_frac < 1.0:
            raise ValueError(f"target_frac must be in (0, 1) to leave context and target patches, got {target_frac}")
        self.encoder = encoder.eval().requires_grad_(False)
        self.predictor = predictor.eval().requires_grad_(False)
        self.secondary = secondary
        self.side = side
        self.target_frac = target_frac
        self.mask_index = mask_index
        self.sec_dim = sec_dim
        self.hook = SecondaryHook(self.predictor, sec_dim + 1)  # + 1: render alpha channel
        # register the projection as a child module so .to(device) and state_dict see it
        self.sec_proj = self.hook.proj

    def trainable_parameters(self):
        return list(self.hook.proj.parameters())

    def load_hook_state(self, state: dict) -> None:
        """Load a saved projection; one saved before the alpha channel existed gets a zero alpha column."""
        w = state["weight"]
        if w.shape[1] == self.sec_dim:
            state = dict(state, weight=torch.cat([w, w.new_zeros(w.shape[0], 1)], dim=1))
        self.sec_proj.load_state_dict(state)

    def _grid(self, h: int, w: int):
        """V-JEPA patch grid for an (h, w) photo: long side ``side``, multiples of the patch size."""
        s = self.side / max(h, w)
        gh = max(1, int(round(h * s / VJEPA_PATCH)))
        gw = max(1, int(round(w * s / VJEPA_PATCH)))
        return gh, gw

    def _clip(self, img: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> (B, 3, T=2, gh*16, gw*16) ImageNet-normalised clip (frame duplicated)."""
        x = F.interpolate(img, size=(gh * VJEPA_PATCH, gw * VJEPA_PATCH), mode="bilinear",
                          align_corners=False, antialias=True)
        mean = torch.tensor(_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.unsqueeze(2).repeat(1, 1, VJEPA_TUBELET, 1, 1)

    @torch.no_grad()
    def _secondary_grid(self, photo: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """Secondary tokens of the photo, resampled from their own grid to the V-JEPA grid (B, N, Cs)."""
        tok = self.secondary.features(photo)  # (B, Ns, Cs), L2-normalised per token
        sh, sw = self.secondary.grid(photo.shape[2], photo.shape[3])
        m = tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], sh, sw)
        m = F.interpolate(m, size=(gh, gw), mode="bilinear", align_corners=False)
        return m.flatten(2).transpose(1, 2).contiguous()

    def forward(self, render: torch.Tensor, photo: torch.Tensor, patch_weight: bool = False,
                weight_floor: float = 0.1, alpha: Optional[torch.Tensor] = None, alpha_condition: bool = True,
                alpha_hole_max: float = 0.5, valid: Optional[torch.Tensor] = None) -> VJEPAOut:
        """``render`` / ``photo``: (B, 3, H, W) in [0, 1]. ``alpha``: (B, 1, H, W) rendered alpha (3DGUT).
        ``valid``: (B, h, w) bool/float, any resolution; patches mostly outside it get no hole term."""
        B, _, H, W = render.shape
        gh, gw = self._grid(H, W)
        N = gh * gw
        n_tgt = min(N - 1, max(1, int(round(N * self.target_frac))))
        n_ctx = N - n_tgt

        # random per-image split of the N patches into context and target
        order = torch.rand(B, N, device=render.device).argsort(dim=1)
        ctx, tgt = order[:, :n_ctx], order[:, n_ctx:]

        a_full = None
        if alpha is not None:
            a_full = F.adaptive_avg_pool2d(alpha.float(), (gh, gw)).flatten(1)  # (B, N), differentiable

        dtype = next(self.encoder.parameters()).dtype
        full_r = self.encoder(self._clip(render, gh, gw).to(dtype))  # (B, N, D), differentiable
        with torch.no_grad():
            full_t = self.encoder(self._clip(photo, gh, gw).to(dtype))  # (B, N, D)
            sec = self._secondary_grid(photo, gh, gw)  # (B, N, Cs)
            a_cond = a_full.detach() if (a_full is not None and alpha_condition) else sec.new_zeros(B, N)
            sec = torch.cat([sec, a_cond.unsqueeze(-1).to(sec.dtype)], dim=-1)  # (B, N, Cs + 1)

        x_ctx = apply_masks(full_r, [ctx])  # (B, n_ctx, D); predictor expects the context gathered
        self.hook.set(sec, [ctx])
        try:
            pred = self.predictor(x_ctx, [ctx], [tgt], mask_index=self.mask_index)  # (B, n_tgt, D)
        finally:
            self.hook.clear()
        target = apply_masks(full_t, [tgt])
        pred_n = F.normalize(pred.float(), dim=-1)
        target_n = F.normalize(target.float(), dim=-1)

        w = None
        if patch_weight:
            w_full = patch_error_weight(render, photo, (gh, gw), floor=weight_floor)  # (B, N)
            w = torch.gather(w_full, 1, tgt)  # (B, n_tgt)
        latent = latent_cosine_loss(pred_n, target_n, w)

        alpha_hole, hole_frac = None, None
        if a_full is not None:
            d = (1.0 - (pred_n * target_n).sum(-1)).detach()  # (B, n_tgt) per-patch latent error
            rel = d / d.mean(dim=1, keepdim=True).clamp_min(1e-6)
            a_t = torch.gather(a_full, 1, tgt)  # (B, n_tgt), differentiable
            hole = (a_t.detach() < alpha_hole_max).float()
            if valid is not None:
                v = F.adaptive_avg_pool2d(valid.float().reshape(B, 1, *valid.shape[-2:]), (gh, gw)).flatten(1)
                hole = hole * (torch.gather(v, 1, tgt) > 0.5).float()
            alpha_hole = (rel * hole * (1.0 - a_t)).mean()
            hole_frac = hole.mean().item()
        return VJEPAOut(latent, alpha_hole, hole_frac)


def load_vjepa2_vitl(checkpoint: str, side: int = 256, device: str = "cuda", half: bool = True):
    """Build the V-JEPA 2 ViT-L encoder + predictor (as ``src/hub/backbones.py`` does) and load a checkpoint.

    Uses the ``target_encoder`` weights, the ones V-JEPA 2 reports for frozen features. RoPE everywhere,
    so the 2-frame clip used here needs no positional-embedding resize."""
    from hypersplat.stream.vjepa import predictor as P, vision_transformer as V

    # Build under the model-loading lock: a from_pretrained on another stream thread would otherwise
    # patch nn.Module registration mid-build and leave parameters on meta (see hf_lock).
    with torch.device("cpu"), hf_load():
        enc, pred = _build_vitl(P, V, side)
    for name, mod in (("encoder", enc), ("predictor", pred)):
        meta = [n for n, t in list(mod.named_parameters()) + list(mod.named_buffers()) if t.device.type == "meta"]
        if meta:
            raise RuntimeError(f"V-JEPA {name} built with meta tensors: {meta[:5]} (+{max(0, len(meta) - 5)} more)")

    sd = torch.load(checkpoint, map_location="cpu")

    def clean(d):
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in d.items()}

    missing_e, _ = enc.load_state_dict(clean(sd["target_encoder"]), strict=False)
    missing_p, _ = pred.load_state_dict(clean(sd["predictor"]), strict=False)
    logger.info(f"[VJEPA] loaded {checkpoint} (encoder missing {len(missing_e)}, predictor missing {len(missing_p)})")
    del sd
    dtype = torch.float16 if half and device.startswith("cuda") else torch.float32
    return enc.to(device, dtype), pred.to(device, dtype)


def _build_vitl(P, V, side):
    enc = V.vit_large(patch_size=VJEPA_PATCH, img_size=(side, side), num_frames=VJEPA_TUBELET,
                      tubelet_size=VJEPA_TUBELET, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
                      uniform_power=False, use_rope=True)
    pred = P.vit_predictor(img_size=(side, side), patch_size=VJEPA_PATCH, use_mask_tokens=True,
                           embed_dim=enc.embed_dim, predictor_embed_dim=384, num_frames=VJEPA_TUBELET,
                           tubelet_size=VJEPA_TUBELET, depth=12, num_heads=12, num_mask_tokens=10,
                           use_rope=True, uniform_power=False, use_sdpa=True, use_silu=False,
                           wide_silu=True)
    return enc, pred
