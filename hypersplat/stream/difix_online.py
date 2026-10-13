"""Online Difix: repair pseudo views of the model *while it trains*.

Replaces the post-training feedback loop (render -> Difix -> write augmented COLMAP -> retrain
from scratch, up to 3x) with rounds inside one training run:

    render pseudo cameras from the live splats -> Difix (conditioned on the source photo)
    -> weight by source trust x angle falloff x change-PSNR trust -> replace the previous
    round's pseudo views in the training set

Placement, trust and the closed-loop orbit angle / weight are the batch loop's
(hypersplat/beta/difix/augment.py, params/strategies/difix.py): each round appends a
difix.rounds entry, and the next round's angle widens, holds or shrinks from it. Rounds stop
once a round says Difix is hallucinating (strategies.difix.round_hallucinating).

VRAM: Difix (UNet + VAE, 1.8 GB fp16; the text encoder is dropped after encoding the fixed
prompt) is parked on the CPU between rounds and only moved to the GPU for the seconds a round
takes. Activations grow with pixels (~1.6 GB at 960x540), so on a CUDA OOM the round steps
down SCALES (Difix on a downscaled copy, resized back) and remembers the scale that fit; if
even the smallest fails, the round is skipped instead of killing training.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from .source import encode_jpeg
from .store import PseudoView

logger = logging.getLogger(__name__)

SCALES = (1.0, 0.85, 0.75, 0.6, 0.5)


class _RoundSkipped(Exception):
    """Difix could not fit in VRAM even at the smallest scale."""


@dataclass
class DifixSources:
    """Real train cameras pseudo views are derived from (trainer's normalized frame)."""
    c2ws: np.ndarray                          # (N, 4, 4)
    K: np.ndarray                             # (3, 3) at (width, height)
    width: int
    height: int
    weights: np.ndarray                       # (N,) trust of each source (pose confidence)
    points: np.ndarray                        # (M, 3) scene points (look-at depth, spread)
    get_image: Callable[[int], np.ndarray]    # source index -> RGB uint8 (height, width, 3)
    ids: Sequence[int]                        # caller's id per source (train index / keyframe id)


class OnlineDifix:
    def __init__(self, model_id: str = "nvidia/difix_ref", device: str = "cuda", profile=None,
                 debug_dir: Optional[str] = None, seed: Optional[int] = 0, deterministic: bool = True,
                 pool_views: int = 8):
        """seed: base of a private per-view generator (None: torch's global RNG, which also
        shifts the trainer's MCMC / sampling stream by however many draws Difix made).
        deterministic: encode with the VAE posterior mean (no sampling noise at all).
        pool_views: judge hallucination / steer angle and weight on the change PSNRs of the
        last rounds holding at least this many views, not on one round alone. The stream cycle
        runs 2-view rounds: one bad pair stopped Difix for the rest of the run and the orbit
        angle random-walked on 2-view means. 0 = per round (the batch loop's rule)."""
        from hypersplat.pipeline.params import SceneProfile

        self.model_id = model_id
        self.device = device
        self.profile = profile if profile is not None else SceneProfile()
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.seed = seed
        self.deterministic = bool(deterministic)
        self.pool_views = int(pool_views)
        self._recent: List[Tuple[List[float], int, int, float]] = []  # (changes, kept, n, angle) per round
        self.fixer = None
        self.rounds = 0
        self.stop_reason: Optional[str] = None
        self.scale_idx = 0      # index into SCALES that last fit in VRAM
        self._round: Optional[dict] = None   # in-progress chunked round (begin/step/finish_round)
        self._preload: Optional[threading.Thread] = None
        self._preload_error: Optional[BaseException] = None

    def view_seed(self, loop: int, k: int) -> Optional[int]:
        """Seed of view k of round ``loop`` (None without a base seed)."""
        if self.seed is None:
            return None
        return (int(self.seed) * 1_000_003 + loop * 1_009 + k) % (2 ** 63)

    # ------------------------------------------------------------------ model residency
    def _build_fixer(self, device: str):
        from hypersplat.beta.difix.augment import DifixFixer  # resolves the shared HF cache

        logger.info(f"[Difix-online] Loading {self.model_id} (seed {self.seed}, "
                    f"{'posterior-mean' if self.deterministic else 'sampled'} latents, "
                    f"control pooled over {self.pool_views or 1} views)")
        fixer = DifixFixer(self.model_id, device)
        fixer.deterministic = self.deterministic
        return fixer

    def preload(self):
        """Load the model into pinned host memory on a background thread, so the first round
        only pays a host-to-device copy instead of a cold load from disk (~10 s)."""
        if self.fixer is not None or self._preload is not None:
            return

        def run():
            t0 = time.time()
            try:
                self.fixer = self._build_fixer("cpu")
                logger.info(f"[Difix-online] Preloaded into pinned host memory in {time.time() - t0:.1f} s")
            except BaseException as e:   # reported (and retried) by the first round
                self._preload_error = e

        self._preload = threading.Thread(target=run, name="difix-preload", daemon=True)
        self._preload.start()

    def _fixer_on_gpu(self):
        import torch

        if self._preload is not None:
            self._preload.join()
            self._preload = None
            if self._preload_error is not None:
                logger.warning(f"[Difix-online] preload failed ({self._preload_error}); loading now")
                self._preload_error = None
        torch.cuda.empty_cache()
        if self.fixer is None:
            self.fixer = self._build_fixer(self.device)
        else:
            self.fixer.to(self.device)
        return self.fixer

    def _park(self):
        import torch

        if self.fixer is not None:
            self.fixer.to("cpu")
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ one round
    def run_round(self, render_fn: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray],
                  src: DifixSources, step: int, max_views: Optional[int] = None
                  ) -> Tuple[List[PseudoView], List[int], dict]:
        """render_fn(c2w, K, width, height) -> RGB uint8 of the current model.

        Returns (pseudo views to train on, source index of each, the round's stats). One
        blocking round = begin_round + step_round until done + finish_round."""
        self.begin_round(src, step, max_views)
        try:
            while not self.step_round(render_fn, budget_s=None):
                pass
        except _RoundSkipped:
            return [], [], {}
        return self.finish_round(step)

    @property
    def round_pending(self) -> bool:
        return self._round is not None

    def begin_round(self, src: DifixSources, step: int, max_views: Optional[int] = None,
                    max_angle: Optional[float] = None):
        """Pick this round's pseudo cameras; views are rendered and fixed by step_round.
        max_angle caps the closed-loop orbit angle (many small rounds would otherwise widen it
        every round up to the strategy's cap)."""
        from hypersplat.beta.difix.augment import camera_spread_deg, make_pseudo_poses
        from hypersplat.pipeline.params import SceneProfile
        from hypersplat.pipeline.params.strategies.difix import (derive_max_angle, derive_num_views,
                                                                 derive_pseudo_weight)

        loop = self.rounds
        if self.profile.get("difix.camera_spread_deg") is None:
            spread = camera_spread_deg(src.c2ws, src.points)
            self.profile.set("difix.camera_spread_deg", round(spread, 3), "difix.online")
        angle = float(derive_max_angle(self.profile, None).value)
        if max_angle is not None:
            angle = min(angle, float(max_angle))
        n_views = int(derive_num_views(SceneProfile(data={"frames": {"count": len(src.c2ws)}}), None).value)
        if max_views:
            n_views = min(n_views, int(max_views))
        base_weight = float(derive_pseudo_weight(self.profile, None).value)
        poses = make_pseudo_poses(src.c2ws, src.weights, src.points, n_views, angle, seed=loop + 1)
        self._round = {"src": src, "poses": poses, "angle": angle, "base_weight": base_weight,
                       "renders": [], "fixed": [], "t0": time.time(), "step0": int(step),
                       "t": {"render_ms": 0.0, "difix_ms": 0.0, "load_ms": 0.0, "park_ms": 0.0},
                       "chunks": 0, "max_chunk_s": 0.0}

    def step_round(self, render_fn: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray],
                   budget_s: Optional[float] = 3.0) -> bool:
        """Render + fix pending views until ``budget_s`` is spent (None: all of them); the
        model is moved to the GPU for the chunk and parked after it. True when every view of
        the round is fixed. Raises _RoundSkipped when even the smallest scale OOMs."""
        import torch

        r = self._round
        t_chunk = time.time()
        t = time.time()
        try:
            fixer = self._fixer_on_gpu()
        except torch.cuda.OutOfMemoryError:
            logger.warning("[Difix-online] CUDA OOM moving Difix to the GPU; round skipped")
            self._park()
            self._round = None
            raise _RoundSkipped()
        r["t"]["load_ms"] += (time.time() - t) * 1000
        use_ref = "ref" in self.model_id
        src = r["src"]
        try:
            while len(r["fixed"]) < len(r["poses"]):
                s, c2w, _ = r["poses"][len(r["fixed"])]
                t = time.time()
                render = render_fn(c2w, src.K, src.width, src.height)
                r["t"]["render_ms"] += (time.time() - t) * 1000
                ref = src.get_image(s) if use_ref else None
                t = time.time()
                while True:
                    try:
                        fixed = fixer(render, ref, scale=SCALES[self.scale_idx],
                                      seed=self.view_seed(self.rounds, len(r["fixed"])))
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if self.scale_idx + 1 >= len(SCALES):
                            logger.warning(f"[Difix-online] CUDA OOM even at scale {SCALES[-1]}; round skipped")
                            self._round = None
                            raise _RoundSkipped()
                        self.scale_idx += 1
                        logger.warning(f"[Difix-online] CUDA OOM; Difix scale -> {SCALES[self.scale_idx]}")
                r["t"]["difix_ms"] += (time.time() - t) * 1000
                r["renders"].append(render)
                r["fixed"].append(fixed)
                if budget_s is not None and time.time() - t_chunk >= budget_s:
                    break
        finally:
            t = time.time()
            self._park()
            if self._round is not None:
                r["t"]["park_ms"] += (time.time() - t) * 1000
                r["chunks"] += 1
                r["max_chunk_s"] = max(r["max_chunk_s"], time.time() - t_chunk)
        return len(r["fixed"]) >= len(r["poses"])

    def finish_round(self, step: int) -> Tuple[List[PseudoView], List[int], dict]:
        """Weights + pseudo views of a fully fixed round (see run_round)."""
        from hypersplat.beta.difix.augment import ANGLE_FALLOFF, change_psnr
        from hypersplat.pipeline.params.strategies.difix import (pseudo_view_weights, round_hallucinating,
                                                                 round_stats)

        r, self._round = self._round, None
        loop, src, poses, angle = self.rounds, r["src"], r["poses"], r["angle"]
        renders, fixed = r["renders"], r["fixed"]
        angles = [a for _, _, a in poses]
        changes = [change_psnr(x, f) for x, f in zip(renders, fixed)]
        base = [r["base_weight"] * float(src.weights[s]) * max(1.0 - ANGLE_FALLOFF * a / max(angle, 1e-6), 0.0)
                for s, _, a in poses]
        weights, threshold = pseudo_view_weights(base, changes)

        # Keep each view at the resolution Difix worked at: a downscaled fix upsampled back to
        # full size is a blurry target that pulls the model toward blur (LPIPS 0.29 -> 0.46)
        scale = SCALES[self.scale_idx]
        W, H, K = src.width, src.height, np.asarray(src.K, np.float64)
        if scale < 1.0:
            w8, h8 = max(8, int(W * scale) // 8 * 8), max(8, int(H * scale) // 8 * 8)
            K = K.copy()
            K[0] *= w8 / W
            K[1] *= h8 / H
            W, H = w8, h8
        views, sources = [], []
        for k, ((s, c2w, _), img, w) in enumerate(zip(poses, fixed, weights)):
            if w <= 0.0:
                continue
            if img.shape[:2] != (H, W):
                import cv2
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            views.append(PseudoView(name=f"pseudo_r{loop}_{k:03d}", jpeg=pa.py_buffer(encode_jpeg(img, 92)),
                                    c2w=np.asarray(c2w, np.float64), K=K, width=W, height=H,
                                    weight=float(w), conf=0.0, src_kid=int(src.ids[s])))
            sources.append(int(s))
        stats = round_stats(loop, angle, angles, changes, threshold, len(views))
        if self.pool_views > 0:
            stats = self._pooled(stats, changes, len(views), len(poses), angle)
        n = max(len(poses), 1)
        stats.update({"step": int(step), "step_begin": r["step0"], "difix_scale": SCALES[self.scale_idx],
                      "wall_s": round(time.time() - r["t0"], 1), "chunks": r["chunks"],
                      "max_chunk_s": round(r["max_chunk_s"], 2),
                      "render_ms_per_view": round(r["t"]["render_ms"] / n, 1),
                      "difix_ms_per_view": round(r["t"]["difix_ms"] / n, 1),
                      "load_ms": round(r["t"]["load_ms"]), "park_ms": round(r["t"]["park_ms"])})
        self.profile.append("difix.rounds", stats, "difix.online")
        self.profile.save()
        self.rounds += 1
        reason = round_hallucinating(stats)
        if reason:
            self.stop_reason = reason
        if self.debug_dir is not None:
            self._save_debug(loop, renders, fixed)
        logger.info(f"[Difix-online] Round {loop} @ step {step}: {len(views)}/{len(poses)} views kept, "
                    f"angle {angle:.1f} deg, mean change PSNR {stats['mean_change_psnr']} dB, "
                    f"base weight {r['base_weight']:.2f}, Difix scale {SCALES[self.scale_idx]}; "
                    f"{stats['difix_ms_per_view']:.0f} ms/view Difix + {stats['render_ms_per_view']:.0f} ms render, "
                    f"{stats['chunks']} chunks (longest {stats['max_chunk_s']:.1f} s, load {stats['load_ms']} ms, "
                    f"park {stats['park_ms']} ms), {stats['wall_s']:.0f} s wall"
                    + (f"; stopping: {reason}" if reason else ""))
        return views, sources, stats

    def _pooled(self, stats: dict, changes: List[float], kept: int, n: int, angle: float) -> dict:
        """The round's stats plus pooled_* change PSNR / views / kept over the last rounds
        holding >= pool_views views; strategies.difix.judged() decides on those. The pool
        restarts when the orbit angle moves, since change PSNR depends on it. While it holds
        fewer than pool_views views it judges nothing (judge "pooling": no stop, angle and
        weight hold)."""
        import math

        if self._recent and abs(self._recent[-1][3] - angle) > 1e-6:
            self._recent.clear()
        self._recent.append((list(changes), int(kept), int(n), float(angle)))
        while len(self._recent) > 1 and sum(r[2] for r in self._recent[1:]) >= self.pool_views:
            self._recent.pop(0)
        xs = [float(c) for r in self._recent for c in r[0] if c is not None and math.isfinite(float(c))]
        n_pool = sum(r[2] for r in self._recent)
        out = dict(stats)
        out.update(judge="pooled" if n_pool >= self.pool_views else "pooling", pooled_rounds=len(self._recent),
                   pooled_mean_change_psnr=round(sum(xs) / len(xs), 3) if xs else None,
                   pooled_n_views=n_pool, pooled_n_kept=sum(r[1] for r in self._recent))
        return out

    def _save_debug(self, loop: int, renders, fixed, n: int = 4):
        """Small before|after strip of the first views (the only thing a round writes)."""
        from PIL import Image
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        rows = [np.concatenate([r, f], axis=1) for r, f in list(zip(renders, fixed))[:n]]
        if rows:
            strip = Image.fromarray(np.concatenate(rows, axis=0))
            strip.thumbnail((1280, 1280))
            strip.save(self.debug_dir / f"round_{loop:02d}.jpg", quality=85)
