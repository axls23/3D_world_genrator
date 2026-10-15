"""Dense depth for stream keyframes: fast monocular prior, aligned to ACE's metric points.

    keyframe RGB --Depth Anything V2 Small (fp16, ~20 ms)--> relative disparity
                 --guided filter (RGB edges)--> sharp disparity
                 --robust affine fit to ACE inlier scene coordinates--> depth in ACE units
                 --flying-pixel / range mask--> valid depth + dense seed points

Monocular networks get the *shape* of a scene right but not its scale / shift, and they are
blind to the multi-view geometry ACE solved. Instead of fine-tuning the network per scene
(too slow for a 6 GB GPU next to 3DGS training and Difix), each prediction is fitted to the up
to 1500 inlier scene coordinates ACE already produced for that keyframe: disparity is affine
in inverse depth, so ``1/z = a * d + b`` is solved by IRLS (Huber) - sub-millisecond, and it
makes every depth map metric-consistent with the ACE map and the trainer's poses. A held-out
fifth of the points measures the result (``rel_err``); frames whose fit is poor are dropped.

Everything but ``DepthEstimator.predict`` is numpy / OpenCV so it can be tested on CPU.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthEstimator:
    """Depth Anything V2 (transformers) in fp16; ``predict`` returns relative disparity at the
    input image size. Loaded lazily on first use (weights come from HF_HOME)."""

    def __init__(self, model_id: str = DEFAULT_MODEL, input_side: int = 518, device: str = "cuda"):
        self.model_id = model_id
        self.input_side = input_side  # long side fed to the network (multiple of 14)
        self.device = device
        self.model = None
        self.ms = 0.0                 # running mean inference time
        self._lock = threading.Lock()  # shared by the preview, the depth worker, the cold start

    def load(self):
        with self._lock:
            return self._load()

    def _load(self):
        if self.model is None:
            import torch
            from transformers import DepthAnythingForDepthEstimation

            from hypersplat.stream.hf_lock import hf_load

            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            with hf_load():
                # _fast_init=False: fast init swaps torch.nn.init out process-wide while it loads
                self.model = DepthAnythingForDepthEstimation.from_pretrained(self.model_id, _fast_init=False).to(
                    self.device, dtype).eval()
            logger.info(f"[Depth] {self.model_id} loaded on {self.device}")
        return self

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> relative disparity float32 (H, W), larger = closer."""
        import torch

        with self._lock:
            self._load()
            return self._predict(rgb, torch)

    def _predict(self, rgb: np.ndarray, torch) -> np.ndarray:
        h, w = rgb.shape[:2]
        s = self.input_side / max(h, w)
        ih, iw = (max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14))
        x = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        p = next(self.model.parameters())
        t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(p.device, p.dtype)
        t0 = time.time()
        with torch.no_grad():
            d = self.model(pixel_values=t).predicted_depth[:, None].float()
            d = torch.nn.functional.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        out = d[0, 0].cpu().numpy()
        ms = (time.time() - t0) * 1000
        self.ms = ms if self.ms == 0 else 0.9 * self.ms + 0.1 * ms
        return out


# ============================================================================ geometry
def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """He et al. guided filter (grey guide): smooths ``src`` inside regions and snaps its
    edges to the guide's edges - the upsampled network output gets the image's sharp
    boundaries back. guide in [0, 1], src any float range (normalised internally)."""
    g = guide.astype(np.float32)
    lo, hi = float(np.percentile(src, 1)), float(np.percentile(src, 99))
    scale = max(hi - lo, 1e-6)
    p = ((src - lo) / scale).astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)
    mg, mp = box(g), box(p)
    a = (box(g * p) - mg * mp) / (box(g * g) - mg * mg + eps)
    b = mp - a * mg
    return (box(a) * g + box(b)) * scale + lo


@dataclass
class Alignment:
    a: float
    b: float
    n: int            # points used
    inliers: float    # fraction within inlier_tol (relative inverse depth)
    rel_err: float    # median |z_pred - z| / z on held-out points
    ok: bool
    reason: str = ""
    anchored: bool = False  # the fit also used rendered-depth anchors (closed loop)


def _irls(A: np.ndarray, y: np.ndarray, base_w: np.ndarray, delta: float) -> np.ndarray:
    w = base_w.copy()
    coef = np.zeros(2)
    for _ in range(8):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = np.abs(A @ coef - y)
        w = base_w * np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return coef


def align_disparity(disp_at_pts: np.ndarray, z: np.ndarray, min_points: int = 40,
                    inlier_tol: float = 0.1, min_inliers: float = 0.5, max_rel_err: float = 0.15,
                    holdout: float = 0.2, seed: int = 0,
                    anchors: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Alignment:
    """Fit ``1/z = a * disp + b`` (IRLS, Huber) on 80 % of the ACE points; score on the rest.

    ``anchors``: (disparity, z) at confident pixels of the 3DGS render. They join the fit with
    the same total weight as the ACE points (dense and multi-view consistent, but produced by
    the model this depth supervises), and the anchored fit is kept only if it predicts the
    held-out ACE points about as well as the ACE-only fit (within 10 %), so the render cannot
    pull the target away from ACE's geometry."""
    m = np.isfinite(disp_at_pts) & np.isfinite(z) & (z > 1e-6)
    d, inv = disp_at_pts[m].astype(np.float64), 1.0 / z[m].astype(np.float64)
    zz = z[m].astype(np.float64)
    n = len(d)
    if n < min_points:
        return Alignment(0, 0, n, 0, 1, False, f"{n} ACE points < {min_points}")
    order = np.random.default_rng(seed).permutation(n)
    k = max(int(n * holdout), 1)
    test, fit = order[:k], order[k:]
    A = np.stack([d[fit], np.ones(len(fit))], 1)
    y = inv[fit]
    delta = 0.05 * np.median(y)

    def held_out_err(c):
        pi = c[0] * d[test] + c[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(1.0 / pi - zz[test]) / zz[test]
        return float(np.median(np.where(np.isfinite(rel) & (pi > 0), rel, 1.0)))

    coef = _irls(A, y, np.ones(len(fit)), delta)
    rel_err = held_out_err(coef)
    anchored = False
    if anchors is not None:
        ad, az = (np.asarray(v, np.float64) for v in anchors)
        am = np.isfinite(ad) & np.isfinite(az) & (az > 1e-6)
        if am.sum() >= min_points:
            A2 = np.concatenate([A, np.stack([ad[am], np.ones(int(am.sum()))], 1)])
            y2 = np.concatenate([y, 1.0 / az[am]])
            bw = np.concatenate([np.ones(len(fit)), np.full(int(am.sum()), len(fit) / am.sum())])
            coef2 = _irls(A2, y2, bw, delta)
            err2 = held_out_err(coef2)
            # The ACE-only fit is tuned to the noise of the same ACE points, so a correct
            # render scores marginally worse (~2 %); one whose scale is off by 2 % already
            # scores ~30 % worse
            if err2 <= 1.1 * rel_err + 1e-3:
                coef, rel_err, anchored = coef2, err2, True
    a, b = float(coef[0]), float(coef[1])
    pred_inv = a * d + b
    inl = float(np.mean(np.abs(pred_inv[fit] - y) <= inlier_tol * y))
    if a <= 0:
        return Alignment(a, b, n, inl, rel_err, False, "inverted depth order", anchored)
    if inl < min_inliers:
        return Alignment(a, b, n, inl, rel_err, False, f"{inl:.0%} inliers < {min_inliers:.0%}", anchored)
    if rel_err > max_rel_err:
        return Alignment(a, b, n, inl, rel_err, False, f"held-out error {rel_err:.0%} > {max_rel_err:.0%}",
                         anchored)
    return Alignment(a, b, n, inl, rel_err, True, "", anchored)


def flying_pixel_mask(depth: np.ndarray, max_rel_grad: float = 0.04) -> np.ndarray:
    """True where depth is trustworthy: not on a depth discontinuity (where the network
    interpolates between foreground and background and unprojection makes 'flying' points)."""
    lg = np.log(np.maximum(depth, 1e-6)).astype(np.float32)
    gx = cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    edge = (np.sqrt(gx * gx + gy * gy) > max_rel_grad).astype(np.uint8)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
    return edge == 0


def unproject(depth: np.ndarray, valid: np.ndarray, K: np.ndarray, c2w: np.ndarray,
              rgb: np.ndarray, stride: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """World points (in c2w's frame) + colours of every ``stride``-th valid pixel."""
    h, w = depth.shape
    ys, xs = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
    ys, xs = ys.ravel(), xs.ravel()
    keep = valid[ys, xs]
    ys, xs = ys[keep], xs[keep]
    z = depth[ys, xs].astype(np.float64)
    x = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], 1)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32), rgb[ys, xs].astype(np.float32) / 255.0


@dataclass
class DepthResult:
    depth: Optional[np.ndarray]     # (H, W) float32 in ACE units, 0 where invalid
    valid: Optional[np.ndarray]     # (H, W) bool
    align: Alignment
    ms: float
    disp: Optional[np.ndarray] = None   # the sharpened disparity the fit used (for re-alignment)


def dense_depth(disp: np.ndarray, rgb: np.ndarray, pix: np.ndarray, z: np.ndarray,
                guide_radius: int = 4, max_range: float = 3.0, max_rel_grad: float = 0.04,
                **align_kw) -> DepthResult:
    """Sharpen, align and mask one prediction. pix: ACE point positions in [0, 1] (N, 2);
    z: their camera-space depth in ACE units."""
    t0 = time.time()
    h, w = disp.shape
    if guide_radius > 0 and rgb is not None:
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        disp = guided_filter(grey, disp, guide_radius)
    xs = np.clip((pix[:, 0] * w).astype(int), 0, w - 1)
    ys = np.clip((pix[:, 1] * h).astype(int), 0, h - 1)
    al = align_disparity(disp[ys, xs], z, **align_kw)
    if not al.ok:
        return DepthResult(None, None, al, (time.time() - t0) * 1000, disp)
    inv = al.a * disp + al.b
    zmax = max_range * float(np.percentile(z, 95))
    zmin = float(np.percentile(z, 5)) / max_range
    with np.errstate(divide="ignore"):
        depth = np.where(inv > 1.0 / zmax, 1.0 / np.maximum(inv, 1e-9), 0.0).astype(np.float32)
    valid = (depth > zmin) & (depth < zmax)
    valid &= flying_pixel_mask(np.where(valid, depth, zmax), max_rel_grad)
    depth[~valid] = 0.0
    return DepthResult(depth, valid, al, (time.time() - t0) * 1000, disp)


def seed_depth(disp: np.ndarray, rgb: Optional[np.ndarray], median_depth: float = 2.0,
               far_ratio: float = 8.0, guide_radius: int = 4,
               max_rel_grad: float = 0.04) -> Tuple[np.ndarray, Dict]:
    """Depth for the cold-start seed frame, before any ACE points exist to align against:
    ``z = c / disp`` (relative disparity taken as shift-free inverse depth) with ``c`` putting
    the median at ``median_depth`` - which sets the map's units. Sky / far background
    (beyond ``far_ratio`` x median) and depth discontinuities are 0 (no target).
    Returns (depth float32 like ``disp``, stats)."""
    d = disp.astype(np.float32)
    if guide_radius > 0 and rgb is not None:
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        d = guided_filter(grey, d, guide_radius)
    pos = np.isfinite(d) & (d > 0)
    if pos.sum() < 100:
        return np.zeros_like(d), {"valid": 0.0, "reason": "no positive disparity"}
    with np.errstate(divide="ignore"):
        z = np.where(pos, 1.0 / np.maximum(d, 1e-9), 0.0)
    z *= median_depth / float(np.median(z[pos]))
    valid = pos & (z < far_ratio * median_depth) & (z > median_depth / (4 * far_ratio))
    valid &= flying_pixel_mask(np.where(valid, z, far_ratio * median_depth), max_rel_grad)
    z = np.where(valid, z, 0.0).astype(np.float32)
    return z, {"valid": float(valid.mean()), "median": float(np.median(z[valid])) if valid.any() else 0.0}


# ============================================================================ render feedback
def render_consistency(disp_real: np.ndarray, disp_render: np.ndarray, k: float = 3.0,
                       floor: float = 0.1) -> Tuple[np.ndarray, float]:
    """Per-pixel trust in a keyframe's monocular depth, from Depth Anything run on the camera
    frame *and* on the 3DGS render at the same pose (closed feedback loop, once per epoch).

    The same network sees both images, so where the map is right the two disparities agree up
    to an affine map (the render's exposure / contrast differs a little). The robust affine fit
    ``disp_render ~ a * disp_real + b`` removes that; the residual, in units of its own MAD,
    is large exactly where the render's geometry (floaters, holes, wrong surfaces) differs
    from what the camera saw. Returns (weight in [floor, 1], agreement score in [0, 1]) -
    1 / (1 + (r / (k sigma))^2) per pixel, score = mean weight."""
    a, b = disp_real.astype(np.float64).ravel(), disp_render.astype(np.float64).ravel()
    step = max(1, a.size // 20000)
    A = np.stack([a[::step], np.ones(len(a[::step]))], 1)
    y = b[::step]
    coef = _irls(A, y, np.ones(len(y)), 0.05 * max(float(np.std(y)), 1e-6))
    r = disp_render - (coef[0] * disp_real + coef[1])
    sigma = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-6
    w = np.clip(1.0 / (1.0 + (r / (k * sigma)) ** 2), floor, 1.0).astype(np.float32)
    return w, float(w.mean())


# ============================================================================ preview
def depth_preview_jpeg(disp: np.ndarray, max_side: int = 640, quality: int = 80) -> bytes:
    """Colour-mapped relative depth (near = bright) as JPEG, for live displays."""
    lo, hi = np.percentile(disp, (2, 98))
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class DepthPreview:
    """Depth of the newest camera frame at ~``fps`` until stopped: something to look at while
    the first ACE map is built. Runs the estimator on its own thread; stop() before anyone
    else uses the estimator."""

    def __init__(self, estimator: DepthEstimator, latest_frame, publish, fps: float = 3.0,
                 max_side: int = 480):
        import threading

        self.estimator, self.latest_frame, self.publish = estimator, latest_frame, publish
        self.period, self.max_side = 1.0 / fps, max_side
        self.frames = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="depth-preview", daemon=True)

    def start(self) -> "DepthPreview":
        self._thread.start()
        return self

    def _run(self):
        from .source import fit_max_side

        last = None
        while not self._stop.is_set():
            t0 = time.time()
            item = self.latest_frame()
            if item is not None and item[3] != last:
                last = item[3]
                try:
                    self.publish(depth_preview_jpeg(self.estimator.predict(fit_max_side(item[0], self.max_side))))
                    self.frames += 1
                except Exception as e:  # preview only: never take the run down
                    logger.warning(f"[Depth] preview failed: {e}")
                    return
            self._stop.wait(max(0.0, self.period - (time.time() - t0)))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
