"""Difix3D+-style pseudo-view augmentation, weighted by ACE-Zero pose confidence.

Replaces the GeNVS-Lite back-view injection (which ran an untrained diffusion model and
produced pure noise). Instead of hallucinating views from scratch, we:

  1. Read ACE-Zero's per-frame registration confidence (RANSAC inlier count, last column
     of poses_final.txt) and turn it into a per-image loss weight.
  2. Place pseudo cameras near the *most confident* real cameras, orbiting their look-at
     point with progressively larger angles (as in Difix3D+'s progressive updates).
  3. Render those poses from the current 3DGS checkpoint and clean the renders with the
     pretrained Difix model (nvidia/difix, single-step SD-Turbo fixer).
  4. Write an augmented COLMAP dataset (original points3D kept) plus image_weights.json,
     which examples/datasets/colmap.py reads to weight each image's RGB loss.

Pseudo-view weight = pseudo_weight * source_confidence_weight * angle falloff * change trust,
so views derived from shaky poses, far from real coverage, or heavily rewritten by Difix get
less trust. "Change PSNR" = PSNR(raw render, Difix output) per view: a low value means Difix
changed a lot (likely hallucinating), so those views are dropped and the round's statistics are
appended to the scene profile (difix.rounds) for the closed-loop strategies in
hypersplat/pipeline/params/strategies/difix.py.

Difix weights are released under the NVIDIA non-commercial license (see LICENSE_DIFIX.txt).

Usage:
    python -m hypersplat.beta.difix.augment \
        --data_dir demo/output/drone_demo_run/acezero_output \
        --ckpt demo/output/.../ckpts/ckpt_6999_rank0.pt \
        --out_dir demo/output/drone_demo_run_difix/augmented_colmap
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "examples"))

from datasets.colmap import Parser  # noqa: E402
from datasets.normalize import transform_cameras  # noqa: E402
from gsplat import rasterization  # noqa: E402

from hypersplat.pipeline.params import SceneProfile  # noqa: E402
from hypersplat.pipeline.params.strategies.difix import (  # noqa: E402
    confidence_ramp,
    pseudo_view_weights,
    round_stats,
)
from hypersplat.beta.genvs.run_completion import (  # noqa: E402
    BaseImage,
    read_images_binary,
    rotmat2qvec,
    write_images_binary,
)

logger = logging.getLogger(__name__)

DIFIX_PROMPT = "remove degradation"

# Pseudo-camera placement, tied to the round's max_angle:
#   ANGLE_STEP_DEG  views are spread over ceil(max_angle / step) progressive levels (clamped to
#                   [1, MAX_LEVELS]), i.e. roughly one level per 3 deg - the spacing at which
#                   difix_ref's faithfulness was validated (3/6/12 deg). A 6 deg round uses
#                   {3, 6}; a 20 deg round uses {5, 10, 15, 20}.
#   PITCH_JITTER    pitch is drawn in +-PITCH_JITTER * yaw, so the orbit stays mostly along the
#                   camera path's horizontal arc and the total angle stays <= ~1.04 * max_angle.
#   ANGLE_FALLOFF   within a round the farthest view gets (1 - ANGLE_FALLOFF) of the nearest
#                   view's weight; actual hallucination is handled by the change-PSNR trust.
ANGLE_STEP_DEG = 3.0
MAX_LEVELS = 4
PITCH_JITTER = 0.3
ANGLE_FALLOFF = 0.5


# ----------------------------------------------------------------------------
# ACE-Zero pose confidence
# ----------------------------------------------------------------------------

def find_pose_file(data_dir: Path) -> Optional[Path]:
    for cand in (data_dir / "poses_final.txt", data_dir / "acezero_output" / "poses_final.txt"):
        if cand.exists():
            return cand
    found = sorted(data_dir.rglob("poses_final.txt"))
    return found[0] if found else None


def load_ace_confidence(data_dir: Path) -> Dict[str, float]:
    """Map image basename -> ACE-Zero registration confidence (inlier count)."""
    pose_file = find_pose_file(data_dir)
    if pose_file is None:
        logger.warning("No ACE-Zero poses_final.txt found; all images get confidence weight 1.0")
        return {}
    conf = {}
    for line in pose_file.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 10:
            conf[Path(parts[0]).name] = float(parts[-1])
    logger.info(f"Loaded ACE-Zero confidence for {len(conf)} frames from {pose_file}")
    return conf


def confidence_weights(conf: Dict[str, float], min_conf: Optional[float] = None,
                       floor: Optional[float] = None) -> Dict[str, float]:
    """Linear ramp: min_conf -> floor, median confidence and above -> 1.0.
    min_conf / floor default to values derived from the confidence distribution."""
    if not conf:
        return {}
    min_conf, floor = confidence_ramp(conf.values(), min_conf, floor)
    median = float(np.median(list(conf.values())))
    span = max(median - min_conf, 1e-6)
    return {name: float(np.clip((c - min_conf) / span, floor, 1.0)) for name, c in conf.items()}


def pose_correction_weights(ckpt: dict, train_names: List[str], cam_spacing: float,
                            rel_scale: float) -> Dict[str, float]:
    """Second uncertainty signal: how far joint pose optimization (--pose_opt) moved each
    ACE-Zero pose. A correction of rel_scale * camera spacing maps to weight exp(-1)."""
    sd = ckpt.get("pose_adjust")
    if not sd or "embeds.weight" not in sd:
        return {}
    from utils import rotation_6d_to_matrix

    emb = sd["embeds.weight"].float().cpu()
    if emb.shape[0] != len(train_names):
        logger.warning(f"pose_adjust has {emb.shape[0]} entries but {len(train_names)} train images; "
                       "skipping pose-correction weights (pass --ckpt_data_dir)")
        return {}
    dx = emb[:, :3].norm(dim=-1).numpy()
    R = rotation_6d_to_matrix(emb[:, 3:] + sd["identity"].float().cpu())
    ang = torch.arccos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).numpy()
    # Rotation error converted to displacement at one camera-spacing distance
    err = np.hypot(dx, ang * cam_spacing) / max(rel_scale * cam_spacing, 1e-9)
    w = np.exp(-err ** 2)
    logger.info(f"Pose-correction weights: min {w.min():.3f}, median {np.median(w):.3f} "
                f"(max shift {dx.max():.4f}, max rot {np.degrees(ang.max()):.3f} deg)")
    return {n: float(v) for n, v in zip(train_names, w)}


# ----------------------------------------------------------------------------
# Pseudo camera placement
# ----------------------------------------------------------------------------

def look_at_depth(c2w: np.ndarray, points: np.ndarray) -> float:
    """Median depth of scene points in front of the camera (OpenCV convention)."""
    w2c = np.linalg.inv(c2w)
    cam_pts = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam_pts[:, 2]
    z = z[z > 1e-3]
    return float(np.median(z)) if len(z) else 1.0


def orbit_pose(c2w: np.ndarray, pivot: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotate a camera about a pivot, around its own up (yaw) and right (pitch) axes."""
    def axis_angle(axis, deg):
        axis = axis / np.linalg.norm(axis)
        a = np.radians(deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K

    R = axis_angle(c2w[:3, 1], yaw_deg) @ axis_angle(c2w[:3, 0], pitch_deg)
    out = np.eye(4)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = R @ (c2w[:3, 3] - pivot) + pivot
    return out


def camera_spread_deg(camtoworlds: np.ndarray, points: np.ndarray) -> float:
    """Angular extent (deg) of the camera positions as seen from the scene pivot (median look-at
    point): 2 x the 90th-percentile angle to the mean viewing direction. Scale invariant, so it
    works in Parser's normalized frame."""
    if len(camtoworlds) < 2:
        return 0.0
    if len(points) > 20000:
        points = points[np.linspace(0, len(points) - 1, 20000).astype(int)]
    centers, fwd = camtoworlds[:, :3, 3], camtoworlds[:, :3, 2]
    depths = np.array([look_at_depth(c2w, points) for c2w in camtoworlds])
    pivot = np.median(centers + fwd * depths[:, None], axis=0)
    d = centers - pivot
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    mean = d.mean(0)
    mean /= max(np.linalg.norm(mean), 1e-9)
    ang = np.degrees(np.arccos(np.clip(d @ mean, -1.0, 1.0)))
    return float(2.0 * np.percentile(ang, 90))


def train_indices(n: int, test_every: int) -> np.ndarray:
    """Train-split indices (must match the trainer's held-out split)."""
    return np.array([i for i in range(n) if i % test_every != 0])


def measure_camera_spread(data_dir: Path, test_every: int = 8) -> float:
    """camera_spread_deg of the train cameras of a COLMAP dataset (CPU only)."""
    parser = Parser(str(data_dir), factor=1, normalize=True, test_every=test_every)
    return camera_spread_deg(parser.camtoworlds[train_indices(len(parser.image_names), test_every)],
                             parser.points)


def change_psnr(render: np.ndarray, fixed: np.ndarray) -> float:
    """PSNR (dB) between a raw render and its Difix output: how much Difix changed it."""
    mse = np.mean((render.astype(np.float64) - fixed.astype(np.float64)) ** 2)
    return float("inf") if mse <= 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def make_pseudo_poses(
    camtoworlds: np.ndarray,
    src_weights: np.ndarray,
    points: np.ndarray,
    num_views: int,
    max_angle: float,
    levels: Optional[int] = None,
    seed: int = 0,
) -> List[Tuple[int, np.ndarray, float]]:
    """Return (source_index, c2w, angle_deg) for pseudo views around confident cameras.
    levels defaults to ~one per ANGLE_STEP_DEG of max_angle (see module constants).
    seed > 0 also rotates which confident cameras seed the views (fresh coverage per round)."""
    if levels is None:
        levels = int(np.clip(np.ceil(max_angle / ANGLE_STEP_DEG), 1, MAX_LEVELS))
    order = np.argsort(-src_weights, kind="stable")
    rng = np.random.default_rng(seed)
    if seed:
        order = np.roll(order, seed * num_views)
    poses = []
    for k in range(num_views):
        src = int(order[k % len(order)])
        # Progressive: early views stay close to real coverage, later ones go further out
        level = 1 + (k * levels) // max(num_views, 1)
        angle = max_angle * level / levels
        yaw = angle * (1 if k % 2 == 0 else -1)
        pitch = rng.uniform(-PITCH_JITTER, PITCH_JITTER) * angle
        c2w = camtoworlds[src]
        pivot = c2w[:3, 3] + c2w[:3, 2] * look_at_depth(c2w, points)
        poses.append((src, orbit_pose(c2w, pivot, yaw, pitch), float(np.hypot(yaw, pitch))))
    return poses


# ----------------------------------------------------------------------------
# Rendering + Difix
# ----------------------------------------------------------------------------

def load_splats(ckpt_path: str, device: str):
    s = torch.load(ckpt_path, map_location=device, weights_only=False)["splats"]
    return {
        "means": s["means"],
        "quats": torch.nn.functional.normalize(s["quats"], dim=-1),
        "scales": torch.exp(s["scales"]),
        "opacities": torch.sigmoid(s["opacities"]),
        "colors": torch.cat([s["sh0"], s["shN"]], dim=1),
        "sh_degree": int(np.sqrt(s["sh0"].shape[1] + s["shN"].shape[1]) - 1),
    }


@torch.no_grad()
def render_view(splats, c2w: np.ndarray, K: np.ndarray, W: int, H: int, device: str) -> np.ndarray:
    viewmat = torch.linalg.inv(torch.from_numpy(c2w).float().to(device))[None]
    Ks = torch.from_numpy(K).float().to(device)[None]
    img, _, _ = rasterization(
        splats["means"], splats["quats"], splats["scales"], splats["opacities"], splats["colors"],
        viewmat, Ks, W, H, sh_degree=splats["sh_degree"],
    )
    return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


class DifixFixer:
    """Single-step Difix artifact remover, fp16 to fit ~3 GB of VRAM."""

    def __init__(self, model_id: str = "nvidia/difix", device: str = "cuda"):
        from hypersplat.beta.difix.pipeline_difix import DifixPipeline

        self.device = device
        pipe = DifixPipeline.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=True)
        # Keep every component on one device: the pipeline infers its execution device
        # from its modules, so offloading the text encoder sends inputs to the CPU
        pipe.to(device)
        # Prompt is fixed: encode it once
        with torch.no_grad():
            tokens = pipe.tokenizer(DIFIX_PROMPT, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt").input_ids.to(device)
            self.prompt_embeds = pipe.text_encoder(tokens)[0]
        self.pipe = pipe

    @torch.no_grad()
    def __call__(self, image: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
        """ref: a real photo near the pseudo pose (difix_ref only); anchors the fix to the
        actual scene instead of letting the model hallucinate structure from artifacts."""
        H, W = image.shape[:2]
        # SD latents need multiples of 8
        H8, W8 = (H // 8) * 8, (W // 8) * 8
        pil = Image.fromarray(image).resize((W8, H8), Image.BICUBIC)
        ref_pil = Image.fromarray(ref).resize((W8, H8), Image.BICUBIC) if ref is not None else None
        out = self.pipe(
            prompt_embeds=self.prompt_embeds, image=pil, ref_image=ref_pil, num_inference_steps=1,
            timesteps=[199], guidance_scale=0.0, height=H8, width=W8,
        ).images[0]
        return np.asarray(out.resize((W, H), Image.BICUBIC))


# ----------------------------------------------------------------------------
# Augmented COLMAP dataset
# ----------------------------------------------------------------------------

def normalized_to_raw(c2w_norm: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Undo Parser's similarity normalization (rotation re-orthonormalized)."""
    c2w = np.linalg.inv(transform) @ c2w_norm
    c2w[:3, :3] /= np.linalg.norm(c2w[:3, 0])
    return c2w


def write_augmented_colmap(src_dir: Path, out_dir: Path, pseudo: List[Tuple[str, np.ndarray]]):
    """Copy the source dataset (images symlinked, points3D kept) and append pseudo cameras."""
    sparse_in, sparse_out = src_dir / "sparse" / "0", out_dir / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for f in ("cameras.bin", "points3D.bin"):
        shutil.copy2(sparse_in / f, sparse_out / f)
    for img in sorted((src_dir / "images").iterdir()):
        dst = out_dir / "images" / img.name
        if not dst.exists():
            os.symlink(img.resolve(), dst)

    images = read_images_binary(str(sparse_in / "images.bin"))
    cam_id = next(iter(images.values())).camera_id
    next_id = max(images) + 1
    for i, (name, c2w_raw) in enumerate(pseudo):
        w2c = np.linalg.inv(c2w_raw)
        images[next_id + i] = BaseImage(
            id=next_id + i, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=cam_id,
            name=name, xys=np.zeros((0, 2)), point3D_ids=np.zeros(0, dtype=np.int64),
        )
    write_images_binary(images, str(sparse_out / "images.bin"))


def write_downscaled(out_dir: Path, factors=(2, 4)):
    """The trainer's Parser requires images_<factor>/ and maps it to images/ by sorted name,
    so every image (real and pseudo) gets a same-stem PNG here."""
    for f in factors:
        d = out_dir / f"images_{f}"
        d.mkdir(exist_ok=True)
        for img_path in sorted((out_dir / "images").iterdir()):
            dst = d / (img_path.stem + ".png")
            # Real frames never change; pseudo views are re-rendered every round
            if dst.exists() and not img_path.name.startswith("pseudo_"):
                continue
            img = Image.open(img_path).convert("RGB")
            img.resize((round(img.width / f), round(img.height / f)), Image.BICUBIC).save(dst)


def run(args) -> Path:
    device = "cuda"
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)

    parser = Parser(str(data_dir), factor=1, normalize=True, test_every=args.test_every)
    names = [Path(n).name for n in parser.image_names]
    K = parser.Ks_dict[parser.camera_ids[0]]
    W, H = parser.imsize_dict[parser.camera_ids[0]]

    # 1. Pose confidence -> per-image weights
    conf = load_ace_confidence(data_dir)
    real_w = confidence_weights(conf, getattr(args, "min_conf", None), getattr(args, "weight_floor", None))
    weights = {n: real_w.get(n, 1.0) for n in names}

    # Only train-split cameras may seed pseudo views (don't leak eval frames)
    train_idx = train_indices(len(names), args.test_every)

    # 1b. Pose uncertainty from joint pose optimization of the checkpoint's own dataset
    ckpt_dir = Path(args.ckpt_data_dir or args.data_dir)
    ckpt_names = sorted(p.name for p in (ckpt_dir / "images").iterdir())
    ckpt_train = [n for i, n in enumerate(ckpt_names)
                  if i % args.test_every != 0 or n.startswith("pseudo_")]
    spacing = float(np.median(np.linalg.norm(np.diff(parser.camtoworlds[:, :3, 3], axis=0), axis=1)))
    pose_w = pose_correction_weights(torch.load(args.ckpt, map_location="cpu", weights_only=False),
                                     ckpt_train, spacing, args.pose_rel_scale)
    for n in names:
        weights[n] *= pose_w.get(n, 1.0)
    src_w = np.array([weights[names[i]] for i in train_idx])
    poses = make_pseudo_poses(parser.camtoworlds[train_idx], src_w, parser.points,
                              args.num_views, args.max_angle, seed=getattr(args, "seed", 0))

    # 2. Render pseudo views from the current reconstruction
    splats = load_splats(args.ckpt, device)
    # The trainer normalizes the world from *its* dataset's cameras. A checkpoint trained on
    # an earlier augmented dataset lives in a different frame, so map poses through raw COLMAP.
    to_ckpt = lambda c2w: c2w
    if ckpt_dir.resolve() != data_dir.resolve():
        T_ckpt = Parser(str(ckpt_dir), factor=1, normalize=True, test_every=args.test_every).transform
        to_ckpt = lambda c2w: transform_cameras(T_ckpt, normalized_to_raw(c2w, parser.transform)[None])[0]
    renders = [render_view(splats, to_ckpt(c2w), K, W, H, device) for _, c2w, _ in poses]
    del splats
    torch.cuda.empty_cache()

    # 3. Clean with Difix
    if args.no_difix:
        fixed = renders
    else:
        fixer = DifixFixer(args.model_id, device)
        if "ref" in args.model_id:
            refs = [np.asarray(Image.open(parser.image_paths[train_idx[src]]).convert("RGB")) for src, _, _ in poses]
            fixed = [fixer(r, ref) for r, ref in zip(renders, refs)]
        else:
            fixed = [fixer(r) for r in renders]
        del fixer
        torch.cuda.empty_cache()

    # 4. Trust each view by how much Difix changed it; drop likely hallucinations
    angles = [angle for _, _, angle in poses]
    changes = [None if args.no_difix else change_psnr(raw, img) for raw, img in zip(renders, fixed)]
    base = [args.pseudo_weight * float(src_w[src]) *
            max(1.0 - ANGLE_FALLOFF * angle / max(args.max_angle, 1e-6), 0.0)
            for src, _, angle in poses]
    pseudo_w, threshold = pseudo_view_weights(base, changes)

    # 5. Write augmented dataset + weights
    pseudo_entries, pseudo_imgs, views = [], [], []
    debug_dir = out_dir / "pseudo_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for k, ((src, c2w, angle), raw, img, change, w) in enumerate(zip(poses, renders, fixed, changes, pseudo_w)):
        name = f"pseudo_{k:04d}.png"
        views.append({"name": name, "angle": round(angle, 3), "weight": w,
                      "change_psnr": round(change, 3) if change is not None and np.isfinite(change) else None})
        Image.fromarray(np.concatenate([raw, img], axis=1)).save(debug_dir / name)
        if w <= 0.0:
            continue
        pseudo_entries.append((name, normalized_to_raw(c2w, parser.transform)))
        pseudo_imgs.append(img)
        weights[name] = w
    stats = round_stats(getattr(args, "loop", None), args.max_angle, angles, changes, threshold,
                        len(pseudo_entries))
    logger.info(f"Difix change PSNR: mean {stats['mean_change_psnr']}, min {stats['min_change_psnr']} dB; "
                f"kept {stats['n_kept']}/{stats['n_views']} views (threshold {threshold:.1f} dB)")

    # Dropped views leave gaps in the numbering; stale pseudo images from an earlier run of this
    # out_dir would break the trainer's sorted images_<f>/ <-> images/ mapping
    for stale in out_dir.glob("images*/pseudo_*"):
        stale.unlink()
    write_augmented_colmap(data_dir, out_dir, pseudo_entries)
    for (name, _), img in zip(pseudo_entries, pseudo_imgs):
        Image.fromarray(img).save(out_dir / "images" / name)
    write_downscaled(out_dir)
    (out_dir / "image_weights.json").write_text(json.dumps(weights, indent=2))
    spread = camera_spread_deg(parser.camtoworlds[train_idx], parser.points)
    meta = {"ckpt": str(args.ckpt), "num_views": args.num_views, "max_angle": args.max_angle,
            "pseudo_weight": args.pseudo_weight, "difix": not args.no_difix,
            "camera_spread_deg": spread, "round": stats, "views": views,
            "ace_confidence": conf}
    (out_dir / "difix_augment.json").write_text(json.dumps(meta, indent=2, default=str))

    profile = SceneProfile.load(args.profile) if getattr(args, "profile", None) else SceneProfile.from_env()
    if profile is not None:
        profile.append("difix.rounds", stats, "difix.augment")
        profile.set("difix.camera_spread_deg", round(spread, 3), "difix.augment")
        profile.save()
    logger.info(f"Wrote {len(pseudo_entries)} pseudo views to {out_dir} (before|after in {debug_dir})")
    return out_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="COLMAP dir from ACE-Zero (sparse/0, images/)")
    ap.add_argument("--ckpt", required=True, help="3DGS checkpoint to render pseudo views from")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_views", type=int, default=24)
    ap.add_argument("--max_angle", type=float, default=20.0, help="Max orbit angle (deg) from a real camera")
    ap.add_argument("--pseudo_weight", type=float, default=0.5, help="Base loss weight of pseudo views")
    ap.add_argument("--min_conf", type=float, default=None,
                    help="Confidence mapped to the weight floor (default: min(P5, median/2) of ACE confidences)")
    ap.add_argument("--weight_floor", type=float, default=None,
                    help="Minimum weight for registered frames (default: P5/median clipped to [0.1, 0.5])")
    ap.add_argument("--test_every", type=int, default=8, help="Must match the trainer's test_every")
    ap.add_argument("--pose_rel_scale", type=float, default=0.1,
                    help="Pose correction (fraction of camera spacing) that maps to weight exp(-1)")
    ap.add_argument("--ckpt_data_dir", default=None,
                    help="Dataset the checkpoint was trained on, if not --data_dir (for pose_adjust indexing)")
    ap.add_argument("--seed", type=int, default=0, help="Pseudo-camera placement seed (0 = deterministic)")
    ap.add_argument("--model_id", default="nvidia/difix_ref",
                    help="nvidia/difix_ref (conditions on the source photo, less hallucination) or nvidia/difix")
    ap.add_argument("--no_difix", action="store_true", help="Use raw renders (ablation)")
    ap.add_argument("--loop", type=int, default=None, help="Feedback-loop index recorded with the round stats")
    ap.add_argument("--profile", default=None,
                    help="scene_profile.json to append difix.rounds to (default: $HYPERSPLAT_PROFILE)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
