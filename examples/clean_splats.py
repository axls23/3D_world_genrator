"""Post-hoc floater cleanup for a trained gsplat checkpoint.

Removes Gaussians that typically show up as noise in novel views:
  - near-transparent ones (low opacity)
  - oversized ones (large max scale relative to the scene)
  - spatial outliers (far from the bulk of the scene)
  - non-finite parameters
and limits needle-like anisotropy by inflating the smallest axes.

Usage:
    python examples/clean_splats.py --ckpt path/to/ckpt_6999_rank0.pt \
        --out path/to/ckpt_6999_clean.pt
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_opacity", type=float, default=0.05)
    parser.add_argument("--max_scale_quantile", type=float, default=0.995,
                        help="Drop Gaussians whose largest axis exceeds this quantile")
    parser.add_argument("--dist_quantile", type=float, default=0.95,
                        help="Drop Gaussians farther from the scene median than this quantile")
    parser.add_argument("--max_aniso", type=float, default=10.0,
                        help="Clamp max/min axis ratio (0 disables)")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]
    n = splats["means"].shape[0]

    keep = torch.ones(n, dtype=torch.bool)
    for v in splats.values():
        keep &= torch.isfinite(v.reshape(n, -1)).all(dim=-1)

    opacity = torch.sigmoid(splats["opacities"])
    keep &= opacity >= args.min_opacity
    print(f"after opacity/finite filter: {keep.sum().item()} / {n}")

    # torch.quantile has an input size limit; estimate thresholds on a subsample
    def quantile(x, q):
        idx = torch.randperm(x.numel())[:1_000_000]
        return torch.quantile(x[idx].float(), q).item()

    max_scale = torch.exp(splats["scales"]).max(dim=-1).values
    keep &= max_scale <= quantile(max_scale[keep], args.max_scale_quantile)
    print(f"after scale filter: {keep.sum().item()} / {n}")

    means = splats["means"]
    center = means[keep].median(dim=0).values
    dist = (means - center).norm(dim=-1)
    keep &= dist <= quantile(dist[keep], args.dist_quantile)
    print(f"after distance filter: {keep.sum().item()} / {n}")

    splats = {k: v[keep] for k, v in splats.items()}

    if args.max_aniso > 0:
        log_s = splats["scales"]
        floor = log_s.max(dim=-1, keepdim=True).values - torch.log(torch.tensor(args.max_aniso))
        splats["scales"] = torch.maximum(log_s, floor)

    ckpt["splats"] = splats
    # Optimizer states no longer match the pruned tensors
    for k in ("optimizers", "pose_optimizers", "app_optimizers", "bil_grid_optimizers", "schedulers"):
        ckpt.pop(k, None)
    torch.save(ckpt, args.out)
    print(f"saved {splats['means'].shape[0]} Gaussians to {args.out}")


if __name__ == "__main__":
    main()
