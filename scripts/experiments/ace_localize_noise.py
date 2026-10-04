#!/usr/bin/env python3
"""Per-frame pose noise of the stream sidecar's DSAC* localization (the poses 3DGS trains on).

Localizes each frame K times against one ACE head with the sidecar's own ``localize`` and
reports how much the pose moves between calls (RANSAC sampling only: the network output is
the same every time), per RANSAC setting. Run with the ACE-Zero interpreter from
scripts/acezero, under the shared GPU lock:

    cd scripts/acezero && flock ../../experiments/.gpu.lock $ACE0_PY ../experiments/ace_localize_noise.py \\
        --head RUN/hybrid_pass.pt --poses RUN/poses_final.txt --frames FRAMES --repeats 8
"""

import argparse
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "acezero"
sys.path.insert(0, str(HERE))


def rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--poses", required=True, help="pose file of the map (focal)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--hypotheses", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--max-tries", type=int, nargs="+", default=[16, 1000000])
    ap.add_argument("--image-resolution", type=int, default=480)
    a = ap.parse_args()

    import torch
    from stream_sidecar import Sidecar

    sc = Sidecar(tempfile.mkdtemp(prefix="locnoise_"), str(HERE / "ace_encoder_pretrained.pt"),
                 64, 10.0, 1500, out=None)
    _, sc.focal = Sidecar.read_pose_file(a.poses)
    sc.image_resolution = a.image_resolution
    net = sc.load_head(a.head)
    frames = sorted(str(p) for p in Path(a.frames).glob("*.jpg"))
    for hyp, tries in itertools.product(a.hypotheses, a.max_tries):
        sc.hypotheses, sc.max_tries = hyp, tries
        centers, rots, confs, t0 = [], [], [], time.time()
        for fr in frames:
            cs, rs, cf = [], [], []
            for _ in range(a.repeats):
                row = sc.localize(fr, net)
                c2w = np.linalg.inv(np.array(row["w2c"]).reshape(4, 4))
                cs.append(c2w[:3, 3])
                rs.append(c2w[:3, :3])
                cf.append(row["conf"])
            centers.append(np.array(cs))
            rots.append(rs)
            confs.append(cf)
        ms = (time.time() - t0) * 1000 / (len(frames) * a.repeats)
        mean_c = np.stack([c.mean(0) for c in centers])
        extent = np.linalg.norm(mean_c - mean_c.mean(0), axis=1).max()
        t_jit = np.array([np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()) for c in centers]) / extent
        r_jit = np.array([np.mean([rot_deg(r, rs[0]) for r in rs[1:]]) for rs in rots])
        conf = np.array(confs)
        print(json.dumps({"hypotheses": hyp, "max_tries": tries, "ms_per_frame": round(ms, 1),
                          "t_jitter_rel_median": float(np.median(t_jit)), "t_jitter_rel_max": float(t_jit.max()),
                          "rot_jitter_deg_median": float(np.median(r_jit)), "rot_jitter_deg_max": float(r_jit.max()),
                          "conf_median": float(np.median(conf)), "frames": len(frames)}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
