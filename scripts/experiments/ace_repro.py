#!/usr/bin/env python3
"""Reproducibility of the stream's ACE-Zero bootstrap, without the trainer.

Extracts the bootstrap keyframes of a video exactly as StreamSession does (same gate, same
ACE-resolution JPEGs, same ACE schedule / arguments), then runs the hybrid ACE-Zero bootstrap
N times on them and compares the maps: where the LR cooldown triggered, total iterations,
registered frames, and camera agreement with the first run after a Sim3 alignment
(ATE relative to the trajectory extent, rotation error).

    python scripts/experiments/ace_repro.py demo/input/drone_demo.mp4 --runs 3 --out /tmp/ace_repro
    python scripts/experiments/ace_repro.py ... --contend    # extra GPU load during each run
    python scripts/experiments/ace_repro.py ... --extra-args --deterministic True

--lock FILE holds a shared GPU lock around each bootstrap (other jobs interleave between them).
"""

import argparse
import fcntl
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ACE_DIR = ROOT / "scripts" / "acezero"
sys.path.insert(0, str(ROOT))


def extract(video: str, out: Path, n: int, max_side: int = 960, short_side: int = 480):
    """First ``n`` keyframes of ``video`` as the stream bootstrap sees them; returns the gate."""
    import cv2

    from hypersplat.stream.keyframes import KeyframeGate
    from hypersplat.stream.source import FrameSource, encode_jpeg

    out.mkdir(parents=True, exist_ok=True)
    gate = KeyframeGate(min_motion=8.0)
    src = FrameSource(video, max_side, "fast")
    size = None
    k = 0
    try:
        for frame in src:
            if not gate.offer(frame.image, frame.t).accept:
                continue
            h, w = frame.image.shape[:2]
            scale = min(1.0, short_side / min(h, w))
            img = cv2.resize(frame.image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
                if scale < 1.0 else frame.image
            size = size or (img.shape[1], img.shape[0])
            (out / f"kf_{k:06d}.jpg").write_bytes(encode_jpeg(img, 92))
            k += 1
            if k >= n:
                break
    finally:
        src.close()
    return gate, size


def ace_args(gate, size, n: int):
    """The arguments StreamSession._start_bootstrap gives the sidecar for these keyframes."""
    from hypersplat.pipeline.params.strategies.ace_schedule import build_ace_args
    from hypersplat.stream.session import StreamConfig, StreamSession

    s = StreamSession.__new__(StreamSession)
    s.cfg = StreamConfig(source="unused.mp4")
    s.gate, s.file_size = gate, size
    return build_ace_args(s._ace_schedule(n), hybrid=True)


def read_poses(path: Path):
    from scipy.spatial.transform import Rotation

    out, focal = {}, None
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) != 10:
            continue
        qw, qx, qy, qz = (float(v) for v in t[1:5])
        w2c = np.eye(4)
        w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, 3] = [float(v) for v in t[5:8]]
        out[Path(t[0]).name] = (np.linalg.inv(w2c), float(t[9]))
        focal = float(t[8])
    return out, focal


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def compare(ref, other, conf_min: float):
    """ATE (fraction of the reference trajectory extent) and mean rotation error (deg) over
    frames registered (conf >= conf_min) in both maps, after a Sim3 fit of camera centres."""
    names = sorted(k for k in ref if k in other and ref[k][1] >= conf_min and other[k][1] >= conf_min)
    if len(names) < 3:
        return None
    A = np.stack([other[k][0][:3, 3] for k in names])
    B = np.stack([ref[k][0][:3, 3] for k in names])
    s, R, t = umeyama(A, B)
    err = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    extent = np.linalg.norm(B - B.mean(0), axis=1).max()
    rot = [np.degrees(np.arccos(np.clip((np.trace(R @ other[k][0][:3, :3] @ ref[k][0][:3, :3].T) - 1) / 2, -1, 1)))
           for k in names]
    return {"n": len(names), "ate_rel": float(np.sqrt((err ** 2).mean()) / extent),
            "rot_deg": float(np.mean(rot)), "scale": float(s)}


def contention():
    """A background CUDA load (other GPU work during a bootstrap, as when the trainer or a
    previous run shares the GPU)."""
    code = ("import torch,time\nx=torch.randn(2048,2048,device='cuda')\n"
            "while True:\n  y=x@x\n  torch.cuda.synchronize()\n  time.sleep(0.002)\n")
    return subprocess.Popen([sys.executable, "-c", code])


def run_once(frames: Path, out: Path, args, log: Path, contend: bool, lock=None):
    fh = open(lock, "a") if lock else None
    if fh is not None:
        fcntl.flock(fh, fcntl.LOCK_EX)
    hog = contention() if contend else None
    t0 = time.time()
    try:
        from hypersplat.pipeline.wrappers.perception import acezero_python_env

        python, env = acezero_python_env(ACE_DIR)  # the interpreter the stream sidecar uses
        with open(log, "w") as f:
            rc = subprocess.call([python, "ace_zero_hybrid.py", str(frames / "*.jpg"), str(out)] + args,
                                 cwd=str(ACE_DIR), stdout=f, stderr=subprocess.STDOUT, env=env)
    finally:
        if hog is not None:
            hog.kill()
            hog.wait()
        if fh is not None:
            fh.close()  # releases the lock
    text = log.read_text()
    trig = re.findall(r"Iteration:\s+(\d+)\|\s*\d+ / .*\n.*Starting learning rate cooldown", text)
    totals = re.findall(r"Iteration:\s+\d+\|\s*(\d+) /", text)
    reason = re.findall(r"Starting learning rate cooldown\. \(Reason: max duration (\w+), dynamic (\w+)\)", text)
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "total_iters": int(totals[-1]) if totals else None,
            "cooldown_after_iter": int(trig[0]) if trig else None,
            "cooldown_reason": "dynamic" if reason and reason[0][1] == "True" else ("max" if reason else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contend", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--lock", help="flock this file around each bootstrap (shared GPU)")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    out = Path(a.out)
    frames = out / "frames"
    gate, size = extract(a.video, frames, a.frames)
    args = ace_args(gate, size, a.frames) + list(a.extra_args)
    (out / "args.json").write_text(json.dumps(args))
    conf_min = float(args[args.index("--registration_confidence") + 1]) if "--registration_confidence" in args else 500.0
    ref = None
    rows = []
    for i in range(a.runs):
        rd = out / f"{a.tag}_{i}"
        row = {"run": rd.name, **run_once(frames, rd, args, out / f"{a.tag}_{i}.log", a.contend, a.lock)}
        pf = rd / "poses_final.txt"
        if pf.exists():
            poses, focal = read_poses(pf)
            row.update(registered=sum(c >= conf_min for _, c in poses.values()), focal=round(focal or -1, 2))
            ref_file = out / "reference_poses.txt"
            if ref is None and ref_file.exists():
                ref = read_poses(ref_file)[0]
            if ref is None:
                ref_file.write_text(pf.read_text())
                ref = poses
            row["vs_ref"] = compare(ref, poses, conf_min)
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps({"tag": a.tag, "contend": a.contend, "extra": a.extra_args, **row}) + "\n")


if __name__ == "__main__":
    main()
