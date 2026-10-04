#!/usr/bin/env python3
"""ACE localization sidecar for the streaming 3DGS pipeline (hypersplat/stream).

Runs inside the ACE-Zero env (py3.8, torch 2.0, compiled dsacstar) next to the 3DGS trainer,
which cannot import dsacstar. Talks Apache Arrow IPC streams: COMMAND_SCHEMA batches on stdin,
EVENT_SCHEMA batches on stdout (hypersplat/stream/schema.py). Everything else - ACE logs,
dsacstar's C++ printf, child processes - goes to stderr so the stream stays clean.

Commands (cmd, frames, args, params-JSON):
    bootstrap  frames, args; params: focal
        Build the first map with ace_zero_hybrid.py on these frames (args: its CLI options),
        then localize every frame against it. Runs in the background.
    localize   frames[0]
        Register one frame against the current map (~35 ms).
    refit      frames, args; params: iterations, conf_threshold
        Grow the map: retrain the head warm-started from the current one, supervised by the
        current poses (so the coordinate frame is kept), then re-localize every frame.
        Runs in the background; localize keeps using the old map meanwhile.
    load       frames; params: head, pose_file, image_resolution
        Adopt an existing ACE map (e.g. a batch ACE-Zero run) and localize the frames on it.
    forget     frames      drop evicted frames from the pose table
    shutdown

Events (kind, version, frame, w2c, conf, focal, points, pix, meta):
    ready      one row
    pose       one row per localize
    map        one batch, a row per frame; meta: reason, seconds
    error      meta: cmd, msg

w2c is world-to-camera (OpenCV / COLMAP convention, as in ACE pose files); focal is in pixels
of the frame files. points are the frame's scene coordinates that reproject within the RANSAC
threshold under the found pose (flat N*3), pix their image positions normalized to [0, 1]
(flat N*2), so the client can sample colors at its own resolution.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # repo root: hypersplat.stream.schema

import dataset_io  # noqa: E402
import dsacstar  # noqa: E402
from ace_network import Regressor  # noqa: E402
from dataset import CamLocDataset  # noqa: E402
from hypersplat.stream.schema import (  # noqa: E402
    EVENT_SCHEMA, IpcWriter, event_batch, ipc_batches, parse_meta,
)

_logger = logging.getLogger("ace-sidecar")

MAP_ID = "stream_map"


class Sidecar:
    def __init__(self, work_dir, encoder_path, hypotheses, threshold, max_points, out,
                 device="cuda"):
        self.work_dir = Path(work_dir)
        self.map_dir = self.work_dir / "map"
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_state = torch.load(encoder_path, map_location="cpu")
        self.hypotheses = hypotheses
        self.threshold = threshold
        self.max_points = max_points
        self.device = torch.device(device)
        self.out = out                  # IpcWriter for EVENT_SCHEMA

        self.network = None             # current Regressor, swapped atomically under net_lock
        self.head_path = None
        self.focal = -1.0               # focal of the frame files, refined by ACE
        self.image_resolution = 480
        self.version = 0
        self.poses = {}                 # frame path -> (w2c 4x4, confidence) in the current map
        self.net_lock = threading.Lock()
        self.out_lock = threading.Lock()
        self.busy = threading.Lock()    # one background bootstrap/refit/load at a time
        self.stopping = threading.Event()
        self.rng = np.random.default_rng(0)
        self.child = None               # running ACE script (bootstrap / refit), for shutdown

    # ------------------------------------------------------------------ io
    def emit(self, rows):
        batch = event_batch(rows)
        with self.out_lock:
            self.out.write(batch)

    def error(self, cmd, msg):
        self.emit([{"kind": "error", "version": self.version, "meta": {"cmd": cmd, "msg": msg}}])

    def run_cmd(self, cmd):
        cmd = [str(c) for c in cmd]
        _logger.info("run: %s", " ".join(cmd[1:4]))
        self.child = subprocess.Popen(cmd, cwd=str(HERE), stdout=sys.stderr, stderr=sys.stderr)
        if self.stopping.is_set():  # shutdown raced the start: don't leave it holding the GPU
            self.terminate_child()
        try:
            code = self.child.wait()
        finally:
            self.child = None
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)

    def terminate_child(self):
        child = self.child
        if child is not None and child.poll() is None:
            _logger.info("terminating running ACE job")
            child.terminate()
            try:
                child.wait(5)
            except subprocess.TimeoutExpired:
                child.kill()

    # ------------------------------------------------------------------ map
    def load_head(self, head_path):
        head_state = torch.load(str(head_path), map_location="cpu")
        net = Regressor.create_from_split_state_dict(self.encoder_state, head_state).to(self.device)
        net.eval()
        return net

    @staticmethod
    def read_pose_file(pose_file):
        """ACE pose file -> {frame: (w2c, conf)}, focal."""
        out, focal = {}, None
        for line in Path(pose_file).read_text().splitlines():
            t = line.split()
            if len(t) != 10:
                continue
            qw, qx, qy, qz = (float(v) for v in t[1:5])
            w2c = np.eye(4)
            w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            w2c[:3, 3] = [float(v) for v in t[5:8]]
            out[t[0]] = (w2c, float(t[9]))
            focal = float(t[8])
        return out, focal

    def write_pose_file(self, path, conf_min):
        with open(path, "w") as f:
            for frame, (w2c, conf) in sorted(self.poses.items()):
                if conf >= conf_min and Path(frame).exists():
                    dataset_io.write_pose_to_pose_file(f, rgb_file=frame, pose=w2c,
                                                       confidence=conf, focal_length=self.focal)

    # ------------------------------------------------------------------ localize
    @torch.no_grad()
    def localize(self, frame, network):
        row = {"kind": "pose", "version": self.version, "frame": frame, "conf": 0.0,
               "focal": self.focal}
        if network is None:
            return row
        ds = CamLocDataset(rgb_files=frame, image_short_size=self.image_resolution,
                           use_heuristic_focal_length=self.focal <= 0)
        if self.focal > 0:
            ds.set_external_focal_length(self.focal)
        image, _, _, _, K, _, _, _, _ = ds[0]
        with torch.cuda.amp.autocast(enabled=True):
            sc = network(image.unsqueeze(0).to(self.device))
        sc = sc.float().cpu()[0]  # 3, h, w
        c2w = torch.zeros((4, 4))
        f, cx, cy = K[0, 0].item(), K[0, 2].item(), K[1, 2].item()
        # Same arguments as register_mapping.py (inlier alpha / max pixel error 100, 16 tries)
        conf = dsacstar.forward_rgb(sc.unsqueeze(0), c2w, self.hypotheses, self.threshold, f, cx, cy,
                                    100, 100, network.OUTPUT_SUBSAMPLE, 1305, 16)
        w2c = np.linalg.inv(c2w.numpy().astype(np.float64))
        points, pix = self._inlier_points(sc.numpy(), w2c, f, cx, cy, image.shape[1:],
                                          network.OUTPUT_SUBSAMPLE)
        row.update(w2c=w2c.reshape(-1).tolist(), conf=float(conf),
                   focal=self.focal if self.focal > 0 else ds.get_focal_length(0),
                   points=points.astype(np.float32).reshape(-1).tolist(),
                   pix=pix.astype(np.float32).reshape(-1).tolist())
        return row

    def _inlier_points(self, sc, w2c, f, cx, cy, hw, sub):
        """Scene coordinates that reproject within the RANSAC threshold under the found pose."""
        _, h, w = sc.shape
        X = sc.reshape(3, -1).T
        ys, xs = np.mgrid[0:h, 0:w]
        px = np.stack([xs.reshape(-1) * sub + sub / 2, ys.reshape(-1) * sub + sub / 2], axis=1)
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = np.stack([f * Xc[:, 0] / z + cx, f * Xc[:, 1] / z + cy], axis=1)
            err = np.linalg.norm(proj - px, axis=1)
        keep = np.where((z > 1e-3) & (err < self.threshold) & np.isfinite(err))[0]
        if len(keep) > self.max_points:
            keep = self.rng.choice(keep, self.max_points, replace=False)
        H, W = hw
        return X[keep], px[keep] / np.array([W, H], dtype=np.float64)

    # ------------------------------------------------------------------ background jobs
    def _publish_map(self, network, head_path, frames, reason, t0):
        rows = [self.localize(frame, network) for frame in frames if Path(frame).exists()]
        with self.net_lock:
            self.network, self.head_path = network, Path(head_path)
            self.version += 1
            for r in rows:
                if r.get("w2c") is not None:
                    self.poses[r["frame"]] = (np.array(r["w2c"]).reshape(4, 4), r["conf"])
        torch.cuda.empty_cache()  # the GPU is shared with 3DGS training: return cached blocks
        meta = {"reason": reason, "seconds": round(time.time() - t0, 1)}
        for r in rows:
            r.update(kind="map", version=self.version, meta=meta)
        # A map event always carries at least one row (the version / focal header)
        self.emit(rows or [{"kind": "map", "version": self.version, "focal": self.focal, "meta": meta}])

    def bootstrap(self, frames, args, params):
        t0 = time.time()
        stage = Path(tempfile.mkdtemp(prefix="boot_", dir=str(self.work_dir)))
        try:
            # ace_zero_hybrid takes a glob: stage exactly the bootstrap frames (symlinks)
            for fr in frames:
                os.symlink(fr, str(stage / Path(fr).name))
            out = self.map_dir / "bootstrap"
            if "--image_resolution" in args:
                self.image_resolution = int(args[args.index("--image_resolution") + 1])
            cmd = [sys.executable, "ace_zero_hybrid.py", str(stage / "*.jpg"), str(out)] + args
            if float(params.get("focal", -1)) > 0:
                cmd += ["--use_external_focal_length", str(params["focal"])]
            self.run_cmd(cmd)
            _, focal = self.read_pose_file(out / "poses_final.txt")
            if focal:
                self.focal = focal
            self._publish_map(self.load_head(out / "hybrid_pass.pt"), out / "hybrid_pass.pt",
                              frames, "bootstrap", t0)
        except Exception as e:  # report and stay alive: the client retries with more frames
            _logger.exception("bootstrap failed")
            self.error("bootstrap", str(e))
            shutil.rmtree(str(self.map_dir / "bootstrap"), ignore_errors=True)
        finally:
            shutil.rmtree(str(stage), ignore_errors=True)

    def load(self, frames, args, params):
        t0 = time.time()
        try:
            self.image_resolution = int(params.get("image_resolution", self.image_resolution))
            poses, focal = self.read_pose_file(params["pose_file"])
            if focal:
                self.focal = focal
            self._publish_map(self.load_head(params["head"]), params["head"],
                              frames or sorted(poses), "load", t0)
        except Exception as e:
            _logger.exception("load failed")
            self.error("load", str(e))

    def refit(self, frames, args, params):
        t0 = time.time()
        frames = [f for f in frames if Path(f).exists()]
        conf_min = float(params.get("conf_threshold", 500))
        out_dir = self.map_dir / "refit_{:04d}".format(self.version + 1)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # With a pose file train_ace takes its images from it (the glob is ignored), so it
            # trains on the registered frames only - ACE-Zero's growth step: the map extends
            # to what registered edge frames see, and frames that failed may register next
            pose_file = out_dir / "poses_prior.txt"
            with self.net_lock:
                self.write_pose_file(pose_file, conf_min)
            head = out_dir / (MAP_ID + ".pt")
            cmd = [sys.executable, "train_ace.py", str(Path(frames[0]).parent / "*.jpg"), str(head),
                   "--iterations", str(int(params.get("iterations", 4000))),
                   "--use_ace_pose_file", str(pose_file),
                   "--ace_pose_file_conf_threshold", str(conf_min),
                   "--load_weights", str(self.head_path),
                   "--image_resolution", str(self.image_resolution)] + args
            self.run_cmd(cmd)
            self._publish_map(self.load_head(head), head, frames, "refit", t0)
            # Keep only the live head (the work dir is tmpfs): older maps are dead weight
            for old in list(self.map_dir.glob("refit_*")) + [self.map_dir / "bootstrap"]:
                if old != out_dir and old.exists():
                    shutil.rmtree(str(old), ignore_errors=True)
        except Exception as e:
            _logger.exception("refit failed")
            self.error("refit", str(e))
            shutil.rmtree(str(out_dir), ignore_errors=True)

    def start_job(self, cmd, fn, frames, args, params):
        # Jobs run one at a time, and a new one waits for the previous one instead of failing
        # as busy: the client sends the next refit as soon as it sees the previous job's map
        # event, which is emitted before that job's thread releases the lock (a rejected
        # refit used to be dropped for good)
        def run():
            with self.busy:
                if self.stopping.is_set():
                    return
                fn(frames, args, params)
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------ main loop
    def serve(self, commands):
        self.emit([{"kind": "ready", "version": 0}])
        jobs = {"bootstrap": self.bootstrap, "refit": self.refit, "load": self.load}
        for batch in commands:
            for msg in batch.to_pylist():
                cmd, frames, args = msg["cmd"], msg["frames"] or [], msg["args"] or []
                try:
                    params = parse_meta(msg["params"])
                    if cmd == "localize":
                        with self.net_lock:
                            network = self.network
                        row = self.localize(frames[0], network)
                        if row.get("w2c") is not None:
                            with self.net_lock:
                                self.poses[row["frame"]] = (np.array(row["w2c"]).reshape(4, 4), row["conf"])
                        self.emit([row])
                    elif cmd in jobs:
                        self.start_job(cmd, jobs[cmd], frames, args, params)
                    elif cmd == "forget":
                        with self.net_lock:
                            for fr in frames:
                                self.poses.pop(fr, None)
                    elif cmd == "shutdown":
                        self.stopping.set()
                        self.terminate_child()
                        return
                    else:
                        self.error(cmd, "unknown command")
                except Exception as e:
                    _logger.exception("command %s failed", cmd)
                    self.error(cmd, str(e))


def claim_stdout():
    """Keep fd 1 for the Arrow stream only: dsacstar (C++) and the ACE scripts print progress
    straight to the stdout fd. Returns a binary writer on a private copy of the original
    stdout; fd 1 and sys.stdout now point at stderr."""
    proto = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def main():
    proto = claim_stdout()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[ace-sidecar] %(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", required=True, help="Scratch dir for maps (tmpfs recommended)")
    ap.add_argument("--encoder_path", default=str(HERE / "ace_encoder_pretrained.pt"))
    ap.add_argument("--hypotheses", type=int, default=64, help="RANSAC hypotheses per frame")
    ap.add_argument("--threshold", type=float, default=10.0, help="RANSAC inlier threshold (px)")
    ap.add_argument("--max_points", type=int, default=1500, help="Seed points returned per frame")
    args = ap.parse_args()
    out = IpcWriter(proto, EVENT_SCHEMA)
    sidecar = Sidecar(args.work_dir, args.encoder_path, args.hypotheses, args.threshold,
                      args.max_points, out)
    try:
        sidecar.serve(ipc_batches(sys.stdin.buffer))
    finally:
        sidecar.stopping.set()
        sidecar.terminate_child()  # stdin closed (client gone) counts as shutdown too
        out.close()


if __name__ == "__main__":
    main()
