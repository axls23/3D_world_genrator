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
    seed       frames[0]; params: depth, focal, image_resolution, num_head_blocks, seed_slots,
               seed_min_iters, seed_max_iters, seed_min_inliers, seed_lr, seed_keep_depth,
               conf_threshold + the mapper params
        Cold start, instead of bootstrap + mapper: a fresh head learns this one frame at the
        identity pose with scene coordinate targets from its depth map (``depth``: .npy,
        camera z, 0 = invalid) - ACE-Zero's seed stage - then continuous mapping starts on it
        (the frame stays anchored: it fixes the gauge). Runs in the background; the map
        event (reason "seed") carries the seed frame only. Focal: ``focal`` (px of the frame
        file) or ACE's heuristic, 70% of the image diagonal.
    forget     frames      drop evicted frames from the pose table
    mapper     frames; params: conf_threshold, iters_per_tick, reloc_per_tick, publish_every,
               max_backlog, lr, passes, samples_per_pass, max_frames
        Switch to continuous mapping (stream_mapper.py): the head keeps training a few
        iterations per tick on the registered frames, and frames are relocalized round-robin
        (`frames`: every keyframe the client has). Refits are refused from then on.
    tick       params: n   run n mapper units (iterations + relocalizations); the client sends one
               per 3DGS step, so mapping keeps pace with training (backlog capped)
    profile    params: units, out   torch.profiler over the next ``units`` mapper units (all
               threads: mapping and localization), summary + trace written to ``out``
    rings      params: delta, seed, feedback   paths of shared-memory rings (hypersplat/stream/
               ring.py) created by the client: from now on pose and reloc rows go there
               instead of the pipe (fixed-size records, no serialization; a full ring blocks
               the sender, which is the backpressure on mapping)
    shutdown

Events (kind, version, frame, w2c, conf, focal, points, pix, meta):
    ready      one row
    pose       one row per localize
    map        one batch, a row per frame; meta: reason, seconds
    reloc      one row per round-robin relocalization by the continuous mapper
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
import zlib
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
from stream_mapper import ContinuousMapper  # noqa: E402
from reloc_schedule import RelocScheduler  # noqa: E402
from dataset import CamLocDataset  # noqa: E402
from hypersplat.stream.ring import (DELTA_DTYPE, KIND_ANCHOR, KIND_POSE,  # noqa: E402
                                    KIND_RELOC, SEED_DTYPE, SEED_MAX_POINTS, ShmRing,
                                    frame_kid)
from hypersplat.stream.schema import (  # noqa: E402
    EVENT_SCHEMA, IpcWriter, event_batch, ipc_batches, parse_meta,
)

_logger = logging.getLogger("ace-sidecar")

MAP_ID = "stream_map"


class Sidecar:
    def __init__(self, work_dir, encoder_path, hypotheses, threshold, max_points, out,
                 device="cuda", max_tries=16):
        self.work_dir = Path(work_dir)
        self.map_dir = self.work_dir / "map"
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_state = torch.load(encoder_path, map_location="cpu")
        self.hypotheses = hypotheses
        self.max_tries = max_tries      # re-draws of an invalid RANSAC hypothesis
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
        self.seed = 0                   # seed points per frame: rng from (seed, frame name)
        self.child = None               # running ACE script (bootstrap / refit), for shutdown
        # Continuous mapping (``mapper`` command): the mapper, its thread, the tick budget the
        # client grants, every known frame (round-robin relocalization) and registrations
        # waiting to be encoded into the patch buffer
        self.mapper = None
        self.mapper_cfg = {}
        self.mapper_thread = None
        self.tick_cv = threading.Condition()
        self.tick_budget = 0
        self.known = []
        self.reloc_pos = 0
        self.pending_add = []
        self.units = 0
        # Anchoring: a frame whose relocalized pose stayed within anchor_tol (of the camera path
        # extent) anchor_after times in a row keeps that pose as its fixed target and leaves
        # the round-robin; relocalization effort goes to the frontier and unregistered frames
        self.anchored = set()
        self.stable_count = {}
        self.extent = None
        # Relocalization scheduling and mapper idling (reloc_schedule.py)
        self.sched = RelocScheduler()
        # Shared-memory rings (``rings`` command): pose / reloc rows bypass the Arrow pipe
        self.rings = None
        self.ring_lock = threading.Lock()   # one producer per ring: localize and mapper threads
        self.ring_seq = 0
        self.ring_fallbacks = 0
        self.prof_request = None        # (units, out dir) from the ``profile`` command

    # ------------------------------------------------------------------ io
    def emit(self, rows):
        if (self.rings is not None and len(rows) == 1 and rows[0].get("kind") in ("pose", "reloc")
                and self._emit_ring(rows[0])):
            return
        batch = event_batch(rows)
        with self.out_lock:
            self.out.write(batch)

    def open_rings(self, params):
        rings = {"delta": ShmRing(params["delta"], DELTA_DTYPE),
                 "seed": ShmRing(params["seed"], SEED_DTYPE)}
        with self.ring_lock:
            self.rings = rings
        _logger.info("pose rows now go through shared-memory rings (%s)", params["delta"])

    def _push_wait(self, ring, rec, timeout=5.0):
        """Push one record, waiting while the ring is full (the consumer is behind)."""
        deadline = time.time() + timeout
        while ring.push(rec) == 0:
            if self.stopping.is_set() or time.time() > deadline:
                return False
            time.sleep(0.002)
        return True

    def _emit_ring(self, row):
        """One pose / reloc row as a DELTA record (+ its seed points as a SEED record).
        False: not sent (not a stream frame, or the ring stayed full): use the pipe."""
        kid = frame_kid(row.get("frame") or "")
        if kid < 0:
            return False
        meta = row.get("meta") or {}
        rec = np.zeros(1, DELTA_DTYPE)
        r = rec[0]
        r["kid"], r["step"], r["mapper_tick"] = kid, -1, self.units
        r["map_version"] = int(row.get("version") or 0)
        r["kind"] = KIND_ANCHOR if meta.get("anchored") else (KIND_RELOC if row["kind"] == "reloc" else KIND_POSE)
        r["anchored"] = 1 if meta.get("anchored") else 0
        r["conf"] = float(row.get("conf") or 0.0)
        r["focal"] = float(row.get("focal") or -1.0)
        m = self.mapper
        r["inlier_frac"] = float(m.inlier_ema) if m is not None and m.inlier_ema is not None else np.nan
        r["move"] = np.nan
        r["w2c"] = np.asarray(row["w2c"], np.float64).reshape(16) if row.get("w2c") is not None else np.nan
        pts = np.asarray(row.get("points") or [], np.float32).reshape(-1, 3)[:SEED_MAX_POINTS]
        pix = np.asarray(row.get("pix") or [], np.float32).reshape(-1, 2)[:SEED_MAX_POINTS]
        with self.ring_lock:
            seq = self.ring_seq + 1
            r["seq"], r["n_seeds"] = seq, len(pts)
            if len(pts):
                seed = np.zeros(1, SEED_DTYPE)
                seed[0]["seq"], seed[0]["kid"] = seq, kid
                seed[0]["map_version"], seed[0]["n"] = r["map_version"], len(pts)
                seed[0]["xyz"][:len(pts)] = pts
                seed[0]["pix"][:len(pix)] = pix
                if not self._push_wait(self.rings["seed"], seed):
                    self.ring_fallbacks += 1
                    return False
            # Seeds first: the consumer finds them published when it reads the delta
            if not self._push_wait(self.rings["delta"], rec):
                self.ring_fallbacks += 1
                return False   # an orphan seed (if any) is dropped by the consumer by seq
            self.ring_seq = seq
        return True

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
    def localize(self, frame, network, w2c=None):
        """Register a frame (RANSAC pose + inlier scene coordinates). ``w2c`` given: no
        RANSAC, the inliers (count = conf) under that pose (the seed frame)."""
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
        f, cx, cy = K[0, 0].item(), K[0, 2].item(), K[1, 2].item()
        conf = None
        if w2c is None:
            c2w = torch.zeros((4, 4))
            # Same arguments as register_mapping.py (inlier alpha / max pixel error 100, 16 tries)
            conf = dsacstar.forward_rgb(sc.unsqueeze(0), c2w, self.hypotheses, self.threshold, f, cx, cy,
                                        100, 100, network.OUTPUT_SUBSAMPLE, 1305, self.max_tries)
            w2c = np.linalg.inv(c2w.numpy().astype(np.float64))
        points, pix, n_inl = self._inlier_points(Path(frame).name, sc.numpy(), w2c, f, cx, cy,
                                                 image.shape[1:], network.OUTPUT_SUBSAMPLE)
        if conf is None:
            conf = n_inl
        row.update(w2c=w2c.reshape(-1).tolist(), conf=float(conf),
                   focal=self.focal if self.focal > 0 else ds.get_focal_length(0),
                   points=points.astype(np.float32).reshape(-1).tolist(),
                   pix=pix.astype(np.float32).reshape(-1).tolist())
        return row

    def _inlier_points(self, frame_key, sc, w2c, f, cx, cy, hw, sub):
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
        n_inl = len(keep)
        if len(keep) > self.max_points:
            # Per-frame rng: the same frame gets the same seed points whatever was localized
            # before it (a process-wide rng made them depend on the call history)
            rng = np.random.default_rng([self.seed, zlib.crc32(str(frame_key).encode())])
            keep = rng.choice(keep, self.max_points, replace=False)
        H, W = hw
        return X[keep], px[keep] / np.array([W, H], dtype=np.float64), n_inl

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
            if params.get("cache_dir"):
                self._save_cache(Path(params["cache_dir"]), out)
            self._publish_map(self.load_head(out / "hybrid_pass.pt"), out / "hybrid_pass.pt",
                              frames, "bootstrap", t0)
        except Exception as e:  # report and stay alive: the client retries with more frames
            if self.stopping.is_set():  # shutdown terminated it: not a failure
                _logger.info("bootstrap cancelled (shutdown)")
                return
            _logger.exception("bootstrap failed")
            self.error("bootstrap", str(e))
            shutil.rmtree(str(self.map_dir / "bootstrap"), ignore_errors=True)
        finally:
            shutil.rmtree(str(stage), ignore_errors=True)

    def _save_cache(self, cache, out):
        """Keep the bootstrap map (head, poses with focal, ACE resolution) for later runs on
        the same capture: they ``load`` it instead of re-bootstrapping."""
        try:
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(out / "hybrid_pass.pt"), str(cache / "head.pt"))
            shutil.copyfile(str(out / "poses_final.txt"), str(cache / "poses.txt"))
            meta = json.loads((cache / "meta.json").read_text()) if (cache / "meta.json").exists() else {}
            meta["image_resolution"] = self.image_resolution
            (cache / "meta.json").write_text(json.dumps(meta))
        except OSError as e:
            _logger.warning(f"map cache not written: {e}")

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
            if self.stopping.is_set():  # training ended mid-refit: not a failure
                _logger.info("refit cancelled (shutdown)")
                shutil.rmtree(str(out_dir), ignore_errors=True)
                return
            _logger.exception("refit failed")
            self.error("refit", str(e))
            shutil.rmtree(str(out_dir), ignore_errors=True)

    # ------------------------------------------------------------------ continuous mapping
    @staticmethod
    def _mapper_params(params):
        """(loop cfg, ContinuousMapper kwargs) from mapper / seed command params."""
        cfg = {"conf_threshold": 500.0, "iters_per_tick": 4, "reloc_every": 2,
               "publish_every": 25, "max_backlog": 4, "adopt_min_inliers": 0.5,
               "anchor_tol": 0.01, "anchor_after": 2}
        cfg.update({k: v for k, v in params.items() if k in cfg})
        kw = {k: params[k] for k in ("lr", "passes", "samples_per_pass", "max_frames", "progress")
              if k in params}
        return cfg, kw

    def start_mapper(self, frames, params):
        if self.mapper is not None:
            return  # cold start: the seed job started it
        if self.head_path is None:
            raise RuntimeError("mapper needs a map (bootstrap or load first)")
        cfg, kw = self._mapper_params(params)
        head_state = torch.load(str(self.head_path), map_location="cpu")
        mapper = ContinuousMapper(self.encoder_state, head_state, self.device,
                                  image_resolution=self.image_resolution, seed=self.seed, **kw)
        t0 = time.time()
        with self.net_lock:
            reg = [f for f, (_, c) in sorted(self.poses.items()) if c >= cfg["conf_threshold"]]
            poses = {f: self.poses[f][0] for f in reg}
        for f in reg:
            if Path(f).exists():
                mapper.add_frame(f, poses[f], self.focal)
        self._run_mapper(mapper, cfg, frames)
        _logger.info(f"continuous mapper on {len(mapper.frames)} registered frames "
                     f"({len(self.known)} known), encoded in {time.time() - t0:.1f} s; {cfg} {kw}")

    def _run_mapper(self, mapper, cfg, frames):
        with self.net_lock:
            self.known = sorted(set(frames) | set(self.poses) | set(self.known))
            self.mapper, self.mapper_cfg = mapper, cfg
            self.sched.configure(cfg)
        self.mapper_thread = threading.Thread(target=self._mapper_loop, name="ace-mapper", daemon=True)
        self.mapper_thread.start()

    def seed_map(self, frames, args, params):
        """Cold start (``seed`` command): ACE-Zero's seed stage on frames[0], then continuous
        mapping. Errors are reported (the client retries with another frame)."""
        t0 = time.time()
        frame = frames[0]
        try:
            if self.mapper is not None or self.head_path is not None:
                raise RuntimeError("seed: a map exists already")
            self.image_resolution = int(params.get("image_resolution", self.image_resolution))
            focal = float(params.get("focal", -1))
            if focal <= 0:
                focal = CamLocDataset(rgb_files=frame, image_short_size=self.image_resolution,
                                      use_heuristic_focal_length=True).get_focal_length(0)
            depth = np.load(params["depth"])
            valid = depth[depth > 0]
            if valid.size < 100:
                raise RuntimeError(f"seed depth has {valid.size} valid pixels")
            cfg, kw = self._mapper_params(params)
            mapper = ContinuousMapper(self.encoder_state, None, self.device,
                                      image_resolution=self.image_resolution, seed=self.seed,
                                      num_head_blocks=int(params.get("num_head_blocks", 1)),
                                      mean=(0.0, 0.0, float(np.median(valid))), **kw)
            mapper.add_frame(frame, np.eye(4), focal, depth_file=str(params["depth"]),
                             slots=int(params.get("seed_slots", 8)))
            t1 = time.time()
            iters = mapper.train_seed(min_iters=int(params.get("seed_min_iters", 300)),
                                      max_iters=int(params.get("seed_max_iters", 2000)),
                                      min_inliers=float(params.get("seed_min_inliers", 0.7)),
                                      lr=float(params.get("seed_lr", 2e-3)),
                                      keep_targets=bool(params.get("seed_keep_depth", False)),
                                      stop=self.stopping)
            if self.stopping.is_set():
                _logger.info("seed cancelled (shutdown)")
                return
            net = mapper.snapshot()
            self.focal = focal
            row = self.localize(frame, net, w2c=np.eye(4))
            if row["conf"] < cfg["conf_threshold"]:
                raise RuntimeError(f"seed frame fits only {row['conf']:.0f} inliers "
                                   f"(< {cfg['conf_threshold']:.0f}) after {iters} iterations")
            with self.net_lock:
                self.network = net
                self.version += 1
                self.poses[frame] = (np.eye(4), row["conf"])
                self.anchored.add(frame)   # identity forever: it fixes the gauge
            self._run_mapper(mapper, cfg, frames)
            torch.cuda.empty_cache()
            meta = {"reason": "seed", "seconds": round(time.time() - t0, 1), "iterations": iters,
                    "train_seconds": round(time.time() - t1, 1),
                    "inliers": round(float(mapper.inlier_ema or 0.0), 3)}
            _logger.info(f"seed map from {Path(frame).name}: {iters} iterations in "
                         f"{meta['train_seconds']} s, batch inliers {meta['inliers'] * 100:.1f}%, "
                         f"{row['conf']:.0f} frame inliers, focal {focal:.1f} px; continuous mapping on")
            row.update(kind="map", version=self.version, meta=meta)
            self.emit([row])
        except Exception as e:
            if self.stopping.is_set():
                _logger.info("seed cancelled (shutdown)")
                return
            _logger.exception("seed failed")
            with self.net_lock:
                self.network, self.mapper = None, None
            self.error("seed", str(e))

    def _mapper_loop(self):
        t_log, t_units = time.time(), 0.0
        while not self.stopping.is_set():
            with self.tick_cv:
                while self.tick_budget <= 0 and not self.stopping.is_set():
                    self.tick_cv.wait(0.5)
                if self.stopping.is_set():
                    return
                self.tick_budget -= 1
            self._profile_unit()
            t0 = time.time()
            try:
                self._mapper_unit()
            except Exception as e:  # report, keep the localization service alive
                _logger.exception("mapper unit failed")
                self.error("tick", str(e))
            t_units += time.time() - t0
            if time.time() - t_log > 30:
                m = self.mapper
                _logger.info(f"mapper: {m.iterations} iterations, {len(m.frames)} frames in buffer, "
                             f"{len(self.anchored)} anchored, "
                             f"{len(self.known)} known, {self.sched.waiting} waiting to retry, "
                             f"{self.sched.idle_ticks} idle units, map v{self.version}, inliers(10px) "
                             f"{(m.inlier_ema or 0) * 100:.1f}%, {1000 * t_units / max(1, self.units):.0f} ms/unit")
                t_log = time.time()

    def _profile_unit(self):
        """Start / advance / finish a requested profiler window at a mapper unit boundary."""
        prof = getattr(self, "_prof", None)
        if prof is None:
            if self.prof_request is None:
                return
            from torch.profiler import ProfilerActivity, profile

            self._prof_units, self._prof_out = self.prof_request
            self.prof_request = None
            torch.cuda.synchronize()
            self._prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self._prof.__enter__()
            self._prof_steps = [(self.units, time.time())]
            _logger.info("CUDA profiler on for %d mapper units", self._prof_units)
            return
        self._prof_steps.append((self.units, time.time()))
        if len(self._prof_steps) <= self._prof_units:
            return
        self._prof = None
        torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        from hypersplat.stream.profiling import write_profile

        out = write_profile(prof, self._prof_steps, self._prof_out, "sidecar")
        _logger.info("profile written: %s", out)

    def _mapper_unit(self):
        m, cfg = self.mapper, self.mapper_cfg
        with self.net_lock:
            adds, self.pending_add = self.pending_add, []
            adds = [(f, self.poses[f][0]) for f in dict.fromkeys(adds) if f in self.poses]
            registered = {f for f, (_, c) in self.poses.items() if c >= cfg["conf_threshold"]}
            busy = self.sched.has_work(adds, m.inlier_ema, m.frames, self.anchored, self.known, registered)
            self.sched.tick(idle=not busy)
            if not busy:
                return
            self.sched.dirty = False
        for f, w2c in adds:
            if not Path(f).exists():
                continue  # evicted meanwhile: nothing joined the map
            new = f not in m.slot_of
            if not m.add_frame(f, w2c, self.focal):
                _logger.warning("mapper buffer full: %s not added", f)
            elif new:
                with self.net_lock:
                    self.sched.note_joined(f)
        for _ in range(int(cfg["iters_per_tick"])):
            m.step()
        self.units += 1
        if self.units % int(cfg["publish_every"]) == 0:
            net = m.snapshot()
            with self.net_lock:
                self.network = net
                self.version += 1
            torch.cuda.empty_cache()
        if self.units % int(cfg["reloc_every"]):
            return
        if self.extent is None or self.units % 200 == 0:
            with self.net_lock:
                cen = [np.linalg.inv(w)[:3, 3] for w, c in self.poses.values() if c >= cfg["conf_threshold"]]
            if len(cen) >= 2:
                cen = np.stack(cen)
                self.extent = float(np.linalg.norm(cen.max(0) - cen.min(0))) or 1.0
        with self.net_lock:
            registered = {f for f, (_, c) in self.poses.items() if c >= cfg["conf_threshold"]}
            frame = self.sched.pick(self.known, registered, self.anchored)
            if frame is None:
                return
            net = self.network
        if not Path(frame).exists():
            return
        row = self.localize(frame, net)
        row["kind"] = "reloc"
        if row.get("w2c") is None or row["conf"] < cfg["conf_threshold"]:
            if frame not in m.slot_of:
                with self.net_lock:
                    self.sched.note_failure(frame)
            return  # nothing adopted: the trainer keeps what it has
        w2c = np.array(row["w2c"]).reshape(4, 4)
        if frame in m.slot_of:
            # A half-trained head relocalizes noisily, and an adopted pose is the next target:
            # adopt only while the map fits its frames (ACE-Zero relocalizes after a mapping
            # round has converged)
            if (m.inlier_ema or 0.0) < cfg["adopt_min_inliers"]:
                return
            with self.net_lock:
                old = self.poses.get(frame, (None, 0.0))[0]
            moved = (float(np.linalg.norm(np.linalg.inv(w2c)[:3, 3] - np.linalg.inv(old)[:3, 3]))
                     if old is not None else np.inf)
            if self.extent and moved < cfg["anchor_tol"] * self.extent:
                self.stable_count[frame] = self.stable_count.get(frame, 0) + 1
            else:
                self.stable_count[frame] = 0
            m.set_pose(frame, w2c)
            if self.stable_count[frame] >= int(cfg["anchor_after"]):
                self.anchored.add(frame)
                row["meta"] = {"anchored": True}
        elif not m.add_frame(frame, w2c, self.focal):  # registered late: joins the map
            _logger.warning("mapper buffer full: %s not added", frame)
        else:
            with self.net_lock:
                self.sched.note_joined(frame)
        with self.net_lock:
            self.poses[frame] = (w2c, row["conf"])
        self.emit([row])

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
        jobs = {"bootstrap": self.bootstrap, "refit": self.refit, "load": self.load, "seed": self.seed_map}
        for batch in commands:
            for msg in batch.to_pylist():
                cmd, frames, args = msg["cmd"], msg["frames"] or [], msg["args"] or []
                try:
                    params = parse_meta(msg["params"])
                    if cmd == "localize":
                        with self.net_lock:
                            network = self.network
                        row = self.localize(frames[0], network)
                        with self.net_lock:
                            if row.get("w2c") is not None:
                                self.poses[row["frame"]] = (np.array(row["w2c"]).reshape(4, 4), row["conf"])
                            if self.mapper is not None:
                                if row["frame"] not in self.known:
                                    self.known.append(row["frame"])
                                if row["conf"] >= self.mapper_cfg["conf_threshold"]:
                                    self.pending_add.append(row["frame"])
                                else:
                                    self.sched.note_failure(row["frame"])
                        self.emit([row])
                    elif cmd == "mapper":
                        self.start_mapper(frames, params)
                    elif cmd == "depth":
                        m = self.mapper
                        if m is not None and frames and Path(str(params.get("depth", ""))).exists():
                            m.depth_frac = float(params.get("depth_frac", m.depth_frac))
                            m.depth_weight = float(params.get("depth_weight", m.depth_weight))
                            with self.net_lock:
                                m.set_depth(frames[0], str(params["depth"]), self.focal)
                                self.sched.dirty = True
                    elif cmd == "rings":
                        self.open_rings(params)
                    elif cmd == "profile":
                        if self.mapper is None:
                            _logger.warning("profile: only continuous mapping is profiled")
                        self.prof_request = (int(params.get("units", 300)), params["out"])
                    elif cmd == "tick":
                        with self.tick_cv:
                            cap = int(self.mapper_cfg.get("max_backlog", 4))
                            self.tick_budget = min(cap, self.tick_budget + int(params.get("n", 1)))
                            self.tick_cv.notify()
                    elif cmd in jobs and self.mapper is not None and cmd != "load":
                        self.error(cmd, "continuous mapping is on: refits are not used")
                    elif cmd in jobs:
                        self.start_job(cmd, jobs[cmd], frames, args, params)
                    elif cmd == "forget":
                        with self.net_lock:
                            for fr in frames:
                                self.poses.pop(fr, None)
                                if fr in self.known:
                                    self.known.remove(fr)
                                self.anchored.discard(fr)
                                self.stable_count.pop(fr, None)
                                self.sched.forget(fr)
                                if self.mapper is not None:
                                    self.pending_add = [f for f in self.pending_add if f != fr]
                                    self.mapper.remove_frame(fr)
                    elif cmd == "shutdown":
                        self.stopping.set()
                        with self.tick_cv:
                            self.tick_cv.notify_all()
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
    ap.add_argument("--max_tries", type=int, default=16, help="re-draws of an invalid RANSAC hypothesis")
    ap.add_argument("--max_points", type=int, default=1500, help="Seed points returned per frame")
    args = ap.parse_args()
    out = IpcWriter(proto, EVENT_SCHEMA)
    sidecar = Sidecar(args.work_dir, args.encoder_path, args.hypotheses, args.threshold,
                      args.max_points, out, max_tries=args.max_tries)
    try:
        sidecar.serve(ipc_batches(sys.stdin.buffer))
    finally:
        sidecar.stopping.set()
        sidecar.terminate_child()  # stdin closed (client gone) counts as shutdown too
        out.close()


if __name__ == "__main__":
    main()
