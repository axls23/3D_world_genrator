"""StreamSession: camera / video / Flight frames -> posed keyframes the trainer can consume.

    source --Frame--> KeyframeGate --JPEG--> KeyframeStore  (bounded, in memory)
                                        \\--> ACE-res copy on tmpfs --> AceSidecar
    AceSidecar --pose/map events (Arrow)--> store poses + seed points (normalized)
    trainer <-- views() Arrow table, take_seeds(), transform / intrinsics

Lifecycle
  1. bootstrap   buffer ``bootstrap_frames`` keyframes (pinned), build the first ACE map with
                 hybrid ACE-Zero, localize them, freeze the world normalization from it
                 (same transform the COLMAP Parser uses, so all downstream code agrees).
  2. track       localize every new keyframe (~35 ms) against the current map; registered
                 ones contribute seed points for new Gaussians.
  3. grow        when frames stop registering (the camera left the mapped area) or enough
                 new frames registered, refit the map in the background, warm-started and
                 supervised by the current poses so the frame stays fixed; then every frame
                 is re-localized (poses refine, failed frames may now register).
  4. finish      a file source ends -> one last refit if frames are still unregistered ->
                 ``finished``. Live sources run until stop_recording() (dashboard button,
                 ``record_stop_file``, ``record_seconds`` / ``record_keyframes`` limits): the
                 camera is released and training continues on the keyframes collected, exactly
                 as if a file had ended. stop() tears everything down.

GPU: the trainer's online Difix rounds and ACE map refits take turns (``gpu_lease``); a
refit that finds the lease taken is deferred until ``release_gpu`` and vice versa.

Nothing is written outside the tmpfs work dir (ACE-res frames + the live ACE head), and
evicted keyframes are removed from it and from the sidecar immediately.
"""

import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow as pa

from .gauge import Sim3, fit_sim3
from .keyframes import KeyframeGate
from .schema import VIEW_SCHEMA
from .sidecar import AceSidecar
from .source import FrameSource, QueueFrameSource, decode_jpeg, encode_jpeg, fit_max_side, parse_source
from .store import Keyframe, KeyframeStore, PseudoView

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@dataclass
class StreamConfig:
    source: str
    pace: str = "auto"                  # auto | realtime | fast (see source.py)
    max_side: int = 960                 # train-resolution long side
    ace_short_side: int = 480           # ACE works at this short side
    jpeg_quality: int = 90
    budget_mb: float = 256.0            # encoded keyframes + ACE copies + pseudo views
    max_keyframes: int = 400
    test_every: int = 8                 # every Nth keyframe is held out for eval
    min_motion: float = 8.0             # keyframe gate novelty (0-255 mean abs diff)
    bootstrap_frames: int = 24
    bootstrap_retry_frames: int = 8     # add this many keyframes before retrying a failed bootstrap
    bootstrap_iterations: Optional[int] = 5000  # cap on ACE training for the first map (latency);
                                                # refits keep improving it in the background
    bootstrap_cooldown: Optional[int] = None    # cap on the first map's LR cooldown iterations
                                                # (after the inlier trigger); refits unaffected
    quality_mode: str = "fast"          # ACE schedule preset
    refit_min_unregistered: int = 6     # refit once this many new frames failed to register
    refit_every_registered: int = 40    # ...or this many new frames registered
    refit_iterations: int = 4000
    # ACE-Zero growth: a refit that registered at least this many more keyframes while others
    # are still unregistered schedules the next one (the extended map may reach them), also
    # after the source ended; the chain stops at the first refit that gains fewer
    refit_min_gain: int = 2
    # Refit training recipe. "legacy": train_ace defaults (circle 1cycle peaking at 5e-3, fixed
    # iterations, randomly initialized pose MLP). "ace0": a mapping iteration as in the ACE-Zero
    # paper - 1cyclepoly warmup 5e-4 -> 3e-3, cooldown to 5e-4 as soon as 70% of reprojection
    # errors are under 10 px, pose MLP starting at the input poses (zero-init output layer)
    refit_recipe: str = "legacy"
    # Align every refit to the previous map by Sim(3) before its poses reach the trainer. ACE-Zero
    # reconstructs up to a similarity and nothing pins it between refits; the trainer's frame is
    # frozen at bootstrap, so without the lock gauge drift moves cameras under the Gaussians
    gauge_lock: bool = False
    # "refit": the map grows by whole ACE refits (train_ace jobs, ~2 min each, landing as one
    # pose update). "continuous": after the first map the sidecar keeps training the head a few
    # iterations per trainer step and relocalizes keyframes round-robin (ACE-Zero's
    # mapping / relocalization loop, interleaved with 3DGS training; scripts/acezero/stream_mapper.py)
    ace_mode: str = "refit"
    mapper_iters_per_tick: int = 4      # head iterations per trainer step
    mapper_reloc_every: int = 2         # trainer steps per round-robin relocalization
    mapper_adopt_min_inliers: float = 0.5   # adopt relocalized poses only while the head fits this well
    mapper_anchor_tol: float = 0.01     # a frame that moved less (x path extent) ...
    mapper_anchor_after: int = 2        # ... this many relocalizations in a row is anchored
    mapper_publish_every: int = 25      # ticks between network snapshots used for localization
    mapper_max_backlog: int = 4         # ticks the sidecar may lag behind the trainer
    mapper_lr: float = 5e-4             # ACE-Zero's cooldown LR: the head is always warm
    mapper_passes: int = 2              # augmented encodings per frame in the patch buffer
    mapper_samples_per_pass: int = 1024
    max_inflight: int = 4               # localize requests in flight (fast pace back-pressure)
    seed_points_per_frame: int = 1500
    flight_port: Optional[int] = None   # also serve views/stats over Arrow Flight
    work_dir: Optional[str] = None      # default /dev/shm/hypersplat-stream-<pid>
    # Stop recording (release the camera, keep training) when this file is created after the
    # start, after this many seconds of frames, or once this many keyframes were accepted
    record_stop_file: Optional[str] = None
    record_seconds: Optional[float] = None
    record_keyframes: Optional[int] = None
    # Dense depth (hypersplat/stream/depth.py): queue registered keyframes with their ACE
    # points for the trainer's depth worker, and keep the aligned maps (half res, fp16)
    depth: bool = False
    # File sources: keep the bootstrap ACE map here and load it on later runs of the same
    # capture instead of re-bootstrapping (A/B runs on one map; ~65-90 s saved per run).
    # Separate bootstraps of one capture differed by 0.55 dB, more than any knob tested.
    map_cache: Optional[str] = None
    # Train ACE maps (bootstrap and refits) with deterministic cuDNN kernels: the autotuner
    # picks algorithms by timing, so a map built while 3DGS shares the GPU varies run to run
    ace_deterministic: bool = False


def normalization_transform(c2ws: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The COLMAP Parser's world normalization (datasets/colmap.py), from ACE cameras/points."""
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    from datasets.normalize import (align_principal_axes, similarity_from_cameras,
                                    transform_cameras, transform_points)

    T1 = similarity_from_cameras(c2ws)
    pts = transform_points(T1, points)
    T2 = align_principal_axes(pts)
    pts = transform_points(T2, pts)
    T = T2 @ T1
    if np.median(pts[:, 2]) > np.mean(pts[:, 2]):  # up-side-down fix, as in the Parser
        T = np.diag([1.0, -1.0, -1.0, 1.0]) @ T
    return T


def confidence_weight_fn(confs: List[float]):
    """ACE confidence -> loss weight: the same ramp augment.py applies to batch runs."""
    from hypersplat.pipeline.params.strategies.difix import confidence_ramp

    if not confs:
        return lambda c: 1.0
    min_conf, floor = confidence_ramp(confs)
    median = float(np.median(confs))
    span = max(median - min_conf, 1e-6)
    return lambda c: float(np.clip((c - min_conf) / span, floor, 1.0))


class GpuLease:
    """One heavy GPU job at a time between the trainer (online Difix rounds) and the ACE
    sidecar (map refits, a separate process): together they OOM a small GPU and one of them
    dies. Training steps and ~35 ms localizations are not leased. Non-blocking: a holder that
    cannot get it defers its job. A plain Lock, so the thread that releases (the sidecar reader,
    on the refit's map event) need not be the one that acquired."""

    def __init__(self):
        self._lock = threading.Lock()
        self.holder: Optional[str] = None

    def try_acquire(self, holder: str) -> bool:
        if not self._lock.acquire(blocking=False):
            return self.holder == holder
        self.holder = holder
        return True

    def release(self, holder: str):
        if self.holder == holder:
            self.holder = None
            self._lock.release()


class StreamSession:
    def __init__(self, cfg: StreamConfig, profile=None):
        self.cfg = cfg
        self.profile = profile
        kind, _ = parse_source(cfg.source)
        self.kind = kind
        self.work_dir = Path(cfg.work_dir or f"/dev/shm/hypersplat-stream-{os.getpid()}")
        self.frames_dir = self.work_dir / "frames"
        self.store = KeyframeStore(cfg.budget_mb, cfg.max_keyframes, cfg.test_every)
        self.gate = KeyframeGate(min_motion=cfg.min_motion)
        self.source = None
        self.sidecar: Optional[AceSidecar] = None
        self.flight_server = None

        self._cond = threading.Condition()
        self._stop = threading.Event()
        self.map_ready = threading.Event()
        self.failed: Optional[str] = None
        self.source_ended = False
        self.recording_stopped: Optional[str] = None   # why recording was stopped early
        self._record_since = time.time() - 1.0         # stale record_stop_file guard (mtime)
        self._record_t0: Optional[float] = None        # first frame (record_seconds clock)
        self.map_version = 0
        self.focal_file = -1.0              # ACE focal in px of the ACE-res files
        self.file_size: Optional[Tuple[int, int]] = None   # ACE-res file (w, h)
        self.train_size: Optional[Tuple[int, int]] = None  # train frame (w, h)
        self.transform: Optional[np.ndarray] = None
        self.scene_scale = 1.0
        self.init_points = np.zeros((0, 3), np.float32)
        self.init_rgbs = np.zeros((0, 3), np.float32)
        self.ace_values: Dict = {}

        self._job: Optional[str] = None     # bootstrap | refit in the sidecar
        self._bootstrap_target = cfg.bootstrap_frames
        self._inflight = 0
        self._new_failed = 0                # frames that failed to register since the last refit
        self._new_registered = 0
        self._grow_pending = False          # last refit gained frames and some are still unregistered
        self._refit_deferred = False        # a refit is due but the GPU lease is taken
        self.gpu_lease = GpuLease()
        self.views_registered = 0           # frames that became registered, ever (monotonic)
        self._seeds: List[Tuple[np.ndarray, np.ndarray]] = []
        self._depth_jobs: List[Tuple[int, np.ndarray, np.ndarray]] = []   # (kid, pix, camera z)
        self._depth: Dict[int, np.ndarray] = {}   # kid -> half-res depth, trainer units, 0 = invalid
        self.depth_counters = {"maps": 0, "rejected": 0, "ms": 0.0, "rel_err": None, "last_reject": None,
                               "realigned": 0, "anchored": 0}
        # Re-alignment after ACE refits: cached sharpened disparity (half res, fp16), the latest
        # ACE points per keyframe and the map version each depth map was aligned against
        self._disp: Dict[int, np.ndarray] = {}
        self._ace_pts: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}   # kid -> (pix, z, version)
        self._aligned_version: Dict[int, int] = {}
        self._refit_requested = 0
        # Gauge lock: map version -> Sim(3) from that map's frame into the bootstrap frame, the
        # per-refit alignment diagnostics, and live poses that beat their map event here
        self._gauges: Dict[int, Sim3] = {}
        self.refit_log: List[Dict] = []
        self.refit_pairs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}   # version -> (new, ref) w2c
        self._early_rows: List[Dict] = []
        self.reloc_moves: List[Tuple[float, int, float]] = []   # (time, kid, camera move) per reloc
        self._views_cache: Tuple[int, Optional[pa.Table]] = (-1, None)
        self._views_lock = threading.Lock()
        self._ingest: Optional[threading.Thread] = None
        self.stats_counters = {"refits": 0, "bootstrap_s": None, "last_refit_s": None,
                               "localized": 0, "localize_ms": 0.0}
        self._loc_sent: Dict[str, float] = {}
        # Latest source frame for live previews (dashboard): (image, t, accepted, sequence)
        self._latest: Optional[Tuple[np.ndarray, float, bool, int]] = None
        self._latest_seq = 0
        self._last_accept_time = 0.0

    # ================================================================ lifecycle
    def start(self):
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.kind == "flight":
            from .flight import FrameFlightServer, flight_location
            self.source = QueueFrameSource(self.cfg.source, self.cfg.max_side)
            self.flight_server = FrameFlightServer(flight_location(self.cfg.source), frames=self.source,
                                                   views=self.views_table, stats=self.stats)
            self.flight_server.serve_in_background()
            logger.info(f"[Stream] Waiting for frames on {flight_location(self.cfg.source)} (Arrow Flight do_put)")
        else:
            self.source = FrameSource(self.cfg.source, self.cfg.max_side, self.cfg.pace)
            if self.cfg.flight_port:
                from .flight import FrameFlightServer
                self.flight_server = FrameFlightServer(f"grpc://0.0.0.0:{self.cfg.flight_port}",
                                                       views=self.views_table, stats=self.stats)
                self.flight_server.serve_in_background()
        self.sidecar = AceSidecar(self.work_dir / "ace", self._on_event,
                                  max_points=self.cfg.seed_points_per_frame)
        if not self.sidecar.ready.wait(120) or not self.sidecar.alive:
            raise RuntimeError("ACE sidecar failed to start (see [ace-sidecar] log lines)")
        self._ingest = threading.Thread(target=self._ingest_loop, name="stream-ingest", daemon=True)
        self._ingest.start()
        logger.info(f"[Stream] {self.kind} source {self.cfg.source} (pace {self.source.pace}), "
                    f"budget {self.cfg.budget_mb:.0f} MB / {self.cfg.max_keyframes} keyframes, "
                    f"work dir {self.work_dir}")

    def wait_for_map(self, timeout: Optional[float] = None, should_stop=None) -> bool:
        """Block until the first map is ready (True) or the stream failed / timed out /
        should_stop() returned True."""
        end = None if timeout is None else time.time() + timeout
        while not self.map_ready.is_set():
            if self.failed or self._stop.is_set() or (should_stop is not None and should_stop()):
                return False
            if end is not None and time.time() > end:
                return False
            self.map_ready.wait(1.0)
        return True

    def stop(self):
        self._stop.set()
        if self.source is not None:
            self.source.close()
        if self._ingest is not None:
            self._ingest.join(timeout=5)
        if self.sidecar is not None:
            self.sidecar.close()
        if self.flight_server is not None:
            try:
                self.flight_server.shutdown()
            except Exception:
                pass
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def stop_recording(self, reason: str = "requested") -> bool:
        """Stop taking frames from the source (the camera is released) while the map and the
        trainer keep working on the keyframes collected; the session then ends like a file
        source. False if the source had already ended."""
        with self._cond:
            if self.source_ended or self.recording_stopped:
                return False
            self.recording_stopped = reason
        logger.info(f"[Stream] Recording stopped ({reason}): {len(self.store)} keyframes kept, "
                    f"training continues on them")
        if self.source is not None and self.source.live:
            self.source.close()  # frees the device now; the ingest loop sees the end
        return True

    def _record_limit(self) -> Optional[str]:
        """Reason to stop recording now, if a stop was requested or a limit reached."""
        c = self.cfg
        if c.record_keyframes and self.gate.accepted >= c.record_keyframes:
            return f"{c.record_keyframes} keyframes recorded"
        if c.record_seconds and self._record_t0 is not None and time.time() - self._record_t0 >= c.record_seconds:
            return f"{c.record_seconds:g} s recorded"
        if c.record_stop_file:
            try:
                if os.stat(c.record_stop_file).st_mtime >= self._record_since:
                    return "stop file"
            except OSError:
                pass
        return None

    @property
    def finished(self) -> bool:
        """File source fully consumed, nothing waiting on the sidecar, no map job running."""
        with self._cond:
            return (self.source_ended and self.map_ready.is_set() and self._job is None
                    and self._inflight == 0 and not self._refit_deferred)

    # ================================================================ ingest
    def _ingest_loop(self):
        try:
            for frame in self.source:
                if self._stop.is_set() or self.recording_stopped:
                    break
                self._record_t0 = self._record_t0 or time.time()
                reason = self._record_limit()
                if reason:
                    self.stop_recording(reason)
                    break
                res = self.gate.offer(frame.image, frame.t)
                self._latest_seq += 1
                self._latest = (frame.image, frame.t, res.accept, self._latest_seq)
                if not res.accept:
                    continue
                self._last_accept_time = time.time()
                self._add_keyframe(frame.image, frame.t, res.sharpness)
        except Exception as e:
            logger.exception("[Stream] ingest failed")
            self.failed = f"ingest: {e}"
        finally:
            # One hold: ``finished`` must not read True between the end and the final refit
            with self._cond:
                self.source_ended = True
                self._maybe_final_refit()
                self._cond.notify_all()
            logger.info(f"[Stream] Source ended: {self.source.frames_read} frames read, "
                        f"{self.gate.accepted} keyframes")
            if not self.map_ready.is_set() and self._job is None and not self.failed:
                what = "recording stopped" if self.recording_stopped else "source ended"
                self.failed = (f"{what} with {len(self.store)} keyframes and no map "
                               f"(bootstrap needs {self._bootstrap_target})")

    def _add_keyframe(self, image: np.ndarray, t: float, sharp: float):
        h, w = image.shape[:2]
        self.train_size = self.train_size or (w, h)
        kf = self.store.add(encode_jpeg(image, self.cfg.jpeg_quality), w, h, t, sharp)
        # ACE copy: short side ace_short_side (never upscaled)
        scale = min(1.0, self.cfg.ace_short_side / min(h, w))
        ace_img = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else image
        self.file_size = self.file_size or (ace_img.shape[1], ace_img.shape[0])
        path = self.frames_dir / f"{kf.name}.jpg"
        data = encode_jpeg(ace_img, 92)
        path.write_bytes(data)
        self.store.set_ace_file(kf.kid, str(path), len(data))
        if not self.map_ready.is_set():
            kf.pinned = True  # bootstrap candidates are never evicted before the map exists
        self._evict()

        with self._cond:
            if self.map_ready.is_set():
                # Back-pressure: files wait for the sidecar; live sources drop frames upstream
                while self._inflight >= self.cfg.max_inflight and not self._stop.is_set():
                    self._cond.wait(0.5)
                self._send_localize(kf)
            elif self._job is None and len(self.store) >= self._bootstrap_target:
                self._start_bootstrap()
            elif self._job == "bootstrap" and self.source.pace == "fast":
                # Just-in-time mode: no point reading ahead of the map
                while self._job == "bootstrap" and not self._stop.is_set():
                    self._cond.wait(0.5)

    def _send_localize(self, kf: Keyframe):
        self._inflight += 1
        self._loc_sent[kf.ace_path] = time.time()
        self.sidecar.localize(kf.ace_path)

    def _evict(self):
        evicted = self.store.evict_if_needed()
        if not evicted:
            return
        paths = [kf.ace_path for kf in evicted if kf.ace_path]
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        self.sidecar.forget(paths)

    # ================================================================ ACE map jobs
    def _ace_schedule(self, n_frames: int) -> Dict:
        from hypersplat.pipeline.params.strategies import ace_schedule as sched

        w, h = self.file_size
        decisions = sched.compute_schedule(self.gate.avg_motion, n_frames, w, h, self.cfg.quality_mode,
                                           os.cpu_count(), None)
        values = {k: d.value for k, d in decisions.items()}
        # The GPU is shared with 3DGS training: keep ACE's training buffer in RAM and its
        # data loading light
        values.update(training_buffer_cpu=True, num_data_workers=min(4, int(values["num_data_workers"])),
                      refinement="mlp", refinement_ortho="gram-schmidt", pose_refinement_lr=0.001)
        cap = self.cfg.bootstrap_iterations
        if cap and values["hybrid_train_iterations"] > cap:
            ratio = cap / float(values["hybrid_train_iterations"])
            values["hybrid_train_iterations"] = int(cap)
            values["hybrid_pose_wait"] = int(values["hybrid_pose_wait"] * ratio)
            values["cooldown_iterations"] = int(values["cooldown_iterations"] * ratio)
        if self.cfg.bootstrap_cooldown and values["cooldown_iterations"] > self.cfg.bootstrap_cooldown:
            values["cooldown_iterations"] = int(self.cfg.bootstrap_cooldown)
        return values

    def _map_cache(self) -> Optional[Tuple[Path, Dict, bool]]:
        """(cache dir, meta, usable) for a file source with stream map caching on, else None.
        A cache made from a different file (path, size or mtime) is not usable."""
        if not self.cfg.map_cache or self.kind != "file":
            return None
        # Absolute: the ACE sidecar runs in scripts/acezero, where a relative path misses
        cache = Path(self.cfg.map_cache).resolve()
        try:
            st = os.stat(self.cfg.source)
            key = {"source": str(Path(self.cfg.source).resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}
        except OSError:
            return None
        meta = {}
        try:
            meta = json.loads((cache / "meta.json").read_text())
        except (OSError, ValueError):
            pass
        usable = ((cache / "head.pt").exists() and (cache / "poses.txt").exists()
                  and all(meta.get(k) == v for k, v in key.items()) and not getattr(self, "_cache_failed", False))
        return cache, {**meta, **key}, usable

    def _ace_extra_args(self) -> List[str]:
        """Options shared by the bootstrap (ace_zero_hybrid.py) and refits (train_ace.py)."""
        return ["--deterministic", "True"] if self.cfg.ace_deterministic else []

    def _start_bootstrap(self):
        frames = [kf.ace_path for kf in self.store.frames() if kf.ace_path][: self._bootstrap_target]
        self._bootstrap_attempted = len(frames)
        self.ace_values = self._ace_schedule(len(frames))
        from hypersplat.pipeline.params.strategies.ace_schedule import build_ace_args
        self.store.conf_threshold = float(self.ace_values["registration_confidence"])
        self._job = "bootstrap"
        self._job_t0 = time.time()
        cached = self._map_cache()
        if cached is not None and cached[2]:
            cache, meta, _ = cached
            logger.info(f"[Stream] Loading the cached ACE map {cache} for {len(frames)} keyframes")
            self.sidecar.load(str(cache / "head.pt"), str(cache / "poses.txt"), frames,
                              image_resolution=int(meta.get("image_resolution", self.cfg.ace_short_side)))
            return
        cache_dir = ""
        if cached is not None:
            cache_dir = str(cached[0])
            cached[0].mkdir(parents=True, exist_ok=True)
            (cached[0] / "meta.json").write_text(json.dumps({k: cached[1][k] for k in ("source", "size", "mtime")}))
        logger.info(f"[Stream] Bootstrapping ACE map from {len(frames)} keyframes "
                    f"(avg motion {self.gate.avg_motion or 0:.1f}, "
                    f"{self.ace_values['hybrid_train_iterations']} iterations, "
                    f"cooldown {self.ace_values['cooldown_iterations']})")
        args = build_ace_args(self.ace_values, hybrid=True) + self._ace_extra_args()
        self.sidecar.bootstrap(frames, args, cache_dir=cache_dir)

    def _maybe_refit(self, force: bool = False):
        """Caller holds _cond."""
        if self._job is not None or not self.map_ready.is_set() or self.cfg.ace_mode == "continuous":
            return
        if not force and not (self._new_failed >= self.cfg.refit_min_unregistered
                              or self._new_registered >= self.cfg.refit_every_registered
                              or self._grow_pending):
            return
        frames = [kf.ace_path for kf in self.store.frames() if kf.ace_path and kf.w2c is not None]
        if len(self.store.registered()) < 3:
            return
        if not self.gpu_lease.try_acquire("ace"):
            # A Difix round holds the GPU; release_gpu() retries (the counters are kept)
            self._refit_deferred = True
            return
        self._refit_deferred = False
        v = self.ace_values
        args = ["--training_buffer_cpu", "True", "--num_head_blocks", str(v["num_head_blocks"]),
                "--num_data_workers", str(v["num_data_workers"]), "--pose_refinement", "mlp",
                "--pose_refinement_wait", "0", "--refine_calibration", "False",
                "--aug_rotation", str(v["aug_rotation"]), "--repro_loss_soft_clamp", str(v["repro_loss_soft_clamp"])]
        if self.cfg.refit_recipe == "ace0":
            # ACE-Zero's mapping iteration (paper Sec. 3.3 / supp.), scaled to the refit budget:
            # warm start from the live head, short warmup, early cooldown on the inlier trigger
            it = int(self.cfg.refit_iterations)
            args += ["--learning_rate_schedule", "1cyclepoly", "--learning_rate_max", "0.003",
                     "--learning_rate_min", "0.0005", "--learning_rate_warmup_learning_rate", "0.0005",
                     "--learning_rate_warmup_iterations", str(max(100, it // 8)),
                     "--learning_rate_cooldown_iterations", str(max(500, it * 3 // 8)),
                     "--learning_rate_cooldown_trigger_px_threshold", "10",
                     "--learning_rate_cooldown_trigger_percent_threshold", "0.7",
                     "--pose_refinement_zero_init", "True"]
        elif self.cfg.refit_recipe != "legacy":
            raise ValueError(f"unknown refit_recipe {self.cfg.refit_recipe!r}")
        args += self._ace_extra_args()
        logger.info(f"[Stream] Refitting ACE map on {len(frames)} keyframes "
                    f"({self._new_failed} unregistered, {self._new_registered} new registered)")
        self._job = "refit"
        self._job_t0 = time.time()
        self._new_failed = self._new_registered = 0
        self._grow_pending = False
        self.sidecar.refit(frames, self.cfg.refit_iterations, self.store.conf_threshold, args)

    def _maybe_final_refit(self):
        with self._cond:
            if self.map_ready.is_set() and self.store.unregistered() and (self._new_failed > 0
                                                                        or self._grow_pending):
                self._maybe_refit(force=True)

    def release_gpu(self, holder: str):
        """Give the GPU lease back and start a refit that waited for it."""
        with self._cond:
            self.gpu_lease.release(holder)
            if self._refit_deferred:
                self._refit_deferred = False
                self._maybe_refit()
                if self.source_ended:
                    self._maybe_final_refit()
            self._cond.notify_all()

    # ================================================================ sidecar events
    def _on_event(self, kind: str, rows: List[Dict]):
        if kind == "pose":
            self._on_pose(rows[0], from_map=False)
        elif kind == "map":
            self._on_map(rows)
        elif kind == "reloc":
            self._on_reloc(rows)
        elif kind == "error":
            self._on_error(rows[0]["meta"])
        elif kind == "exit":
            if not self._stop.is_set():
                self.failed = f"ACE sidecar exited ({rows[0]['meta'].get('returncode')})"
                logger.error(f"[Stream] {self.failed}")
            with self._cond:
                self.gpu_lease.release("ace")  # its refit died with it: don't block Difix
                self._cond.notify_all()

    def _to_bootstrap_frame(self, row: Dict) -> Dict:
        S = self._gauges.get(row["version"])
        if S is None or row.get("w2c") is None:
            return row
        row = dict(row)
        row["w2c"] = S.w2c(row["w2c"])
        if len(row["points"]):
            row["points"] = S.points(row["points"]).astype(np.float32)
        return row

    def _on_pose(self, row: Dict, from_map: bool) -> Optional[Keyframe]:
        if self.cfg.gauge_lock and self.cfg.ace_mode == "refit":
            if not from_map and row["version"] > self.map_version:
                # Localized by a map whose event (and so its alignment) has not arrived yet
                self._early_rows.append(row)
                return None
            row = self._to_bootstrap_frame(row)
        kf = self.store.by_ace_path().get(row["frame"])
        registered = was_registered = False
        if kf is not None:  # else evicted meanwhile
            was_registered = self.store.is_registered(kf)
            registered = self.store.update_pose(kf.kid, row["w2c"], row["conf"], row["version"])
            if registered and not kf.seeded and self.transform is not None:
                self._queue_seeds(kf, row["points"], row["pix"])
            elif registered and self.cfg.depth and len(row["points"]) and kf.w2c is not None:
                z = np.asarray(row["points"], np.float64) @ kf.w2c[2, :3] + kf.w2c[2, 3]
                self._ace_pts[kf.kid] = (np.asarray(row["pix"], np.float32), z, int(row["version"]))
        # One hold for the in-flight count and the refit decision: ``finished`` must not read
        # True in between (no localize in flight, final refit not started yet)
        with self._cond:
            if registered and not was_registered:
                self.views_registered += 1
            if not from_map:
                self._inflight = max(0, self._inflight - 1)
                sent = self._loc_sent.pop(row["frame"], None)
                if sent is not None:
                    n = self.stats_counters["localized"]
                    ms = (time.time() - sent) * 1000
                    self.stats_counters["localize_ms"] = (self.stats_counters["localize_ms"] * n + ms) / (n + 1)
                    self.stats_counters["localized"] = n + 1
                if kf is not None:
                    if registered and not was_registered:
                        self._new_registered += 1
                    elif not registered:
                        self._new_failed += 1
                    self._maybe_refit()
            self._cond.notify_all()
        return kf

    def _fit_gauge(self, rows: List[Dict], meta: Dict):
        """Align a refit to the poses the trainer has now (bootstrap frame) and log how far the
        refit really moved the cameras. Measured always; applied with ``gauge_lock``."""
        version = rows[0]["version"]
        by_path = self.store.by_ace_path()
        new, ref = [], []
        for row in rows:
            kf = by_path.get(row["frame"]) if row["frame"] else None
            if (kf is None or row.get("w2c") is None or row["conf"] < self.store.conf_threshold
                    or not self.store.is_registered(kf)):
                continue
            new.append(row["w2c"])
            ref.append(kf.w2c)
        if len(new):
            self.refit_pairs[version] = (np.stack(new), np.stack(ref))
        prev = self._gauges.get(max(self._gauges)) if self._gauges else Sim3()
        r = fit_sim3(np.stack(new), np.stack(ref), candidates=(Sim3(), prev)) if len(new) >= 3 else None
        if r is None or r.inliers < 5:
            logger.warning(f"[Stream] Refit v{version}: {len(new)} frames registered in both maps, "
                           f"too few to align; keeping the previous gauge")
            if self.cfg.gauge_lock:
                self._gauges[version] = prev
            return
        cref = np.stack([np.linalg.inv(w)[:3, 3] for w in ref])
        extent = float(np.linalg.norm(cref.max(0) - cref.min(0))) or 1.0
        entry = {"version": version, "seconds": meta.get("seconds"), "common": r.used,
                 "inliers": r.inliers, "scale": r.sim3.s, "rot_deg": r.sim3.angle_deg,
                 "shift": float(np.linalg.norm(r.sim3.t)) / extent,
                 "res_median": r.center_median / extent, "res_p90": r.center_p90 / extent,
                 "res_max": r.center_max / extent, "rot_res_median_deg": r.rot_median_deg,
                 "rot_res_max_deg": r.rot_max_deg, "raw_median": r.raw_center_median / extent,
                 "raw_max": r.raw_center_max / extent, "locked": self.cfg.gauge_lock}
        # Over all shared frames, not just the alignment inliers (the trim is relative to the
        # median, so with a tight median it also drops frames that barely moved)
        rel = r.all_res / extent
        entry.update(all_median=float(np.median(rel)), all_p90=float(np.percentile(rel, 90)),
                     all_max=float(rel.max()), moved_1pct=int((rel > 0.01).sum()),
                     moved_5pct=int((rel > 0.05).sum()))
        self.refit_log.append(entry)
        logger.info(f"[Stream] Refit v{version} gauge: scale {r.sim3.s:.3f}, rot {r.sim3.angle_deg:.2f} deg, "
                    f"shift {entry['shift']:.1%} of path; after alignment cameras moved median "
                    f"{entry['res_median']:.2%} / p90 {entry['res_p90']:.2%} / max {entry['res_max']:.2%} "
                    f"of path, {r.rot_median_deg:.2f} deg (raw median {entry['raw_median']:.2%}, "
                    f"max {entry['raw_max']:.2%}); {r.inliers}/{r.used} frames; all frames aligned "
                    f"median {entry['all_median']:.2%} p90 {entry['all_p90']:.2%} max {entry['all_max']:.2%}, "
                    f"{entry['moved_1pct']} moved >1%, {entry['moved_5pct']} >5% of path"
                    + ("" if self.cfg.gauge_lock else " [not locked]"))
        if self.cfg.gauge_lock:
            self._gauges[version] = r.sim3

    def _on_reloc(self, rows: List[Dict]):
        """Round-robin relocalizations of the continuous mapper: pose updates (and late
        registrations) one frame at a time. Records how far registered cameras move."""
        by_path = self.store.by_ace_path()
        for row in rows:
            kf = by_path.get(row["frame"])
            old = kf.w2c if kf is not None and self.store.is_registered(kf) else None
            if row.get("version", 0) > self.map_version:
                self.map_version = row["version"]
            self._on_pose(row, from_map=True)
            if (old is not None and row.get("w2c") is not None
                    and row["conf"] >= self.store.conf_threshold):
                moved = float(np.linalg.norm(np.linalg.inv(row["w2c"])[:3, 3] - np.linalg.inv(old)[:3, 3]))
                self.reloc_moves.append((time.time(), kf.kid, moved))
            self.stats_counters["relocs"] = self.stats_counters.get("relocs", 0) + 1
            if (row.get("meta") or {}).get("anchored"):
                self.stats_counters["anchored"] = self.stats_counters.get("anchored", 0) + 1

    def tick(self, n: int = 1) -> bool:
        """One trainer step's worth of continuous mapping. Skipped while another job (online
        Difix) holds the GPU lease, or before the first map."""
        if (self.cfg.ace_mode != "continuous" or not self.map_ready.is_set() or self.sidecar is None
                or not self.sidecar.alive or self.gpu_lease.holder not in (None, "ace")):
            return False
        try:
            self.sidecar.tick(n)
        except RuntimeError:
            return False
        self.stats_counters["mapper_ticks"] = self.stats_counters.get("mapper_ticks", 0) + n
        return True

    def _on_map(self, rows: List[Dict]):
        meta = rows[0]["meta"]
        self.map_version = rows[0]["version"]
        if rows[0]["focal"] and rows[0]["focal"] > 0:
            self.focal_file = float(rows[0]["focal"])
        first = not self.map_ready.is_set()
        if first:
            self._freeze_normalization(rows)
        n_before = len(self.store.registered())
        if meta.get("reason") == "refit":
            self._fit_gauge(rows, meta)
        for row in rows:
            if row["frame"]:
                self._on_pose(row, from_map=True)
        early, self._early_rows = self._early_rows, []
        for row in early:
            self._on_pose(row, from_map=False)
        n_reg = len(self.store.registered())
        logger.info(f"[Stream] ACE map v{self.map_version} ({meta.get('reason')}, {meta.get('seconds')} s): "
                    f"{n_reg}/{len(self.store)} keyframes registered, focal {self.focal_file:.1f} px")
        with self._cond:
            if meta.get("reason") in ("bootstrap", "load"):
                self.stats_counters["bootstrap_s"] = meta.get("seconds")
            elif meta.get("reason") == "refit":
                self.stats_counters["refits"] += 1
                self.stats_counters["last_refit_s"] = meta.get("seconds")
            self._job = None
            self.gpu_lease.release("ace")
            if meta.get("reason") == "refit":
                # The map event's poses are not live localizations, so they never reached the
                # refit counters: once the source ended (no more localize calls) a refit that
                # just registered more frames was the last one, leaving the rest unregistered
                gain = n_reg - n_before
                self._new_registered += max(0, gain)
                self._grow_pending = gain >= self.cfg.refit_min_gain and bool(self.store.unregistered())
                if self._grow_pending:
                    logger.info(f"[Stream] Refit registered {gain} more keyframes, "
                                f"{len(self.store.unregistered())} still unregistered: refitting again")
                    self._maybe_refit()
            if first:
                for kf in self.store.frames():
                    kf.pinned = False
                self.store.map_ready_at = time.time()
                self.map_ready.set()
                # Keyframes that arrived during the bootstrap
                for kf in self.store.pending():
                    self._send_localize(kf)
                if self.cfg.ace_mode == "continuous":
                    c = self.cfg
                    self.sidecar.mapper(
                        [kf.ace_path for kf in self.store.frames() if kf.ace_path],
                        conf_threshold=self.store.conf_threshold, iters_per_tick=c.mapper_iters_per_tick,
                        reloc_every=c.mapper_reloc_every, publish_every=c.mapper_publish_every,
                        adopt_min_inliers=c.mapper_adopt_min_inliers, anchor_tol=c.mapper_anchor_tol,
                        anchor_after=c.mapper_anchor_after,
                        max_backlog=c.mapper_max_backlog, lr=c.mapper_lr, passes=c.mapper_passes,
                        samples_per_pass=c.mapper_samples_per_pass, max_frames=c.max_keyframes)
                    logger.info("[Stream] Continuous ACE mapping on: the head trains in step with 3DGS")
            self._cond.notify_all()
        if self.profile is not None:
            self.profile.set("stream.map_version", self.map_version, "stream")
            self.profile.set("stream.registered", n_reg, "stream")
        if self.source_ended:
            self._maybe_final_refit()

    def _on_error(self, meta: Dict):
        cmd, msg = meta.get("cmd"), meta.get("msg")
        logger.warning(f"[Stream] ACE sidecar {cmd} error: {msg}")
        with self._cond:
            if cmd == "load":  # unusable cache: bootstrap from scratch (and rewrite the cache)
                self._job = None
                self._cache_failed = True
                self._start_bootstrap()
            elif cmd == "bootstrap":
                self._job = None
                self._bootstrap_target += self.cfg.bootstrap_retry_frames
                if not self.source_ended:
                    logger.info(f"[Stream] Retrying the bootstrap once {self._bootstrap_target} keyframes are in")
                elif len(self.store) > getattr(self, "_bootstrap_attempted", 0):
                    # No more frames are coming: retry with every keyframe we have
                    self._bootstrap_target = len(self.store)
                    logger.info(f"[Stream] Source ended; retrying the bootstrap with all {len(self.store)} keyframes")
                    self._start_bootstrap()
                else:
                    self.failed = f"bootstrap failed: {msg}"
            elif cmd == "refit":
                self._job = None
                self.gpu_lease.release("ace")
            elif cmd == "localize":
                self._inflight = max(0, self._inflight - 1)
            self._cond.notify_all()

    def _freeze_normalization(self, rows: List[Dict]):
        good = [r for r in rows if r["w2c"] is not None and r["conf"] >= self.store.conf_threshold]
        if len(good) < 2:
            good = [r for r in rows if r["w2c"] is not None]
        c2ws = np.stack([np.linalg.inv(r["w2c"]) for r in good])
        pts = np.concatenate([r["points"] for r in good]) if good else np.zeros((0, 3))
        if len(pts) < 10:
            pts = c2ws[:, :3, 3] + c2ws[:, :3, 2]  # degenerate: points one unit in front
        self.transform = normalization_transform(c2ws, pts.astype(np.float64))
        centers = self.to_trainer(c2ws)[:, :3, 3]
        self.scene_scale = float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))
        logger.info(f"[Stream] World normalization frozen from {len(good)} cameras / {len(pts)} points; "
                    f"scene scale {self.scene_scale:.3f}")

    # ================================================================ seeds
    def _queue_seeds(self, kf: Keyframe, points: np.ndarray, pix: np.ndarray):
        self.store.mark_seeded(kf.kid)
        if len(points) == 0:
            return
        if self.cfg.depth and kf.w2c is not None:
            z = points.astype(np.float64) @ kf.w2c[2, :3] + kf.w2c[2, 3]
            pix32 = np.asarray(pix, np.float32).copy()
            with self._cond:
                self._depth_jobs.append((kf.kid, pix32, z))
                # also kept for re-fits (ACE-driven after refits, render-anchored every cycle);
                # same map version as its first alignment, so no ACE re-alignment is queued
                self._ace_pts[kf.kid] = (pix32, z, int(kf.map_version))
        img = decode_jpeg(kf.jpeg.to_pybytes())
        h, w = img.shape[:2]
        xs = np.clip((pix[:, 0] * w).astype(int), 0, w - 1)
        ys = np.clip((pix[:, 1] * h).astype(int), 0, h - 1)
        rgb = img[ys, xs].astype(np.float32) / 255.0
        xyz = self.transform_points(points.astype(np.float64)).astype(np.float32)
        with self._cond:
            self._seeds.append((xyz, rgb))

    def take_seeds(self) -> Tuple[np.ndarray, np.ndarray]:
        """New seed points (normalized xyz, rgb in [0, 1]) since the last call."""
        with self._cond:
            seeds, self._seeds = self._seeds, []
        if not seeds:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        return np.concatenate([s[0] for s in seeds]), np.concatenate([s[1] for s in seeds])

    # ================================================================ dense depth
    def take_depth_jobs(self, n: int) -> List[Tuple[Keyframe, np.ndarray, np.ndarray]]:
        """Up to n (keyframe, pix, z) jobs, newest first (evicted keyframes are skipped)."""
        out = []
        with self._cond:
            while self._depth_jobs and len(out) < n:
                kid, pix, z = self._depth_jobs.pop()
                kf = self.store.get(kid)
                if kf is not None and kf.w2c is not None and kid not in self._depth:
                    out.append((kf, pix, z))
        return out

    @property
    def depth_backlog(self) -> int:
        return len(self._depth_jobs)

    def set_depth(self, kid: int, depth_ace: Optional[np.ndarray], stats: Dict,
                  disp: Optional[np.ndarray] = None, version: Optional[int] = None):
        """Store one aligned depth map (ACE units, 0 = invalid; any resolution) or count a
        rejected one. ``disp``: the sharpened network output, cached for re-alignment;
        ``version``: the ACE map version it was aligned against."""
        if disp is not None:
            dh, dw = disp.shape
            self._disp[kid] = cv2.resize(disp, (dw // 2, dh // 2), interpolation=cv2.INTER_AREA).astype(np.float16)
        if version is not None:
            self._aligned_version[kid] = int(version)
        c = self.depth_counters
        realign = bool(stats.get("realign"))
        if realign and depth_ace is None:
            return  # keep the previous map; version above marks this ACE update as handled
        if not realign:
            n = c["maps"] + c["rejected"]
            c["ms"] = (c["ms"] * n + stats.get("ms", 0.0)) / (n + 1)
        if depth_ace is None:
            c["rejected"] += 1
            c["last_reject"] = stats.get("reason")
            return
        kf = self.store.get(kid)
        tw, th = ((kf.width // 2, kf.height // 2) if kf is not None
                  else (depth_ace.shape[1] // 2, depth_ace.shape[0] // 2))
        small = (depth_ace if depth_ace.shape == (th, tw)
                 else cv2.resize(depth_ace, (tw, th), interpolation=cv2.INTER_NEAREST))
        scale = float(np.linalg.norm(self.transform[:3, 0]))
        with self._cond:
            self._depth[kid] = (small * scale).astype(np.float16)
            live = {kf.kid for kf in self.store.frames()}
            for d in (self._depth, self._disp, self._ace_pts, self._aligned_version):
                for k in [k for k in d if k not in live]:
                    del d[k]
        err = stats.get("rel_err")
        if err is not None:
            c["rel_err"] = err if c["rel_err"] is None else 0.9 * c["rel_err"] + 0.1 * err
        if stats.get("anchored"):
            c["anchored"] += 1
        elif realign:
            c["realigned"] += 1
        else:
            c["maps"] += 1

    def depth_map(self, kid: int) -> Optional[np.ndarray]:
        return self._depth.get(kid)

    def realign_jobs(self) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]:
        """(kid, cached disparity, pix, z, version) of depth maps whose keyframe got newer ACE
        points (a refit re-localized it) than the ones they were aligned against."""
        out = []
        with self._cond:
            for kid, (pix, z, version) in list(self._ace_pts.items()):
                if kid in self._disp and version > self._aligned_version.get(kid, -1):
                    out.append((kid, self._disp[kid].astype(np.float32), pix, z, version))
        return out

    def anchor_jobs(self, kids: List[int]) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]:
        """(kid, cached disparity, pix, z, version) of these keyframes, for re-aligning their
        depth against the 3DGS render (closed loop)."""
        out = []
        with self._cond:
            for kid in kids:
                if kid in self._disp and kid in self._ace_pts:
                    pix, z, version = self._ace_pts[kid]
                    out.append((kid, self._disp[kid].astype(np.float32), pix, z, version))
        return out

    def request_refit(self, min_changes: int = 1) -> bool:
        """Start a background ACE refit now if the map is idle and at least ``min_changes``
        keyframes changed since the last one (newly registered or unregistered); training is
        not blocked."""
        with self._cond:
            if self._job is not None or not self.map_ready.is_set() or self.failed:
                return False
            if self._new_registered + self._new_failed < max(1, min_changes):
                return False
            before = self._job
            self._maybe_refit(force=True)
            started = self._job == "refit" and before is None
            if started:
                self._refit_requested += 1
            return started

    # ================================================================ geometry
    def to_trainer(self, c2w: np.ndarray) -> np.ndarray:
        """ACE camera-to-world (4x4 or Nx4x4) -> trainer's normalized frame."""
        if str(EXAMPLES_DIR) not in sys.path:
            sys.path.insert(0, str(EXAMPLES_DIR))
        from datasets.normalize import transform_cameras

        single = c2w.ndim == 2
        out = transform_cameras(self.transform, np.array(c2w, dtype=np.float64).reshape(-1, 4, 4))
        return out[0] if single else out

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.transform[:3, :3].T + self.transform[:3, 3]

    def intrinsics(self, width: int, height: int) -> np.ndarray:
        """K for a frame of this size: ACE's focal (px of the ACE files) scaled, centred pp."""
        f = self.focal_file * width / self.file_size[0]
        return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])

    # ================================================================ trainer views
    def views_table(self) -> pa.Table:
        return self.views()[1]

    def views(self) -> Tuple[int, pa.Table]:
        """(version, VIEW_SCHEMA table), cached per store version."""
        if self.transform is None or self.focal_file <= 0:
            return -1, VIEW_SCHEMA.empty_table()
        # Prefetch, dashboard, Flight and Difix threads all call this: build each version once,
        # and never let a slower builder overwrite a newer snapshot with an older one
        with self._views_lock:
            if self._views_cache[0] == self.store.version:
                return self._views_cache
            weight = confidence_weight_fn([kf.conf for kf in self.store.registered()])
            self._views_cache = self.store.views(self.to_trainer, self.intrinsics, lambda kf: weight(kf.conf))
            return self._views_cache

    def set_pseudo(self, views: List[PseudoView]):
        self.store.set_pseudo(views)
        self._evict()

    def latest_frame(self) -> Optional[Tuple[np.ndarray, float, bool, int]]:
        """(RGB image, source time, was a keyframe, sequence number) of the newest frame."""
        return self._latest

    def recently_accepted(self, window: float = 0.35) -> bool:
        return time.time() - self._last_accept_time < window

    def phase(self) -> Tuple[str, str]:
        """(phase id, human hint) for live displays."""
        if self.failed:
            return "failed", self.failed
        source_status = getattr(self.source, "status", None)
        if source_status and not self.source_ended:
            return "reconnecting", f"Camera lost: {source_status} (training continues meanwhile)"
        if not self.map_ready.is_set():
            if self._job == "bootstrap":
                return "mapping", (f"Building the ACE map from {self._bootstrap_target} keyframes "
                                   "(about a minute); keep the camera moving")
            n = len(self.store)
            if self.gate.offered == 0:
                return "waiting", "Waiting for frames from the source"
            return "collecting", (f"Move the camera slowly around the subject: "
                                  f"{n}/{self._bootstrap_target} keyframes")
        if self.source_ended:
            if self.recording_stopped:
                return "ended", f"Recording stopped ({self.recording_stopped}); training on the keyframes collected"
            return "ended", "Source ended; training on the keyframes collected"
        if self._job == "refit":
            return "tracking", "Tracking; growing the ACE map in the background"
        return "tracking", "Tracking: new views are added to training as you move"

    def stats(self) -> Dict:
        s = self.store
        return {
            "source": self.cfg.source, "kind": self.kind, "pace": getattr(self.source, "pace", None),
            "frames_read": getattr(self.source, "frames_read", 0),
            "frames_dropped": getattr(self.source, "frames_dropped", 0),
            "keyframes_offered": self.gate.offered, "keyframes_accepted": self.gate.accepted,
            "keyframes": len(s), "registered": len(s.registered()), "pending": len(s.pending()),
            "unregistered": len(s.unregistered()), "pseudo": len(s.pseudo()), "evicted": s.evicted,
            "store_mb": round(s.nbytes / 2 ** 20, 2), "budget_mb": self.cfg.budget_mb,
            "map_version": self.map_version, "job": self._job, "source_ended": self.source_ended,
            "recording_stopped": self.recording_stopped,
            "source_status": getattr(self.source, "status", None),
            "depth_maps": len(self._depth) if self.cfg.depth else None,
            "depth_rejected": self.depth_counters["rejected"] if self.cfg.depth else None,
            "depth_rel_err": (round(self.depth_counters["rel_err"], 4)
                              if self.cfg.depth and self.depth_counters["rel_err"] is not None else None),
            "depth_ms": round(self.depth_counters["ms"], 1) if self.cfg.depth else None,
            "depth_backlog": len(self._depth_jobs) if self.cfg.depth else None,
            "depth_realigned": self.depth_counters["realigned"] if self.cfg.depth else None,
            "depth_anchored": self.depth_counters["anchored"] if self.cfg.depth else None,
            "refits_requested": self._refit_requested,
            "reconnects": getattr(self.source, "reconnects", 0), "live": bool(getattr(self.source, "live", False)),
            "views_registered": self.views_registered, "gpu_lease": self.gpu_lease.holder,
            "failed": self.failed, **self.stats_counters,
        }

    def config_dict(self) -> Dict:
        return asdict(self.cfg)
