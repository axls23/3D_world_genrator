"""Single source of truth for the streaming mini-project (Policy & Process Map, 2026-10-27).

Six layers, each a separate process that talks to the next only through what is defined here:

    data source --(Kafka topics)--> processor --(model.score)--> database <--(reads)-- dashboard
    pose estimator / trainer hooks,  sliding windows, live       PostgreSQL     Streamlit,
    or the replay producer           metrics, Isolation Forest   history+jobs   2 s refresh

    poses     : ACE pose estimator (WSL) or replay   -> TOPIC_POSES
    training  : 3DGS trainer (Windows) or replay     -> TOPIC_TRAINING
    anomalies : processor                            -> TOPIC_ANOMALIES

The video-to-3D reconstruction stays a batch job; only small numeric events are streamed
(images, depth maps and point clouds never go through Kafka, see MAX_MESSAGE_BYTES).

Stdlib only and py3.8-compatible: the ACE env imports this file next to stream/schema.py.
Every setting can be overridden with the HYPERSPLAT_<NAME> environment variable named beside it.
"""

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple, Union

SCHEMA_VERSION = 1


def _env(name: str, default: str) -> str:
    return os.environ.get("HYPERSPLAT_" + name, default)


# ---------------------------------------------------------------------------------------------
# Network (rows 5, 19; constraints "two operating systems" and "ports already taken")
# ---------------------------------------------------------------------------------------------

RESERVED_PORTS = (8080, 8081, 8092)  # gsplat-server in docker-compose.yml

# Kafka advertises two listeners: INTERNAL for containers on the compose network and EXTERNAL
# on the Docker Desktop host, which Windows reaches at localhost and WSL reaches at localhost
# with mirrored networking (otherwise set HYPERSPLAT_KAFKA to the Windows host IP).
KAFKA_INTERNAL_PORT = 29092
KAFKA_EXTERNAL_PORT = int(_env("KAFKA_PORT", "9094"))
KAFKA_BOOTSTRAP = _env("KAFKA", "localhost:%d" % KAFKA_EXTERNAL_PORT)
KAFKA_BOOTSTRAP_INTERNAL = _env("KAFKA_INTERNAL", "kafka:%d" % KAFKA_INTERNAL_PORT)

DB_PORT = int(_env("DB_PORT", "5433"))  # 5433 so a native Windows PostgreSQL on 5432 is untouched
DB_DSN = _env("DB_DSN", "postgresql://hypersplat:hypersplat@localhost:%d/hypersplat" % DB_PORT)
DB_DSN_INTERNAL = _env("DB_DSN_INTERNAL", "postgresql://hypersplat:hypersplat@db:5432/hypersplat")

DASHBOARD_PORT = int(_env("DASHBOARD_PORT", "8501"))


# ---------------------------------------------------------------------------------------------
# Topics (row 6). Keyed by job_id so one job's events stay ordered within a partition.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Topic:
    name: str
    partitions: int
    retention_ms: int
    description: str


_DAY_MS = 24 * 3600 * 1000
TOPIC_PREFIX = _env("TOPIC_PREFIX", "")  # lets tests and parallel setups share one broker
TOPIC_POSES = Topic(TOPIC_PREFIX + "hypersplat.camera-poses", 1, 7 * _DAY_MS, "one CameraPose per registered frame")
TOPIC_TRAINING = Topic(TOPIC_PREFIX + "hypersplat.training-measurements", 1, 7 * _DAY_MS,
                       "one TrainingSnapshot every TRAIN_LOG_EVERY steps")
TOPIC_ANOMALIES = Topic(TOPIC_PREFIX + "hypersplat.flagged-problems", 1, 30 * _DAY_MS,
                        "one Anomaly per event the models flag")
TOPIC_DEAD_LETTER = Topic(TOPIC_PREFIX + "hypersplat.dead-letter", 1, 30 * _DAY_MS,
                          "one DeadLetter per message the processor could not decode or store")
TOPICS = (TOPIC_POSES, TOPIC_TRAINING, TOPIC_ANOMALIES, TOPIC_DEAD_LETTER)
DEAD_LETTER_RAW_CHARS = 256   # how much of the bad payload is kept: enough to diagnose, never the whole thing


# ---------------------------------------------------------------------------------------------
# Producer policy (rows 3, 4; constraints "training speed comes first", "small messages only")
# ---------------------------------------------------------------------------------------------

TRAIN_LOG_EVERY = int(_env("TRAIN_LOG_EVERY", "10"))   # trainer steps between snapshots
REPLAY_RATE_HZ = float(_env("REPLAY_RATE_HZ", "5"))     # replay producer pace, events/s
MAX_MESSAGE_BYTES = 2048                                # a pose event is ~400 B as JSON

# Never block the trainer: sends go to a bounded in-process queue drained by a background
# thread; a full queue or an unreachable broker drops the event and bumps the drop counter.
PRODUCER_QUEUE_MAX = 10000
PRODUCER_CONFIG = {
    "acks": 1,
    "linger_ms": 50,
    "max_block_ms": 2000,       # only the sender thread waits; callers never touch Kafka
    "retries": 0,
    "request_timeout_ms": 2000,
}

# Constraint "keep the old path as a fallback": poses_final.txt / shared-folder hand-off stays
# on until Kafka has carried one full reconstruction cleanly.
FILE_HANDOFF_ENABLED = _env("FILE_HANDOFF", "1") == "1"
STREAMING_ENABLED = _env("STREAMING", "0") == "1"


# ---------------------------------------------------------------------------------------------
# Events (row 2)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraPose:
    """One registered frame from the pose estimator, as ACE writes it to poses_final.txt.

    Rotation and translation are world-to-camera (OpenCV); camera_center() gives the position.
    """
    job_id: str
    seq: int                     # per job and producer, 0, 1, 2, ... (gaps = lost events)
    ts: float                    # unix seconds when produced (replay: when replayed)
    frame: str                   # image file name, no directory
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    focal: float                 # px
    confidence: float            # ACE inlier count / score, >= 0
    dropped: int = 0             # producer-side drops so far for this job (row 4 counter)
    injected: Optional[str] = None  # ground-truth label for row 12: shuffle | blur | None
    source: str = "pose"         # see POSE_SOURCES


# Where a pose came from. reloc/map rows move many cameras at once when continuous mapping or
# a refit lands, so they update the camera path but are not scored as anomalies.
POSE_SOURCES = ("pose", "reloc", "map", "file", "replay")
SCORED_SOURCES = ("pose", "file", "replay")


@dataclass(frozen=True)
class TrainingSnapshot:
    """Trainer state every TRAIN_LOG_EVERY steps; names mirror the train/* TensorBoard tags."""
    job_id: str
    seq: int
    ts: float
    step: int
    loss: float                  # train/loss
    num_gaussians: int           # train/num_GS
    gpu_mem_gb: float            # train/mem (torch max_memory_allocated)
    l1: Optional[float] = None   # train/l1loss
    ssim: Optional[float] = None  # train/ssimloss
    grad_norm: Optional[float] = None  # train/grad_norm
    steps_per_s: Optional[float] = None  # streaming on/off throughput comparison
    dropped: int = 0
    injected: Optional[str] = None


@dataclass(frozen=True)
class Anomaly:
    """Processor output for one flagged event; also written as is_anomaly in history."""
    job_id: str
    ts: float
    source: str                  # "pose" | "training"
    source_seq: int
    model: str                   # model file stem, e.g. "pose_iforest"
    score: float                 # sklearn decision_function: lower = more anomalous
    reason: str                  # top feature, e.g. "turn_deg=74.2"
    injected: Optional[str] = None
    anomaly_id: str = ""         # stable per flagged event (anomaly_id()): consumers dedupe replays on it


@dataclass(frozen=True)
class DeadLetter:
    """A message the processor could not decode or store, kept for inspection instead of vanishing."""
    job_id: str                  # from the Kafka key, "unknown" when the message had none
    ts: float
    source_topic: str
    reason: str
    raw: str                     # first DEAD_LETTER_RAW_CHARS characters of the payload


def anomaly_id(job_id: str, source: str, source_seq: int, model: str) -> str:
    """Deterministic identity of one flagged event: a replayed batch re-sends the same id, so the
    anomalies topic is at-least-once but deduplicable (it is not exactly-once; see architecture)."""
    return hashlib.sha1(("%s|%s|%d|%s" % (job_id, source, source_seq, model)).encode()).hexdigest()[:16]


Event = Union[CameraPose, TrainingSnapshot, Anomaly, DeadLetter]
KIND_OF = {CameraPose: "pose", TrainingSnapshot: "training", Anomaly: "anomaly", DeadLetter: "dead_letter"}
CLASS_OF = {v: k for k, v in KIND_OF.items()}
TOPIC_OF = {CameraPose: TOPIC_POSES, TrainingSnapshot: TOPIC_TRAINING, Anomaly: TOPIC_ANOMALIES,
            DeadLetter: TOPIC_DEAD_LETTER}

# Fields that must be present and finite; anything else optional (row 4).
_REQUIRED_FINITE = {
    CameraPose: ("ts", "qw", "qx", "qy", "qz", "tx", "ty", "tz", "focal", "confidence"),
    TrainingSnapshot: ("ts", "loss", "gpu_mem_gb"),
    Anomaly: ("ts", "score"),
    DeadLetter: ("ts",),
}
_INT_FIELDS = {"seq", "step", "num_gaussians", "dropped", "source_seq"}


class InvalidEvent(ValueError):
    pass


class Unregistered(InvalidEvent):
    """ACE could not localize the frame (null w2c): a normal outcome, counted apart from drops."""


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(cls, data: Dict) -> Event:
    """Build an event from a dict, or raise InvalidEvent with a short reason (row 4).

    Unknown keys are ignored so older consumers survive newer producers. Pose quaternions are
    renormalized; a quaternion far from unit length is rejected as corrupt.
    """
    names = {f.name for f in fields(cls)}
    kw = {k: v for k, v in data.items() if k in names}
    for k, v in kw.items():  # numpy scalars (Arrow rows, torch .item() aside) -> float
        if v is not None and not isinstance(v, (bool, str, int, float)) and hasattr(v, "__float__"):
            kw[k] = float(v)
    for name in _REQUIRED_FINITE[cls]:
        if not _finite(kw.get(name)):
            raise InvalidEvent("%s: %s=%r" % (KIND_OF[cls], name, kw.get(name)))
    for name in names & _INT_FIELDS:
        if name in kw:
            if not _finite(kw[name]) or int(kw[name]) != kw[name] or kw[name] < 0:
                raise InvalidEvent("%s: %s=%r" % (KIND_OF[cls], name, kw[name]))
            kw[name] = int(kw[name])
    for name in ("l1", "ssim", "grad_norm", "steps_per_s"):
        if kw.get(name) is not None and not _finite(kw[name]):
            kw[name] = None  # optional measurement went bad: keep the event, drop the value
    if not kw.get("job_id"):
        raise InvalidEvent("%s: job_id missing" % KIND_OF[cls])
    if cls is CameraPose:
        n = math.sqrt(kw["qw"] ** 2 + kw["qx"] ** 2 + kw["qy"] ** 2 + kw["qz"] ** 2)
        if not 0.9 < n < 1.1:
            raise InvalidEvent("pose: |q|=%.3f" % n)
        if abs(n - 1) > 1e-9:  # leave unit quaternions untouched so decode(encode(e)) == e
            for k in ("qw", "qx", "qy", "qz"):
                kw[k] = kw[k] / n
        if kw.get("source", "pose") not in POSE_SOURCES:
            raise InvalidEvent("pose: source=%r" % kw["source"])
        if kw["focal"] <= 0 or kw["confidence"] < 0 or not kw.get("frame"):
            raise InvalidEvent("pose: focal=%r confidence=%r frame=%r"
                               % (kw["focal"], kw["confidence"], kw.get("frame")))
    if cls is TrainingSnapshot and (kw["loss"] < 0 or kw.get("num_gaussians", 0) < 0):
        raise InvalidEvent("training: loss=%r" % kw["loss"])
    try:
        return cls(**kw)
    except TypeError as e:  # missing required field
        raise InvalidEvent("%s: %s" % (KIND_OF[cls], e))


def encode(event: Event) -> Tuple[str, bytes, bytes]:
    """(topic name, key, value) ready for producer.send(topic, key=key, value=value)."""
    body = asdict(event)
    body["kind"] = KIND_OF[type(event)]
    body["v"] = SCHEMA_VERSION
    try:
        value = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
    except ValueError as e:  # NaN / inf in an event that skipped validate()
        raise InvalidEvent("%s: %s" % (body["kind"], e))
    if len(value) > MAX_MESSAGE_BYTES:
        raise InvalidEvent("%s: %d bytes > %d" % (body["kind"], len(value), MAX_MESSAGE_BYTES))
    return TOPIC_OF[type(event)].name, event.job_id.encode(), value


def decode(value: bytes) -> Event:
    """Inverse of encode(); raises InvalidEvent on anything malformed."""
    try:
        data = json.loads(value)
        cls = CLASS_OF[data["kind"]]
    except (ValueError, KeyError, TypeError) as e:
        raise InvalidEvent("undecodable message: %s" % e)
    return validate(cls, data)


# ---------------------------------------------------------------------------------------------
# Adapters from the data sources that already exist (rows 2, 3)
# ---------------------------------------------------------------------------------------------

def pose_from_ace_line(line: str, job_id: str, seq: int, ts: Optional[float] = None,
                       **extra) -> CameraPose:
    """One poses_final.txt line (scripts/acezero/dataset_io.write_pose_to_pose_file):
    ``<image_path> qw qx qy qz tx ty tz <focal> <confidence>``."""
    parts = line.split()
    if len(parts) < 10:
        raise InvalidEvent("pose: short line (%d fields)" % len(parts))
    try:
        nums = [float(p) for p in parts[1:10]]
    except ValueError as e:
        raise InvalidEvent("pose: %s" % e)
    keys = ("qw", "qx", "qy", "qz", "tx", "ty", "tz", "focal", "confidence")
    data = dict(zip(keys, nums), job_id=job_id, seq=seq, ts=time.time() if ts is None else ts,
                frame=os.path.basename(parts[0].replace("\\", "/")), **extra)
    return validate(CameraPose, data)


def _quat_from_rotmat(R: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """(w, x, y, z) of a 3x3 rotation, Shepperd's method (stable for every rotation)."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = 2 * math.sqrt(tr + 1)
        return 0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s
    if R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = 2 * math.sqrt(1 + R[0][0] - R[1][1] - R[2][2])
        return (R[2][1] - R[1][2]) / s, 0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s
    if R[1][1] > R[2][2]:
        s = 2 * math.sqrt(1 + R[1][1] - R[0][0] - R[2][2])
        return (R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s
    s = 2 * math.sqrt(1 + R[2][2] - R[0][0] - R[1][1])
    return (R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s


def _flat16(m) -> List[float]:
    if hasattr(m, "reshape"):  # numpy, without importing it
        return [float(v) for v in m.reshape(-1)[:16]]
    flat = [v for r in m for v in r] if m and isinstance(m[0], (list, tuple)) else list(m)
    return [float(v) for v in flat[:16]]


def pose_from_event_row(row: Dict, job_id: str, seq: int, focal: Optional[float] = None,
                        ts: Optional[float] = None, **extra) -> CameraPose:
    """One ACE EVENT row as the stream session receives it (hypersplat/stream/schema.py):
    kind pose | reloc | map, frame, w2c (4x4, flat 16, or None), conf, focal.

    ``focal`` is the fallback when the row carries none (pose rows reuse the map's focal).
    Raises Unregistered for a frame ACE could not localize, InvalidEvent for bad values.
    """
    if row.get("w2c") is None:
        raise Unregistered("pose: %s not registered" % row.get("frame"))
    m = _flat16(row["w2c"])
    if len(m) < 12 or not all(math.isfinite(v) for v in m):
        raise InvalidEvent("pose: bad w2c for %s" % row.get("frame"))
    qw, qx, qy, qz = _quat_from_rotmat([m[0:3], m[4:7], m[8:11]])
    row_focal = row.get("focal")
    data = dict(qw=qw, qx=qx, qy=qy, qz=qz, tx=m[3], ty=m[7], tz=m[11],
                focal=row_focal if _finite(row_focal) and row_focal > 0 else focal,
                confidence=row.get("conf"), job_id=job_id, seq=seq,
                ts=time.time() if ts is None else ts, source=row.get("kind", "pose"),
                frame=os.path.basename(str(row.get("frame", "")).replace("\\", "/")))
    data.update(extra)
    return validate(CameraPose, data)


# TensorBoard tag -> TrainingSnapshot field. Used by the trainer hook (same values it already
# passes to add_scalar) and by the offline model trainer reading past runs' event files.
TB_TAGS = {
    "train/loss": "loss",
    "train/num_GS": "num_gaussians",
    "train/mem": "gpu_mem_gb",
    "train/l1loss": "l1",
    "train/ssimloss": "ssim",
    "train/grad_norm": "grad_norm",
}


def snapshot_from_scalars(step: int, scalars: Dict[str, float], job_id: str, seq: int,
                          ts: Optional[float] = None, **extra) -> TrainingSnapshot:
    data = {TB_TAGS[k]: v for k, v in scalars.items() if k in TB_TAGS}
    data.update(job_id=job_id, seq=seq, step=step, ts=time.time() if ts is None else ts, **extra)
    return validate(TrainingSnapshot, data)


# ---------------------------------------------------------------------------------------------
# Processor (rows 7, 8, 9)
# ---------------------------------------------------------------------------------------------

# 30 snapshots = 300 trainer steps at TRAIN_LOG_EVERY=10, long enough to see a loss trend
# through per-batch noise yet short enough to react within a minute; 30 poses ~ one second of
# 30 fps keyframes, about one camera motion segment.
WINDOW = int(_env("WINDOW", "30"))

# Offsets are committed only after the DB write, so a restarted processor resumes where it
# stopped; inserts are idempotent on (job_id, seq), so the replayed tail is not duplicated.
CONSUMER_GROUP = _env("CONSUMER_GROUP", "hypersplat-processor")
CONSUMER_CONFIG = {
    "group_id": CONSUMER_GROUP,
    "enable_auto_commit": False,
    "auto_offset_reset": "earliest",
}


def _rotmat(q: Sequence[float]) -> List[List[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def camera_center(p: CameraPose) -> Tuple[float, float, float]:
    """World position of the camera: C = -R^T t for world-to-camera (R, t)."""
    R = _rotmat((p.qw, p.qx, p.qy, p.qz))
    t = (p.tx, p.ty, p.tz)
    return tuple(-sum(R[r][c] * t[r] for r in range(3)) for c in range(3))


def turn_deg(a: CameraPose, b: CameraPose) -> float:
    dot = abs(a.qw * b.qw + a.qx * b.qx + a.qy * b.qy + a.qz * b.qz)
    return math.degrees(2 * math.acos(min(1.0, dot)))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _slope(ys: Sequence[float]) -> float:
    """Least-squares slope per sample; 0 for fewer than two points."""
    n = len(ys)
    if n < 2:
        return 0.0
    mx, my = (n - 1) / 2, _mean(ys)
    den = sum((i - mx) ** 2 for i in range(n))
    return sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den


# Feature vectors are defined here, not in the processor or the offline trainer, so the model
# is scored on exactly what it was fitted on (row 11, no train/serve skew).
# Scale-free on purpose: step_dist and raw confidence depend on the scene (a metre in a desk scan is
# a jump, in a courtyard a stroll), so the model sees each pose relative to the poses before it.
# step_dist stays in pose_features() for the KPIs and the database, just not in the model.
POSE_FEATURES = ("step_ratio", "turn_deg", "conf_ratio", "focal_ratio")
TRAINING_FEATURES = ("loss_ratio_best", "loss_trend", "loss_cv", "gs_growth", "mem_delta_gb")


def pose_features(window: Sequence[CameraPose]) -> Dict[str, float]:
    """Features of window[-1] against the poses before it (oldest first)."""
    cur = window[-1]
    prev = window[-2] if len(window) > 1 else None
    hist = window[:-1] or window
    step = 0.0
    if prev is not None:
        a, b = camera_center(prev), camera_center(cur)
        step = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
    past = [camera_center(p) for p in hist]
    steps = sorted(math.dist(a, b) for a, b in zip(past, past[1:]))
    typical = steps[len(steps) // 2] if steps else 0.0
    conf_mean = _mean([p.confidence for p in hist])
    focals = sorted(p.focal for p in hist)
    return {
        "step_dist": step,
        "step_ratio": step / typical if typical > 0 else 1.0,
        "turn_deg": turn_deg(prev, cur) if prev is not None else 0.0,
        "confidence": cur.confidence,
        "conf_ratio": cur.confidence / conf_mean if conf_mean > 0 else 1.0,
        "focal_ratio": cur.focal / focals[len(focals) // 2],
    }


def training_features(window: Sequence[TrainingSnapshot]) -> Dict[str, float]:
    losses = [s.loss for s in window]
    mean = _mean(losses)
    var = _mean([(x - mean) ** 2 for x in losses])
    span = max(1, window[-1].step - window[0].step)
    return {
        "loss_ratio_best": losses[-1] / min(losses) if min(losses) > 0 else 1.0,
        "loss_trend": _slope(losses) / mean if mean > 0 else 0.0,
        "loss_cv": math.sqrt(var) / mean if mean > 0 else 0.0,
        "gs_growth": (window[-1].num_gaussians - window[0].num_gaussians) / span,
        "mem_delta_gb": window[-1].gpu_mem_gb - window[0].gpu_mem_gb,
    }


# Rising loss: the Isolation Forest sees one window's shape, so a slow climb (each step only a little
# worse than the last) looks normal to it. A rank-trend test on the newest losses catches it instead.
RISING_LEN = 8        # newest snapshots examined
RISING_TAU = 0.6      # Mann-Kendall tau (-1 falling .. +1 rising) needed: mostly-increasing, not one bump
RISING_MIN_RISE = 0.10  # and the newest 3 losses average this much above the oldest 3 (relative)


def rising_loss(window: Sequence[TrainingSnapshot]) -> Tuple[bool, float]:
    """(flag, relative rise) for a loss that has been climbing over the last RISING_LEN snapshots."""
    losses = [s.loss for s in window][-RISING_LEN:]
    if len(losses) < RISING_LEN:
        return False, 0.0
    pairs = [(losses[j] > losses[i]) - (losses[j] < losses[i])
             for i in range(len(losses)) for j in range(i + 1, len(losses))]
    tau = sum(pairs) / len(pairs)
    first, last = _mean(losses[:3]), _mean(losses[-3:])
    rise = (last - first) / first if first > 0 else 0.0
    return (tau >= RISING_TAU and rise >= RISING_MIN_RISE), rise


# Live KPIs (row 9), the column names the processor writes and the dashboard reads.
POSE_METRICS = ("step_dist", "turn_deg", "conf_roll")
TRAINING_METRICS = ("loss_roll", "loss_trend", "gs_growth", "best_loss", "peak_mem_gb")


# ---------------------------------------------------------------------------------------------
# Anomaly models (rows 10, 11, 12)
# ---------------------------------------------------------------------------------------------

MODEL_DIR = _env("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
POSE_MODEL = "pose_iforest"
TRAINING_MODEL = "training_iforest"


def model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name + ".joblib")


CONTAMINATION_GRID = (0.01, 0.05, 0.10)    # row 12 experiment
CONTAMINATION = float(_env("CONTAMINATION", "0.05"))
IFOREST_PARAMS = {"n_estimators": 200, "random_state": 0}
MIN_WINDOW_TO_SCORE = 5   # fewer events than this: metrics only, no score
INJECTIONS = ("shuffle", "blur")           # row 12/21 ground-truth bad-pose kinds


# ---------------------------------------------------------------------------------------------
# Database (rows 13, 14, 15). Processor writes, dashboard reads; PostgreSQL DDL.
# ---------------------------------------------------------------------------------------------

DASHBOARD_ROWS = int(_env("DASHBOARD_ROWS", "500"))   # dashboard reads the current job's tail

DDL = """
CREATE TABLE IF NOT EXISTS jobs (                 -- one-row current snapshot per job
    job_id          TEXT PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    current_step    INTEGER,
    current_loss    DOUBLE PRECISION,
    best_loss       DOUBLE PRECISION,
    num_gaussians   INTEGER,
    peak_gpu_mem_gb DOUBLE PRECISION,
    poses_accepted  INTEGER NOT NULL DEFAULT 0,
    poses_rejected  INTEGER NOT NULL DEFAULT 0,   -- flagged by the pose model
    events_dropped  INTEGER NOT NULL DEFAULT 0,   -- producer + processor validation drops
    anomalies       INTEGER NOT NULL DEFAULT 0,
    archived        BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS pose_history (
    job_id        TEXT NOT NULL REFERENCES jobs(job_id),
    seq           INTEGER NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    frame         TEXT NOT NULL,
    cx            DOUBLE PRECISION, cy DOUBLE PRECISION, cz DOUBLE PRECISION,
    focal         DOUBLE PRECISION,
    confidence    DOUBLE PRECISION,
    step_dist     DOUBLE PRECISION,
    turn_deg      DOUBLE PRECISION,
    conf_roll     DOUBLE PRECISION,
    anomaly_score DOUBLE PRECISION,
    is_anomaly    BOOLEAN NOT NULL DEFAULT false,
    injected      TEXT,
    PRIMARY KEY (job_id, seq)
);

CREATE TABLE IF NOT EXISTS training_history (
    job_id        TEXT NOT NULL REFERENCES jobs(job_id),
    seq           INTEGER NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    step          INTEGER NOT NULL,
    loss          DOUBLE PRECISION NOT NULL,
    num_gaussians INTEGER,
    gpu_mem_gb    DOUBLE PRECISION,
    steps_per_s   DOUBLE PRECISION,
    loss_roll     DOUBLE PRECISION,
    loss_trend    DOUBLE PRECISION,
    gs_growth     DOUBLE PRECISION,
    best_loss     DOUBLE PRECISION,
    peak_mem_gb   DOUBLE PRECISION,
    anomaly_score DOUBLE PRECISION,
    is_anomaly    BOOLEAN NOT NULL DEFAULT false,
    injected      TEXT,
    PRIMARY KEY (job_id, seq)
);

-- Growth plan (row 15): the dashboard only reads the newest DASHBOARD_ROWS of one job, which
-- these indexes serve without scanning old jobs; old jobs are archived=true, not deleted.
CREATE INDEX IF NOT EXISTS pose_history_job_ts ON pose_history (job_id, ts DESC);
CREATE INDEX IF NOT EXISTS training_history_job_ts ON training_history (job_id, ts DESC);
"""

# Health check (row 20): counts must climb between two runs of it, anomalies must be > 0.
HEALTH_QUERY = """
SELECT j.job_id,
       (SELECT count(*) FROM pose_history p WHERE p.job_id = j.job_id)     AS poses,
       (SELECT count(*) FROM training_history t WHERE t.job_id = j.job_id) AS snapshots,
       j.anomalies, j.events_dropped, j.updated_at
FROM jobs j WHERE NOT j.archived ORDER BY j.updated_at DESC
"""


# ---------------------------------------------------------------------------------------------
# Dashboard (rows 16, 17, 18)
# ---------------------------------------------------------------------------------------------

DASHBOARD_REFRESH_S = float(_env("DASHBOARD_REFRESH_S", "2"))  # fresh DB connection each time
ANOMALY_COLOR = "#d62728"   # red markers on the loss curve and top-down camera path


# ---------------------------------------------------------------------------------------------
# Architecture registry (rows 1, 19, 22): one entry per layer, in start order.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Layer:
    order: int
    name: str
    tool: str
    module: Optional[str]       # entry point under hypersplat/streaming, None = external
    compose_service: Optional[str]
    port: Optional[int]


LAYERS = (
    Layer(1, "broker", "Apache Kafka (KRaft)", None, "kafka", KAFKA_EXTERNAL_PORT),
    Layer(2, "topics", "kafka-topics", "hypersplat.streaming.topics", "kafka-init", None),
    Layer(3, "database", "PostgreSQL", None, "db", DB_PORT),
    Layer(4, "anomaly model", "scikit-learn IsolationForest", "hypersplat.streaming.train_models",
          None, None),
    Layer(5, "processor", "Python consumer", "hypersplat.streaming.processor", "processor", None),
    Layer(6, "dashboard", "Streamlit", "hypersplat.streaming.dashboard", "dashboard",
          DASHBOARD_PORT),
    Layer(7, "data source", "trainer/ACE hooks or replay", "hypersplat.streaming.replay", None, None),
)


def check_ports() -> None:
    used = [l.port for l in LAYERS if l.port is not None] + [KAFKA_INTERNAL_PORT]
    clash = sorted(set(used) & set(RESERVED_PORTS))
    if clash or len(used) != len(set(used)):
        raise RuntimeError("streaming ports %s clash with %s or each other" % (used, RESERVED_PORTS))


check_ports()
