"""Processor layer: Kafka -> sliding windows -> live metrics + anomaly score -> PostgreSQL
(Policy & Process Map rows 7, 8, 9, 10 serving, 14, 15).

    TOPIC_POSES ----\\                          /--> pose_history / training_history / jobs
                     >-- per-job windows ------<
    TOPIC_TRAINING -/    metrics, IsolationForest \\--> TOPIC_ANOMALIES (flagged events only)

    python -m hypersplat.streaming.processor [--once] [--max-seconds S]

Delivery (row 7): offsets are committed only after the DB transaction of a poll batch has
committed, so a restarted processor resumes exactly where it stopped. The tail it may replay is
deduplicated by the (job_id, seq) primary keys, and the ``jobs`` counters are bumped only for rows
the INSERT really added, so replays leave every count unchanged.

State (row 8): one deque(maxlen=WINDOW) of poses and one of snapshots per job, in memory. After a
restart they are rebuilt lazily from the history tables on a job's first event.

Models (rows 10, 11): loaded once at startup, never refitted here. Without a model file the
processor still writes every metric and leaves anomaly_score NULL.

The pure logic (JobState, pose_row, training_row, AnomalyModel, Processor) runs without Kafka or
PostgreSQL; kafka-python, psycopg and joblib are imported lazily so tests can stub them.
"""

import argparse
import logging
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import contract as c

logger = logging.getLogger(__name__)

POLL_MS = 500          # poll timeout; also the worst-case reaction time to SIGINT/SIGTERM
MAX_RECORDS = 500      # one DB transaction per poll batch of at most this many events
LOG_EVERY_S = 10.0     # throughput log line period
# --once with no partitions assigned yet: give up after this. Longer than the broker's 45 s
# session timeout, which a restart after SIGKILL waits out before the group rebalances.
ONCE_WAIT_ASSIGN_S = 60.0


# ---------------------------------------------------------------------------------------------
# Anomaly models (rows 10, 11)
# ---------------------------------------------------------------------------------------------

_STAT_KEYS = (("mean", "std"), ("feature_mean", "feature_std"), ("feature_mean_", "feature_std_"),
              ("train_mean", "train_std"))


def _as_vector(v, features: Sequence[str], saved_order: Optional[Sequence[str]]) -> Optional[List[float]]:
    if isinstance(v, dict):
        return [float(v[f]) for f in features] if all(f in v for f in features) else None
    try:
        vals = [float(x) for x in v]
    except TypeError:
        return None
    order = list(saved_order) if saved_order else list(features)
    if len(vals) != len(order) or not set(features) <= set(order):
        return None
    by_name = dict(zip(order, vals))
    return [by_name[f] for f in features]


def _feature_stats(obj, est, features: Sequence[str]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Training-set mean/std per feature if the model file stores them (any of a few layouts:
    the offline trainer is a separate unit, so accept what it is likely to write)."""
    saved_order = obj.get("features") if isinstance(obj, dict) else None
    sources = [obj] if isinstance(obj, dict) else []
    stats = obj.get("stats") if isinstance(obj, dict) else None
    if isinstance(stats, dict):
        sources.append(stats)
        if all(isinstance(stats.get(f), dict) for f in features):  # {feature: {mean, std}}
            sources.append({"mean": {f: stats[f].get("mean") for f in features},
                            "std": {f: stats[f].get("std") for f in features}})
    sources.append(getattr(est, "__dict__", {}))
    steps = getattr(est, "steps", None)  # sklearn Pipeline whose first step is a StandardScaler
    if steps and hasattr(steps[0][1], "mean_") and hasattr(steps[0][1], "scale_"):
        sources.append({"mean": steps[0][1].mean_, "std": steps[0][1].scale_})
    for src in sources:
        for mk, sk in _STAT_KEYS:
            if src.get(mk) is not None and src.get(sk) is not None:
                try:
                    mean = _as_vector(src[mk], features, saved_order)
                    std = _as_vector(src[sk], features, saved_order)
                except (TypeError, ValueError, KeyError):
                    continue
                if mean is not None and std is not None:
                    return mean, std
    return None, None


class AnomalyModel:
    """A fitted estimator plus the contract feature order it is scored on."""

    def __init__(self, name: str, estimator, features: Sequence[str],
                 mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None):
        self.name = name
        self.estimator = estimator
        self.features = tuple(features)
        self.mean = list(mean) if mean is not None else None
        self.std = list(std) if std is not None else None

    def _X(self, x: List[float]):
        if hasattr(self.estimator, "feature_names_in_"):  # fitted on a DataFrame: keep sklearn quiet
            import pandas as pd

            return pd.DataFrame([x], columns=list(self.features))
        return [x]

    def reason(self, x: Sequence[float]) -> str:
        """Top feature: largest |z| against the training stats, else the largest |value|."""
        if self.mean is not None and self.std is not None:
            mags = [abs(v - m) / (s if s and s > 0 else 1.0) for v, m, s in zip(x, self.mean, self.std)]
        else:
            mags = [abs(v) for v in x]
        i = max(range(len(x)), key=lambda k: mags[k])
        return "%s=%.4g" % (self.features[i], x[i])

    def score(self, feats: Dict[str, float]) -> Tuple[float, bool, str]:
        """(decision_function score, predict == -1, reason)."""
        x = [float(feats[f]) for f in self.features]
        X = self._X(x)
        s = float(self.estimator.decision_function(X)[0])
        flagged = int(self.estimator.predict(X)[0]) == -1
        return s, flagged, self.reason(x)


def load_model(name: str, features: Sequence[str]) -> Optional[AnomalyModel]:
    """The model saved by the offline trainer, or None (metrics-only) if missing or unusable."""
    path = c.model_path(name)
    if not os.path.exists(path):
        logger.warning("[processor] no model at %s: %s events get metrics only, no score", path, name)
        return None
    try:
        import joblib

        obj = joblib.load(path)
    except Exception as e:
        logger.error("[processor] cannot load %s (%s): metrics only", path, e)
        return None
    est = obj
    if isinstance(obj, dict):
        est = next((obj[k] for k in ("model", "estimator", "iforest") if k in obj), None)
        saved = obj.get("features")
        if saved is not None and tuple(saved) != tuple(features):
            if sorted(saved) != sorted(features):
                logger.error("[processor] %s was fitted on %s, contract says %s: metrics only",
                             path, tuple(saved), tuple(features))
                return None
            features = tuple(saved)  # same features, other column order: feed it what it saw
    if not (hasattr(est, "decision_function") and hasattr(est, "predict")):
        logger.error("[processor] %s holds no estimator with decision_function/predict: metrics only", path)
        return None
    mean, std = _feature_stats(obj, est, features)
    logger.info("[processor] loaded %s (%s)", path, "with feature stats" if mean else "no feature stats")
    return AnomalyModel(name, est, features, mean, std)


# ---------------------------------------------------------------------------------------------
# Per-job state and rows (rows 8, 9)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RestoredPose(c.CameraPose):
    """A pose rebuilt from pose_history after a restart. The table keeps the camera center but
    not the rotation, so rotation is identity, t = -C (camera_center() gives C back) and the turn
    from it to the next pose is unknown; ``turn_hint`` (its own stored turn_deg) stands in."""
    turn_hint: float = 0.0


@dataclass
class JobState:
    poses: deque = field(default_factory=lambda: deque(maxlen=c.WINDOW))   # scored sources only
    snaps: deque = field(default_factory=lambda: deque(maxlen=c.WINDOW))
    poses_loaded: bool = False
    snaps_loaded: bool = False
    last_pose_seq: int = -1
    last_snap_seq: int = -1
    best_loss: Optional[float] = None
    peak_mem_gb: Optional[float] = None
    dropped_base: int = 0       # jobs.events_dropped found at restart
    producer_dropped: int = 0   # max ``dropped`` seen since start
    decode_drops: int = 0       # undecodable / invalid messages since start

    def events_dropped(self) -> int:
        # Before a restart: max(dropped) + decode drops, exactly. After one, the old total is a
        # floor: the producer's cumulative count only adds once it passes it.
        return max(self.dropped_base, self.producer_dropped) + self.decode_drops


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pose_window_features(window: Sequence[c.CameraPose]) -> Dict[str, float]:
    feats = c.pose_features(window)
    if len(window) > 1 and isinstance(window[-2], RestoredPose):
        feats["turn_deg"] = window[-2].turn_hint
    return feats


def pose_row(window: Sequence[c.CameraPose], model: Optional[AnomalyModel],
             scored: bool) -> Tuple[Dict, Optional[str]]:
    """pose_history row for window[-1] (POSE_METRICS + score) and the anomaly reason if flagged."""
    p = window[-1]
    feats = pose_window_features(window)
    cx, cy, cz = c.camera_center(p)
    row = dict(job_id=p.job_id, seq=p.seq, ts=p.ts, frame=p.frame, cx=cx, cy=cy, cz=cz,
               focal=p.focal, confidence=p.confidence, step_dist=feats["step_dist"],
               turn_deg=feats["turn_deg"], conf_roll=_mean([q.confidence for q in window]),
               anomaly_score=None, is_anomaly=False, injected=p.injected)
    reason = None
    if scored and model is not None and len(window) >= c.MIN_WINDOW_TO_SCORE:
        row["anomaly_score"], row["is_anomaly"], reason = model.score(feats)
    return row, reason if row["is_anomaly"] else None


def training_row(window: Sequence[c.TrainingSnapshot], best_loss: float, peak_mem_gb: float,
                 model: Optional[AnomalyModel]) -> Tuple[Dict, Optional[str]]:
    """training_history row for window[-1] (TRAINING_METRICS + score) and the reason if flagged."""
    s = window[-1]
    feats = c.training_features(window)
    row = dict(job_id=s.job_id, seq=s.seq, ts=s.ts, step=s.step, loss=s.loss,
               num_gaussians=s.num_gaussians, gpu_mem_gb=s.gpu_mem_gb, steps_per_s=s.steps_per_s,
               loss_roll=_mean([w.loss for w in window]), loss_trend=feats["loss_trend"],
               gs_growth=feats["gs_growth"], best_loss=best_loss, peak_mem_gb=peak_mem_gb,
               anomaly_score=None, is_anomaly=False, injected=s.injected)
    reason = None
    if model is not None and len(window) >= c.MIN_WINDOW_TO_SCORE:
        row["anomaly_score"], row["is_anomaly"], reason = model.score(feats)
    return row, reason if row["is_anomaly"] else None


@dataclass
class JobUpdate:
    """What one poll batch changes on a jobs row; counters only for rows actually inserted."""
    poses_accepted: int = 0
    poses_rejected: int = 0
    anomalies: int = 0
    current_seq: int = -1
    current_step: Optional[int] = None
    current_loss: Optional[float] = None
    num_gaussians: Optional[int] = None

    def params(self, job_id: str, st: JobState) -> Dict:
        return dict(job_id=job_id, poses_accepted=self.poses_accepted,
                    poses_rejected=self.poses_rejected, anomalies=self.anomalies,
                    current_step=self.current_step, current_loss=self.current_loss,
                    num_gaussians=self.num_gaussians, best_loss=st.best_loss,
                    peak_gpu_mem_gb=st.peak_mem_gb, events_dropped=st.events_dropped())


# ---------------------------------------------------------------------------------------------
# PostgreSQL (rows 14, 15)
# ---------------------------------------------------------------------------------------------

POSE_COLUMNS = ("job_id", "seq", "ts", "frame", "cx", "cy", "cz", "focal", "confidence",
                "step_dist", "turn_deg", "conf_roll", "anomaly_score", "is_anomaly", "injected")
TRAINING_COLUMNS = ("job_id", "seq", "ts", "step", "loss", "num_gaussians", "gpu_mem_gb",
                    "steps_per_s", "loss_roll", "loss_trend", "gs_growth", "best_loss",
                    "peak_mem_gb", "anomaly_score", "is_anomaly", "injected")


def _insert_sql(table: str, cols: Sequence[str]) -> str:
    vals = ", ".join("to_timestamp(%(ts)s)" if k == "ts" else "%%(%s)s" % k for k in cols)
    return "INSERT INTO %s (%s) VALUES (%s) ON CONFLICT (job_id, seq) DO NOTHING" % (
        table, ", ".join(cols), vals)


_INSERT_POSE = _insert_sql("pose_history", POSE_COLUMNS)
_INSERT_TRAINING = _insert_sql("training_history", TRAINING_COLUMNS)

# LEAST/GREATEST skip NULLs in PostgreSQL, so the first value simply lands.
_UPDATE_JOB = """
UPDATE jobs SET updated_at = now(),
    current_step    = COALESCE(%(current_step)s, current_step),
    current_loss    = COALESCE(%(current_loss)s, current_loss),
    num_gaussians   = COALESCE(%(num_gaussians)s, num_gaussians),
    best_loss       = LEAST(best_loss, %(best_loss)s),
    peak_gpu_mem_gb = GREATEST(peak_gpu_mem_gb, %(peak_gpu_mem_gb)s),
    poses_accepted  = poses_accepted + %(poses_accepted)s,
    poses_rejected  = poses_rejected + %(poses_rejected)s,
    anomalies       = anomalies + %(anomalies)s,
    events_dropped  = GREATEST(events_dropped, %(events_dropped)s)
WHERE job_id = %(job_id)s
"""


class PgStore:
    """The processor's only view of PostgreSQL; one connection, one transaction per batch."""

    def __init__(self, conn):
        self.conn = conn

    @classmethod
    def connect(cls, dsn: str = c.DB_DSN) -> "PgStore":
        import psycopg

        return cls(psycopg.connect(dsn))

    def init_schema(self) -> None:
        self.conn.execute(c.DDL)
        self.conn.commit()

    def ensure_job(self, job_id: str) -> None:
        self.conn.execute("INSERT INTO jobs (job_id) VALUES (%s) ON CONFLICT DO NOTHING", (job_id,))

    def job_row(self, job_id: str) -> Optional[Dict]:
        cur = self.conn.execute("SELECT best_loss, peak_gpu_mem_gb, events_dropped FROM jobs "
                                "WHERE job_id = %s", (job_id,))
        r = cur.fetchone()
        return None if r is None else dict(best_loss=r[0], peak_gpu_mem_gb=r[1], events_dropped=r[2])

    def pose_tail(self, job_id: str, before_seq: int, n: int) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT seq, extract(epoch FROM ts)::float8, frame, cx, cy, cz, focal, confidence, "
            "turn_deg, injected FROM pose_history WHERE job_id = %s AND seq < %s "
            "ORDER BY seq DESC LIMIT %s", (job_id, before_seq, n))
        keys = ("seq", "ts", "frame", "cx", "cy", "cz", "focal", "confidence", "turn_deg", "injected")
        return [dict(zip(keys, r)) for r in reversed(cur.fetchall())]

    def training_tail(self, job_id: str, before_seq: int, n: int) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT seq, extract(epoch FROM ts)::float8, step, loss, num_gaussians, gpu_mem_gb, "
            "steps_per_s, injected FROM training_history WHERE job_id = %s AND seq < %s "
            "ORDER BY seq DESC LIMIT %s", (job_id, before_seq, n))
        keys = ("seq", "ts", "step", "loss", "num_gaussians", "gpu_mem_gb", "steps_per_s", "injected")
        return [dict(zip(keys, r)) for r in reversed(cur.fetchall())]

    def insert_pose(self, row: Dict) -> bool:
        return self.conn.execute(_INSERT_POSE, row).rowcount == 1

    def insert_training(self, row: Dict) -> bool:
        return self.conn.execute(_INSERT_TRAINING, row).rowcount == 1

    def update_job(self, params: Dict) -> None:
        self.conn.execute(_UPDATE_JOB, params)

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------------------------
# Processor core
# ---------------------------------------------------------------------------------------------

def _job_of_key(key) -> Optional[str]:
    if not key:
        return None
    try:
        return key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
    except UnicodeDecodeError:
        return None


def _num(v) -> Optional[float]:
    return None if v is None else float(v)


_INT4_MAX = 2 ** 31 - 1
_TS_MAX = 1e11  # year ~5138; to_timestamp() rejects far larger values


def _check_db_range(ev) -> None:
    """validate() bounds nothing above; a value PostgreSQL rejects would fail the whole batch and
    replay it forever (poison message), so it is a drop instead."""
    for name in ("seq", "step", "num_gaussians"):
        v = getattr(ev, name, 0)
        if v > _INT4_MAX:
            raise c.InvalidEvent("%s: %s=%r out of INTEGER range" % (c.KIND_OF[type(ev)], name, v))
    if not 0 <= ev.ts < _TS_MAX:
        raise c.InvalidEvent("%s: ts=%r out of range" % (c.KIND_OF[type(ev)], ev.ts))


class Processor:
    """Turns batches of (topic, key, value) into history rows, jobs updates and anomalies."""

    def __init__(self, store, sender=None, pose_model: Optional[AnomalyModel] = None,
                 training_model: Optional[AnomalyModel] = None):
        self.store = store
        self.sender = sender
        self.pose_model = pose_model
        self.training_model = training_model
        self.jobs: Dict[str, JobState] = {}
        self.events = 0
        self.inserted = 0
        self.anomalies = 0
        self.drops = 0

    # -- state ---------------------------------------------------------------------------------
    def _job(self, job_id: str) -> JobState:
        st = self.jobs.get(job_id)
        if st is None:
            st = JobState()
            self.store.ensure_job(job_id)
            r = self.store.job_row(job_id)
            if r is not None:  # restart: running min/max and drop floor carry over
                st.best_loss = _num(r["best_loss"])
                st.peak_mem_gb = _num(r["peak_gpu_mem_gb"])
                st.dropped_base = int(r["events_dropped"] or 0)
            self.jobs[job_id] = st
        return st

    def _preload_poses(self, st: JobState, job_id: str, before_seq: int) -> None:
        for r in self.store.pose_tail(job_id, before_seq, c.WINDOW):
            st.poses.append(RestoredPose(
                job_id=job_id, seq=r["seq"], ts=r["ts"], frame=r["frame"], qw=1.0, qx=0.0, qy=0.0,
                qz=0.0, tx=-(r["cx"] or 0.0), ty=-(r["cy"] or 0.0), tz=-(r["cz"] or 0.0),
                focal=r["focal"] or 1.0, confidence=r["confidence"] or 0.0,
                injected=r["injected"], turn_hint=r["turn_deg"] or 0.0))
        st.poses_loaded = True

    def _preload_snaps(self, st: JobState, job_id: str, before_seq: int) -> None:
        for r in self.store.training_tail(job_id, before_seq, c.WINDOW):
            st.snaps.append(c.TrainingSnapshot(
                job_id=job_id, seq=r["seq"], ts=r["ts"], step=r["step"], loss=r["loss"],
                num_gaussians=r["num_gaussians"] or 0, gpu_mem_gb=r["gpu_mem_gb"] or 0.0,
                steps_per_s=r["steps_per_s"], injected=r["injected"]))
        st.snaps_loaded = True

    # -- events --------------------------------------------------------------------------------
    def _drop(self, key, err: Exception, touched: Dict[str, JobUpdate]) -> None:
        self.drops += 1
        job_id = _job_of_key(key)
        if self.drops == 1 or self.drops % 100 == 0:
            logger.warning("[processor] dropped %d undecodable message(s); latest job=%s: %s",
                           self.drops, job_id, err)
        if job_id is not None:
            self._job(job_id).decode_drops += 1
            touched.setdefault(job_id, JobUpdate())

    def _anomaly(self, ev, kind: str, model: AnomalyModel, row: Dict, reason: str) -> c.Anomaly:
        # ts is the flagged event's own time, so a replayed tail re-sends identical messages.
        return c.Anomaly(job_id=ev.job_id, ts=ev.ts, source=kind, source_seq=ev.seq, model=model.name,
                         score=row["anomaly_score"], reason=reason, injected=ev.injected)

    def _pose(self, st: JobState, p: c.CameraPose, upd: JobUpdate, out: List[c.Anomaly]) -> None:
        if p.seq <= st.last_pose_seq:  # already handled (and committed) by this process
            return
        if not st.poses_loaded:
            self._preload_poses(st, p.job_id, p.seq)
        st.last_pose_seq = p.seq
        # reloc/map rows re-place older frames when a refit lands; keeping them out of the window
        # stops them from turning the next live pose into a spurious jump.
        scored = p.source in c.SCORED_SOURCES
        window = list(st.poses) + [p]
        if scored:
            st.poses.append(p)
        row, reason = pose_row(window, self.pose_model, scored)
        if self.store.insert_pose(row):
            self.inserted += 1
            if row["is_anomaly"]:
                upd.poses_rejected += 1
                upd.anomalies += 1
            else:
                upd.poses_accepted += 1
        if reason is not None:
            out.append(self._anomaly(p, "pose", self.pose_model, row, reason))

    def _training(self, st: JobState, s: c.TrainingSnapshot, upd: JobUpdate,
                  out: List[c.Anomaly]) -> None:
        if s.seq <= st.last_snap_seq:
            return
        if not st.snaps_loaded:
            self._preload_snaps(st, s.job_id, s.seq)
        st.last_snap_seq = s.seq
        st.snaps.append(s)
        st.best_loss = s.loss if st.best_loss is None else min(st.best_loss, s.loss)
        st.peak_mem_gb = s.gpu_mem_gb if st.peak_mem_gb is None else max(st.peak_mem_gb, s.gpu_mem_gb)
        row, reason = training_row(list(st.snaps), st.best_loss, st.peak_mem_gb, self.training_model)
        if self.store.insert_training(row):
            self.inserted += 1
            if row["is_anomaly"]:
                upd.anomalies += 1
            if s.seq > upd.current_seq:
                upd.current_seq, upd.current_step = s.seq, s.step
                upd.current_loss, upd.num_gaussians = s.loss, s.num_gaussians
        if reason is not None:
            out.append(self._anomaly(s, "training", self.training_model, row, reason))

    def handle(self, topic: str, key, value: bytes, touched: Dict[str, JobUpdate],
               out: List[c.Anomaly]) -> None:
        self.events += 1
        try:
            ev = c.decode(value)
            if not isinstance(ev, (c.CameraPose, c.TrainingSnapshot)):
                raise c.InvalidEvent("%s event on %s" % (c.KIND_OF[type(ev)], topic))
            _check_db_range(ev)
        except c.InvalidEvent as e:
            self._drop(key, e, touched)
            return
        st = self._job(ev.job_id)
        st.producer_dropped = max(st.producer_dropped, ev.dropped)
        upd = touched.setdefault(ev.job_id, JobUpdate())
        if isinstance(ev, c.CameraPose):
            self._pose(st, ev, upd, out)
        else:
            self._training(st, ev, upd, out)

    def process_batch(self, records: Iterable[Tuple[str, object, bytes]]) -> List[c.Anomaly]:
        """Handle one poll batch in one DB transaction; returns the anomalies it sent.

        Anomalies are sent after the commit (no alert for a row that rolled back) and for every
        flagged event, inserted or replayed: at-least-once on TOPIC_ANOMALIES, consumers can
        dedupe on (job_id, source, source_seq). If the sender drops one, this raises so the
        caller does not commit offsets and the replay sends it again.
        """
        touched: Dict[str, JobUpdate] = {}
        out: List[c.Anomaly] = []
        try:
            for topic, key, value in records:
                self.handle(topic, key, value, touched, out)
            for job_id, upd in touched.items():
                self.store.update_job(upd.params(job_id, self.jobs[job_id]))
            self.store.commit()
        except Exception:
            self.store.rollback()
            self.jobs.clear()  # windows/last seqs saw the rolled-back events; rebuild from the DB
            raise
        self.anomalies += len(out)
        if self.sender is not None and out:
            jobs = {a.job_id for a in out}
            before = {j: self.sender.dropped(j) for j in jobs}
            sent = [self.sender.send(a) for a in out]
            self.sender.flush()  # before the caller commits offsets
            if not all(sent) or any(self.sender.dropped(j) > before[j] for j in jobs):
                raise RuntimeError("[processor] %d anomaly message(s) not delivered; offsets not "
                                   "committed" % len(out))
        return out


# ---------------------------------------------------------------------------------------------
# Kafka loop
# ---------------------------------------------------------------------------------------------

def _kafka_consumer(on_rebalance):
    from kafka import ConsumerRebalanceListener, KafkaConsumer  # lazy, like producer.py

    class _Listener(ConsumerRebalanceListener):
        def on_partitions_revoked(self, revoked):
            on_rebalance()

        def on_partitions_assigned(self, assigned):
            on_rebalance()

    consumer = KafkaConsumer(bootstrap_servers=c.KAFKA_BOOTSTRAP, **c.CONSUMER_CONFIG)
    consumer.subscribe([c.TOPIC_POSES.name, c.TOPIC_TRAINING.name], listener=_Listener())
    return consumer


def _caught_up(consumer) -> bool:
    assigned = consumer.assignment()
    if not assigned:
        return False
    end = consumer.end_offsets(list(assigned))
    return all(consumer.position(tp) >= end[tp] for tp in assigned)


def run(once: bool = False, max_seconds: Optional[float] = None, log_every: float = LOG_EVERY_S,
        stop: Optional[threading.Event] = None, consumer=None, store=None, sender=None) -> Processor:
    stop = stop or threading.Event()
    pose_model = load_model(c.POSE_MODEL, c.POSE_FEATURES)
    training_model = load_model(c.TRAINING_MODEL, c.TRAINING_FEATURES)
    if store is None:
        store = PgStore.connect(c.DB_DSN)
    store.init_schema()
    if sender is None:
        from .producer import Sender

        sender = Sender()
    proc = Processor(store, sender, pose_model, training_model)
    # A partition that moves away and comes back may have been advanced by another member:
    # drop the per-job state on any rebalance and rebuild it lazily from the DB.
    consumer = consumer or _kafka_consumer(proc.jobs.clear)
    logger.info("[processor] consuming %s, %s as %s; DB %s", c.TOPIC_POSES.name,
                c.TOPIC_TRAINING.name, c.CONSUMER_GROUP, c.DB_DSN.rsplit("@", 1)[-1])
    t0 = last_log = time.time()
    last_events = 0
    try:
        while not stop.is_set():
            polled = consumer.poll(timeout_ms=POLL_MS, max_records=MAX_RECORDS)
            records = [(r.topic, r.key, r.value) for recs in polled.values() for r in recs]
            now = time.time()
            if records:
                proc.process_batch(records)
                consumer.commit()  # only after the DB commit (row 7)
            elif once and (_caught_up(consumer) or
                           (not consumer.assignment() and now - t0 > ONCE_WAIT_ASSIGN_S)):
                break
            if now - last_log >= max(log_every, 0.1):
                logger.info("[processor] %.1f events/s, %d events, %d inserted, %d anomalies, %d drops",
                            (proc.events - last_events) / (now - last_log), proc.events,
                            proc.inserted, proc.anomalies, proc.drops)
                last_log, last_events = now, proc.events
            if max_seconds is not None and now - t0 >= max_seconds:
                break
    finally:
        logger.info("[processor] stopping: %d events, %d inserted, %d anomalies, %d drops",
                    proc.events, proc.inserted, proc.anomalies, proc.drops)
        for close in (lambda: consumer.close(autocommit=False), lambda: sender.close(2.0), store.close):
            try:
                close()
            except Exception as e:
                logger.warning("[processor] close failed: %s", e)
    return proc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true", help="exit once caught up with the topics")
    ap.add_argument("--max-seconds", type=float, default=None, help="exit after this many seconds")
    ap.add_argument("--log-every", type=float, default=LOG_EVERY_S, help="throughput log period, s")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()

    def _stop(signum, _frame):
        logger.info("[processor] %s: finishing the current batch", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    run(once=args.once, max_seconds=args.max_seconds, log_every=args.log_every, stop=stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
