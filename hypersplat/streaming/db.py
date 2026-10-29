"""PostgreSQL access for the streaming mini-project.

This module keeps SQL in one place so processor/dashboard stay small.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from . import contract as c


def _connect():
    import psycopg

    return psycopg.connect(c.DB_DSN, autocommit=False)


def init_db() -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(c.DDL)
        conn.commit()


def _utc_ts(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class Writer:
    """Idempotent writes used by the processor."""

    def __init__(self):
        self._conn = _connect()

    def close(self) -> None:
        self._conn.close()

    def _ensure_job(self, job_id: str, event_dropped: int = 0) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (job_id, events_dropped)
                VALUES (%s, %s)
                ON CONFLICT (job_id) DO UPDATE
                SET updated_at = now(),
                    events_dropped = GREATEST(jobs.events_dropped, EXCLUDED.events_dropped)
                """,
                (job_id, event_dropped),
            )

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def write_pose(
        self,
        pose: c.CameraPose,
        metrics: Dict[str, float],
        score: Optional[float],
        is_anomaly: bool,
    ) -> bool:
        self._ensure_job(pose.job_id, pose.dropped)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pose_history (
                    job_id, seq, ts, frame, cx, cy, cz, focal, confidence,
                    step_dist, turn_deg, conf_roll, anomaly_score, is_anomaly, injected
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (job_id, seq) DO NOTHING
                """,
                (
                    pose.job_id,
                    pose.seq,
                    _utc_ts(pose.ts),
                    pose.frame,
                    metrics.get("cx"),
                    metrics.get("cy"),
                    metrics.get("cz"),
                    pose.focal,
                    pose.confidence,
                    metrics.get("step_dist"),
                    metrics.get("turn_deg"),
                    metrics.get("conf_roll"),
                    score,
                    is_anomaly,
                    pose.injected,
                ),
            )
            inserted = cur.rowcount > 0
            if inserted:
                cur.execute(
                    """
                    UPDATE jobs
                    SET updated_at = now(),
                        poses_accepted = jobs.poses_accepted + 1,
                        poses_rejected = jobs.poses_rejected + %s,
                        anomalies = jobs.anomalies + %s,
                        events_dropped = GREATEST(jobs.events_dropped, %s)
                    WHERE job_id = %s
                    """,
                    (
                        1 if is_anomaly else 0,
                        1 if is_anomaly else 0,
                        pose.dropped,
                        pose.job_id,
                    ),
                )
            return inserted

    def write_training(
        self,
        snap: c.TrainingSnapshot,
        metrics: Dict[str, float],
        score: Optional[float],
        is_anomaly: bool,
    ) -> bool:
        self._ensure_job(snap.job_id, snap.dropped)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_history (
                    job_id, seq, ts, step, loss, num_gaussians, gpu_mem_gb, steps_per_s,
                    loss_roll, loss_trend, gs_growth, best_loss, peak_mem_gb,
                    anomaly_score, is_anomaly, injected
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (job_id, seq) DO NOTHING
                """,
                (
                    snap.job_id,
                    snap.seq,
                    _utc_ts(snap.ts),
                    snap.step,
                    snap.loss,
                    snap.num_gaussians,
                    snap.gpu_mem_gb,
                    snap.steps_per_s,
                    metrics.get("loss_roll"),
                    metrics.get("loss_trend"),
                    metrics.get("gs_growth"),
                    metrics.get("best_loss"),
                    metrics.get("peak_mem_gb"),
                    score,
                    is_anomaly,
                    snap.injected,
                ),
            )
            inserted = cur.rowcount > 0
            if inserted:
                cur.execute(
                    """
                    UPDATE jobs
                    SET updated_at = now(),
                        current_step = %s,
                        current_loss = %s,
                        best_loss = CASE
                            WHEN jobs.best_loss IS NULL THEN %s
                            ELSE LEAST(jobs.best_loss, %s)
                        END,
                        num_gaussians = %s,
                        peak_gpu_mem_gb = CASE
                            WHEN jobs.peak_gpu_mem_gb IS NULL THEN %s
                            ELSE GREATEST(jobs.peak_gpu_mem_gb, %s)
                        END,
                        anomalies = jobs.anomalies + %s,
                        events_dropped = GREATEST(jobs.events_dropped, %s)
                    WHERE job_id = %s
                    """,
                    (
                        snap.step,
                        snap.loss,
                        snap.loss,
                        snap.loss,
                        snap.num_gaussians,
                        snap.gpu_mem_gb,
                        snap.gpu_mem_gb,
                        1 if is_anomaly else 0,
                        snap.dropped,
                        snap.job_id,
                    ),
                )
            return inserted


def _fetch_all(query: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def active_jobs() -> List[Dict[str, Any]]:
    return _fetch_all(
        """
        SELECT *
        FROM jobs
        WHERE archived = false
        ORDER BY updated_at DESC
        """,
        (),
    )


def latest_job_id() -> Optional[str]:
    jobs = active_jobs()
    return jobs[0]["job_id"] if jobs else None


def pose_tail(job_id: str, limit: int = c.DASHBOARD_ROWS) -> List[Dict[str, Any]]:
    return _fetch_all(
        """
        SELECT *
        FROM pose_history
        WHERE job_id = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (job_id, int(limit)),
    )


def training_tail(job_id: str, limit: int = c.DASHBOARD_ROWS) -> List[Dict[str, Any]]:
    return _fetch_all(
        """
        SELECT *
        FROM training_history
        WHERE job_id = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (job_id, int(limit)),
    )


def health_rows() -> List[Dict[str, Any]]:
    return _fetch_all(c.HEALTH_QUERY, ())
