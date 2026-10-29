"""What the streaming processor has concluded about the run, for the 8090 telemetry view.

The processor (hypersplat.streaming) scores every pose and training snapshot and writes the result to
PostgreSQL. This reads the newest job's flagged rows, so the view shows the models' runtime output as
it arrives, not stored or hard-coded values. Any failure (no database, no job yet) comes back as
``available: False`` with the reason, never as zero flags.
"""

import os
import threading
import time
from typing import Dict, List, Optional

FLAGS_SHOWN = 20
CONNECT_TIMEOUT_S = 1
RETRY_AFTER_S = 5.0     # after a failed read, do not hammer a dead database


class StreamedResults:
    def __init__(self, dsn: Optional[str] = None, job_id: Optional[str] = None, ttl: float = 1.5):
        self.dsn, self.ttl = dsn, ttl
        self.job_id = job_id or os.environ.get("HYPERSPLAT_JOB") or None
        self._lock = threading.Lock()
        self._cache = (0.0, self._down("not read yet"))

    @staticmethod
    def _down(reason: str) -> Dict:
        return {"available": False, "reason": reason, "job_id": None, "poses_scored": 0, "poses_flagged": 0,
                "training_flagged": 0, "age_s": None, "flags": [], "flagged_frames": []}

    def snapshot(self) -> Dict:
        with self._lock:
            t, cached = self._cache
            wait = RETRY_AFTER_S if not cached["available"] and cached["reason"].startswith("database") else self.ttl
            if time.time() - t < wait:
                return cached
            try:
                cached = self._read()
            except Exception as e:  # noqa: BLE001  psycopg missing, connection refused, schema absent...
                cached = self._down("database unreachable (%s)" % type(e).__name__)
            self._cache = (time.time(), cached)
            return cached

    def _read(self) -> Dict:
        import psycopg

        from hypersplat.streaming import contract as c

        with psycopg.connect(self.dsn or c.DB_DSN, connect_timeout=CONNECT_TIMEOUT_S, autocommit=True) as cn:
            job = self.job_id or (cn.execute("SELECT job_id FROM jobs WHERE NOT archived "
                                             "ORDER BY updated_at DESC LIMIT 1").fetchone() or [None])[0]
            if job is None:
                return self._down("no job in the database yet")
            scored, flagged = cn.execute("SELECT count(anomaly_score), count(*) FILTER (WHERE is_anomaly) "
                                         "FROM pose_history WHERE job_id = %s", (job,)).fetchone()
            train_flagged = cn.execute("SELECT count(*) FROM training_history WHERE job_id = %s AND is_anomaly",
                                       (job,)).fetchone()[0]
            age = cn.execute("SELECT extract(epoch FROM now() - updated_at) FROM jobs WHERE job_id = %s",
                             (job,)).fetchone()
            flags: List[Dict] = []
            for seq, frame, score, inj in cn.execute(
                    "SELECT seq, frame, anomaly_score, injected FROM pose_history WHERE job_id = %s AND is_anomaly "
                    "ORDER BY seq DESC LIMIT %s", (job, FLAGS_SHOWN)):
                flags.append({"source": "pose", "seq": seq, "frame": frame, "score": score, "injected": inj,
                              "reason": "pose model flagged %s (score %.3f)" % (frame, score if score is not None else 0.0)})
            for seq, step, score, inj in cn.execute(
                    "SELECT seq, step, anomaly_score, injected FROM training_history WHERE job_id = %s AND is_anomaly "
                    "ORDER BY seq DESC LIMIT %s", (job, FLAGS_SHOWN)):
                flags.append({"source": "training", "seq": seq, "step": step, "score": score, "injected": inj,
                              "reason": "training model flagged step %d (score %.3f)" % (step, score if score is not None else 0.0)})
            frames = [r[0] for r in cn.execute("SELECT frame FROM pose_history WHERE job_id = %s AND is_anomaly "
                                               "ORDER BY seq", (job,))]
        return {"available": True, "reason": "", "job_id": job, "poses_scored": scored, "poses_flagged": flagged,
                "training_flagged": train_flagged, "age_s": None if age is None or age[0] is None else float(age[0]),
                "flags": flags[:FLAGS_SHOWN], "flagged_frames": frames}
