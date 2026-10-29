"""Processor unit tests (pure logic on an in-memory store) plus PostgreSQL / Kafka integration
tests that skip when the local infra (ops/infra.sh) is not reachable."""

import json
import os
import socket
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from hypersplat.streaming import contract as c
from hypersplat.streaming import processor as P
from hypersplat.streaming.producer import FakeSender

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

class MemoryStore:
    """Same interface and ON CONFLICT semantics as PgStore, in dicts; commit/rollback snapshot."""

    def __init__(self):
        self.jobs, self.poses, self.snaps = {}, {}, {}
        self._saved = None
        self.commit()

    def init_schema(self):
        pass

    def ensure_job(self, job_id):
        self.jobs.setdefault(job_id, dict(best_loss=None, peak_gpu_mem_gb=None, events_dropped=0,
                                          poses_accepted=0, poses_rejected=0, anomalies=0,
                                          current_step=None, current_loss=None, num_gaussians=None))

    def job_row(self, job_id):
        return dict(self.jobs[job_id]) if job_id in self.jobs else None

    def _tail(self, table, job_id, before, n):
        rows = sorted((r for (j, s), r in table.items() if j == job_id and s < before),
                      key=lambda r: r["seq"])
        return [dict(r) for r in rows[-n:]]

    def pose_tail(self, job_id, before_seq, n):
        return self._tail(self.poses, job_id, before_seq, n)

    def training_tail(self, job_id, before_seq, n):
        return self._tail(self.snaps, job_id, before_seq, n)

    def _insert(self, table, row):
        assert row["job_id"] in self.jobs  # the FK
        k = (row["job_id"], row["seq"])
        if k in table:
            return False
        table[k] = dict(row)
        return True

    def insert_pose(self, row):
        assert set(row) == set(P.POSE_COLUMNS)
        return self._insert(self.poses, row)

    def insert_training(self, row):
        assert set(row) == set(P.TRAINING_COLUMNS)
        return self._insert(self.snaps, row)

    def update_job(self, p):
        j = self.jobs[p["job_id"]]
        for k in ("current_step", "current_loss", "num_gaussians"):
            if p[k] is not None:
                j[k] = p[k]
        if p["best_loss"] is not None:
            j["best_loss"] = p["best_loss"] if j["best_loss"] is None else min(j["best_loss"], p["best_loss"])
        if p["peak_gpu_mem_gb"] is not None:
            j["peak_gpu_mem_gb"] = max(j["peak_gpu_mem_gb"] or 0, p["peak_gpu_mem_gb"])
        for k in ("poses_accepted", "poses_rejected", "anomalies"):
            j[k] += p[k]
        j["events_dropped"] = max(j["events_dropped"], p["events_dropped"])

    def commit(self):
        import copy
        self._saved = copy.deepcopy((self.jobs, self.poses, self.snaps))

    def rollback(self):
        import copy
        self.jobs, self.poses, self.snaps = copy.deepcopy(self._saved)

    def close(self):
        pass


def pose(seq, x=None, conf=1000.0, job="job1", source="pose", dropped=0, injected=None, turn=0.0):
    """Camera at (x, 0, 0) (default 0.1 m per seq), yawed by ``turn`` degrees about y."""
    import math
    x = seq * 0.1 if x is None else x
    h = math.radians(turn) / 2
    qw, qy = math.cos(h), math.sin(h)
    R = c._rotmat((qw, 0.0, qy, 0.0))
    C = (x, 0.0, 0.0)
    t = [-sum(R[r][k] * C[k] for k in range(3)) for r in range(3)]  # t = -R C
    return c.CameraPose(job, seq, 1000.0 + seq, "f%05d.jpg" % seq, qw, 0.0, qy, 0.0, *t,
                        500.0, conf, dropped=dropped, injected=injected, source=source)


def bad(seq, x=60.0, **kw):
    """A mis-registered frame: far jump, big turn, few inliers."""
    return pose(seq, x=x, turn=120.0, conf=50.0, **kw)


def snap(seq, loss=None, job="job1", mem=2.0, dropped=0):
    loss = 1.0 / (1 + seq) if loss is None else loss
    return c.TrainingSnapshot(job, seq, 1000.0 + seq, seq * 10, loss, 1000 + 10 * seq, mem,
                              dropped=dropped)


def rec(ev):
    topic, key, value = c.encode(ev)
    return topic, key, value


def fit_pose_model(model_dir, with_stats=False, as_dict=False):
    """Tiny IsolationForest on windows of normal poses, saved where contract.model_path looks."""
    import random

    import joblib
    import numpy as np
    from sklearn.ensemble import IsolationForest

    rnd = random.Random(0)
    poses, x = [], 0.0
    for i in range(400):
        x += 0.1 + rnd.gauss(0, 0.02)
        poses.append(pose(i, x=x, conf=1000 + rnd.gauss(0, 30), turn=rnd.gauss(0, 0.5)))
    X = np.array([[c.pose_features(poses[max(0, i - c.WINDOW + 1):i + 1])[f] for f in c.POSE_FEATURES]
                  for i in range(c.MIN_WINDOW_TO_SCORE - 1, len(poses))])
    est = IsolationForest(n_estimators=50, contamination=0.05, random_state=0).fit(X)
    obj = est
    if as_dict or with_stats:
        obj = {"model": est, "features": list(c.POSE_FEATURES)}
        if with_stats:
            obj.update(mean=X.mean(0).tolist(), std=X.std(0).tolist())
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(obj, os.path.join(str(model_dir), c.POSE_MODEL + ".joblib"))


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERSPLAT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(c, "MODEL_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------------------------
# metrics and windows (rows 8, 9)
# ---------------------------------------------------------------------------------------------

def test_pose_metrics_metrics_only_without_model(model_dir):
    assert P.load_model(c.POSE_MODEL, c.POSE_FEATURES) is None
    store = MemoryStore()
    proc = P.Processor(store)
    proc.process_batch([rec(pose(i, conf=100.0 * (i + 1))) for i in range(10)])
    r = store.poses[("job1", 9)]
    assert r["step_dist"] == pytest.approx(0.1) and r["turn_deg"] == pytest.approx(0.0, abs=1e-6)
    assert (r["cx"], r["cy"], r["cz"]) == pytest.approx((0.9, 0.0, 0.0))
    assert r["conf_roll"] == pytest.approx(550.0)  # mean of 100..1000
    assert r["anomaly_score"] is None and r["is_anomaly"] is False
    assert store.jobs["job1"]["poses_accepted"] == 10 and store.jobs["job1"]["poses_rejected"] == 0
    assert len(proc.jobs["job1"].poses) == 10


def test_window_is_bounded(monkeypatch):
    store = MemoryStore()
    proc = P.Processor(store)
    proc.process_batch([rec(pose(i)) for i in range(c.WINDOW + 7)])
    proc.process_batch([rec(snap(i)) for i in range(c.WINDOW + 7)])
    assert len(proc.jobs["job1"].poses) == c.WINDOW and len(proc.jobs["job1"].snaps) == c.WINDOW


def test_turn_deg_metric():
    store = MemoryStore()
    P.Processor(store).process_batch([rec(pose(0)), rec(pose(1, turn=30.0))])
    assert store.poses[("job1", 1)]["turn_deg"] == pytest.approx(30.0)


def test_training_metrics_and_job_snapshot():
    store = MemoryStore()
    losses = [0.5, 0.4, 0.45, 0.3, 0.35]
    mems = [1.0, 3.0, 2.0, 2.5, 2.0]
    P.Processor(store).process_batch([rec(snap(i, loss=l, mem=m)) for i, (l, m) in enumerate(zip(losses, mems))])
    r = store.snaps[("job1", 4)]
    assert r["best_loss"] == 0.3 and r["peak_mem_gb"] == 3.0
    assert r["loss_roll"] == pytest.approx(sum(losses) / 5)
    w = [snap(i, loss=l, mem=m) for i, (l, m) in enumerate(zip(losses, mems))]
    feats = c.training_features(w)
    assert r["loss_trend"] == pytest.approx(feats["loss_trend"]) and r["gs_growth"] == pytest.approx(1.0)
    assert store.snaps[("job1", 0)]["best_loss"] == 0.5 and store.snaps[("job1", 1)]["peak_mem_gb"] == 3.0
    j = store.jobs["job1"]
    assert (j["current_step"], j["current_loss"], j["num_gaussians"]) == (40, 0.35, 1040)
    assert j["best_loss"] == 0.3 and j["peak_gpu_mem_gb"] == 3.0


def test_unscored_sources_stay_out_of_the_window(model_dir):
    fit_pose_model(model_dir)
    store = MemoryStore()
    proc = P.Processor(store, pose_model=P.load_model(c.POSE_MODEL, c.POSE_FEATURES))
    recs = [rec(pose(i)) for i in range(10)]
    recs.append(rec(pose(10, x=500.0, source="map")))  # a refit re-placing an old frame far away
    recs.append(rec(pose(11, x=1.0)))
    proc.process_batch(recs)
    assert store.poses[("job1", 10)]["anomaly_score"] is None
    assert not store.poses[("job1", 10)]["is_anomaly"]
    assert store.poses[("job1", 11)]["step_dist"] == pytest.approx(0.1)  # measured from seq 9
    assert [p.seq for p in proc.jobs["job1"].poses][-2:] == [9, 11]


# ---------------------------------------------------------------------------------------------
# scoring (row 10)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("with_stats", [False, True])
def test_jump_is_flagged_and_sent(model_dir, with_stats):
    fit_pose_model(model_dir, with_stats=with_stats)
    model = P.load_model(c.POSE_MODEL, c.POSE_FEATURES)
    assert model is not None and (model.mean is not None) == with_stats
    store, sender = MemoryStore(), FakeSender()
    proc = P.Processor(store, sender, pose_model=model)
    recs = [rec(pose(i)) for i in range(c.MIN_WINDOW_TO_SCORE - 1)]
    out = proc.process_batch(recs)
    assert not out and all(store.poses[("job1", i)]["anomaly_score"] is None for i in range(4))
    recs = [rec(pose(i)) for i in range(4, 20)] + [rec(bad(20, injected="shuffle"))]
    out = proc.process_batch(recs)
    assert store.poses[("job1", 4)]["anomaly_score"] is not None  # window reached MIN_WINDOW_TO_SCORE
    row = store.poses[("job1", 20)]
    assert row["is_anomaly"] and row["anomaly_score"] < 0 and row["injected"] == "shuffle"
    assert [a.source_seq for a in out] == [20]
    a = out[0]
    assert (a.source, a.model, a.injected) == ("pose", c.POSE_MODEL, "shuffle")
    assert a.reason.startswith("step_ratio=")
    assert a.score == pytest.approx(row["anomaly_score"])
    sender.flush()
    assert [e.source_seq for e in sender.events()] == [20]
    assert sender.messages[0][0] == c.TOPIC_ANOMALIES.name
    j = store.jobs["job1"]
    assert (j["poses_rejected"], j["anomalies"], j["poses_accepted"]) == (1, 1, 20)
    sender.close()


def test_reason_uses_z_scores_when_stats_stored():
    class Est:
        def decision_function(self, X):
            return [-0.1]

        def predict(self, X):
            return [-1]

    feats = dict(step_ratio=5.0, turn_deg=3.0, conf_ratio=1.0, focal_ratio=1.0)
    plain = P.AnomalyModel("m", Est(), c.POSE_FEATURES)
    assert plain.score(feats) == (-0.1, True, "step_ratio=5")
    z = P.AnomalyModel("m", Est(), c.POSE_FEATURES, mean=[4.0, 0.0, 1.0, 1.0],
                       std=[1.0, 0.5, 0.1, 0.1])
    assert z.score(feats)[2] == "turn_deg=3"


def test_feature_stats_layouts():
    f = ("a", "b")
    assert P._feature_stats({"mean": [1, 2], "std": [3, 4]}, None, f) == ([1, 2], [3, 4])
    assert P._feature_stats({"features": ["b", "a"], "mean": [2, 1], "std": [4, 3]}, None, f) == ([1, 2], [3, 4])
    assert P._feature_stats({"stats": {"a": {"mean": 1, "std": 3}, "b": {"mean": 2, "std": 4}}},
                            None, f) == ([1, 2], [3, 4])
    assert P._feature_stats({}, object(), f) == (None, None)


def test_unusable_model_file_means_metrics_only(model_dir):
    import joblib
    joblib.dump({"nothing": 1}, os.path.join(str(model_dir), c.TRAINING_MODEL + ".joblib"))
    assert P.load_model(c.TRAINING_MODEL, c.TRAINING_FEATURES) is None


def test_training_model_scores_snapshots(model_dir):
    import joblib
    import numpy as np
    from sklearn.ensemble import IsolationForest

    rnd = np.random.default_rng(0)
    # many short runs, so the short windows at the start of a job are well represented too
    runs = [[snap(i, loss=0.5 * (1 + 0.05 * rnd.standard_normal())) for i in range(40)] for _ in range(20)]
    X = np.array([[c.training_features(run[max(0, i - c.WINDOW + 1):i + 1])[k] for k in c.TRAINING_FEATURES]
                  for run in runs for i in range(c.MIN_WINDOW_TO_SCORE - 1, len(run))])
    train = runs[0]
    joblib.dump(IsolationForest(n_estimators=50, contamination=0.05, random_state=0).fit(X),
                c.model_path(c.TRAINING_MODEL))
    store = MemoryStore()
    proc = P.Processor(store, training_model=P.load_model(c.TRAINING_MODEL, c.TRAINING_FEATURES))
    out = proc.process_batch([rec(s) for s in train[:20]] + [rec(snap(20, loss=5.0))])
    assert store.snaps[("job1", 19)]["anomaly_score"] is not None
    assert out[-1].source_seq == 20 and out[-1].source == "training"
    assert out[-1].reason.startswith("loss_ratio_best=")
    scores = [r["anomaly_score"] for r in store.snaps.values() if r["anomaly_score"] is not None]
    assert out[-1].score == min(scores) == store.snaps[("job1", 20)]["anomaly_score"]
    assert store.jobs["job1"]["anomalies"] == len(out) and store.jobs["job1"]["poses_rejected"] == 0


# ---------------------------------------------------------------------------------------------
# drops, idempotency and restart (rows 4, 7)
# ---------------------------------------------------------------------------------------------

def test_undecodable_messages_count_as_drops_for_their_job():
    store = MemoryStore()
    proc = P.Processor(store)
    recs = [rec(pose(0, dropped=2)), (c.TOPIC_POSES.name, b"job1", b"{not json"),
            (c.TOPIC_POSES.name, b"job1", json.dumps({"kind": "pose", "job_id": "job1"}).encode()),
            (c.TOPIC_POSES.name, None, b"garbage"),  # no key: no job to charge it to
            rec(pose(1, dropped=5)), rec(snap(0, dropped=3))]
    proc.process_batch(recs)
    assert store.jobs["job1"]["events_dropped"] == 5 + 2
    assert proc.drops == 3 and len(store.poses) == 2
    # a job only ever seen through garbage still gets its jobs row
    proc.process_batch([(c.TOPIC_TRAINING.name, b"job2", b"\xff")])
    assert store.jobs["job2"]["events_dropped"] == 1


def test_replayed_tail_after_restart_changes_nothing(model_dir):
    fit_pose_model(model_dir)
    model = P.load_model(c.POSE_MODEL, c.POSE_FEATURES)
    batch = [rec(pose(i)) for i in range(15)] + [rec(bad(15, x=80.0))] + \
            [rec(snap(i)) for i in range(15)] + [rec(pose(16, x=80.1, dropped=4))]
    store = MemoryStore()
    P.Processor(store, pose_model=model).process_batch(batch)
    before = (dict(store.jobs["job1"]), dict(store.poses), dict(store.snaps))
    assert before[0]["poses_rejected"] >= 1 and before[0]["events_dropped"] == 4
    # crash between DB commit and offset commit: a fresh process replays the whole batch
    out = P.Processor(store, pose_model=model).process_batch(batch)
    assert (dict(store.jobs["job1"]), dict(store.poses), dict(store.snaps)) == before
    assert out  # anomalies are re-sent (at-least-once on the anomalies topic)


def test_restart_rebuilds_windows_from_history():
    straight = MemoryStore()
    ref = P.Processor(straight)
    ref.process_batch([rec(pose(i, conf=1000.0 + i)) for i in range(40)] +
                      [rec(snap(i, mem=1.0 + (i % 7))) for i in range(40)])
    ref.process_batch([rec(pose(40, conf=2000.0)), rec(snap(40, loss=0.001, mem=0.5))])

    store = MemoryStore()
    P.Processor(store).process_batch([rec(pose(i, conf=1000.0 + i)) for i in range(40)] +
                                     [rec(snap(i, mem=1.0 + (i % 7))) for i in range(40)])
    proc = P.Processor(store)  # restart
    proc.process_batch([rec(pose(40, conf=2000.0)), rec(snap(40, loss=0.001, mem=0.5))])
    st = proc.jobs["job1"]
    assert [p.seq for p in st.poses] == list(range(41 - c.WINDOW, 41))
    assert [s.seq for s in st.snaps] == list(range(41 - c.WINDOW, 41))
    for key in ("step_dist", "turn_deg", "conf_roll", "cx"):
        assert store.poses[("job1", 40)][key] == pytest.approx(straight.poses[("job1", 40)][key])
    for key in ("loss_roll", "loss_trend", "gs_growth", "best_loss", "peak_mem_gb"):
        assert store.snaps[("job1", 40)][key] == pytest.approx(straight.snaps[("job1", 40)][key])
    assert store.jobs["job1"] == straight.jobs["job1"]


def test_restart_mid_replay_preloads_only_older_rows():
    store = MemoryStore()
    P.Processor(store).process_batch([rec(pose(i)) for i in range(20)])
    proc = P.Processor(store)
    proc.process_batch([rec(pose(i)) for i in range(15, 21)])  # replay 15..19, then new 20
    assert [p.seq for p in proc.jobs["job1"].poses] == list(range(21))
    assert store.jobs["job1"]["poses_accepted"] == 21


def test_failed_batch_rolls_back():
    class Boom(MemoryStore):
        def insert_training(self, row):
            raise RuntimeError("db down")

    store = Boom()
    proc = P.Processor(store)
    with pytest.raises(RuntimeError):
        proc.process_batch([rec(pose(0)), rec(snap(0))])
    assert not store.poses and not store.jobs and not proc.jobs


def test_values_postgres_would_reject_are_drops_not_poison():
    store = MemoryStore()
    proc = P.Processor(store)
    proc.process_batch([rec(pose(2 ** 31)), rec(c.TrainingSnapshot("job1", 0, 1e15, 0, 0.1, 1, 1.0))])
    assert proc.drops == 2 and not store.poses and not store.snaps
    assert store.jobs["job1"]["events_dropped"] == 2


def test_anomaly_delivery_failure_blocks_offset_commit(model_dir):
    fit_pose_model(model_dir)

    class Down(FakeSender):
        def send(self, event):
            self.drop(event.job_id, "broker down")
            return False

    store = MemoryStore()
    proc = P.Processor(store, Down(), pose_model=P.load_model(c.POSE_MODEL, c.POSE_FEATURES))
    with pytest.raises(RuntimeError, match="not delivered"):
        proc.process_batch([rec(pose(i)) for i in range(10)] + [rec(bad(10))])
    assert ("job1", 10) in store.poses  # the DB side is committed; the replay re-sends


class RecordingEst:
    """Picklable stand-in estimator that records what it was asked to score."""
    seen = []

    def decision_function(self, X):
        RecordingEst.seen.append(list(X[0]))
        return [0.1]

    def predict(self, X):
        return [1]


def test_saved_feature_order_is_respected(model_dir):
    import joblib

    RecordingEst.seen.clear()
    order = list(reversed(c.POSE_FEATURES))
    joblib.dump({"model": RecordingEst(), "features": order}, c.model_path(c.POSE_MODEL))
    m = P.load_model(c.POSE_MODEL, c.POSE_FEATURES)
    feats = {f: float(i) for i, f in enumerate(c.POSE_FEATURES)}
    m.score(feats)
    assert RecordingEst.seen == [[feats[f] for f in order]]
    joblib.dump({"model": RecordingEst(), "features": ["other"]}, c.model_path(c.POSE_MODEL))
    assert P.load_model(c.POSE_MODEL, c.POSE_FEATURES) is None


# ---------------------------------------------------------------------------------------------
# integration (skipped without the local infra)
# ---------------------------------------------------------------------------------------------

def _port_open(hostport):
    host, port = hostport.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except OSError:
        return False


def _pg_or_skip():
    psycopg = pytest.importorskip("psycopg")
    try:
        psycopg.connect(c.DB_DSN, connect_timeout=2).close()
    except Exception as e:
        pytest.skip("PostgreSQL unreachable: %s" % e)
    return psycopg


@pytest.fixture
def pg_schema():
    """A throwaway schema in the configured database; yields a DSN whose search_path points at it."""
    psycopg = _pg_or_skip()
    schema = "t_" + uuid.uuid4().hex[:12]
    with psycopg.connect(c.DB_DSN, autocommit=True) as conn:
        conn.execute("CREATE SCHEMA %s" % schema)
    dsn = c.DB_DSN + ("&" if "?" in c.DB_DSN else "?") + "options=-csearch_path%3D" + schema
    yield dsn
    with psycopg.connect(c.DB_DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA %s CASCADE" % schema)


def test_pgstore_roundtrip_and_idempotency(pg_schema, model_dir):
    fit_pose_model(model_dir)
    model = P.load_model(c.POSE_MODEL, c.POSE_FEATURES)
    batch = [rec(pose(i)) for i in range(12)] + [rec(bad(12, x=70.0, dropped=1))] + \
            [rec(snap(i)) for i in range(8)]
    store = P.PgStore.connect(pg_schema)
    store.init_schema()
    store.init_schema()  # idempotent DDL
    P.Processor(store, pose_model=model).process_batch(batch)
    store.close()
    store = P.PgStore.connect(pg_schema)
    P.Processor(store, pose_model=model).process_batch(batch + [rec(snap(8, loss=0.01))])
    conn = store.conn
    assert conn.execute("SELECT count(*) FROM pose_history").fetchone()[0] == 13
    assert conn.execute("SELECT count(*) FROM training_history").fetchone()[0] == 9
    row = conn.execute("SELECT cx, step_dist, is_anomaly, extract(epoch FROM ts) FROM pose_history "
                       "WHERE seq = 12").fetchone()
    assert row[0] == pytest.approx(70.0) and row[2] is True and float(row[3]) == pytest.approx(1012.0)
    j = conn.execute("SELECT poses_accepted, poses_rejected, anomalies, events_dropped, current_step, "
                     "current_loss, best_loss, num_gaussians, peak_gpu_mem_gb FROM jobs").fetchone()
    assert j == (12, 1, 1, 1, 80, 0.01, 0.01, 1080, 2.0)
    store.close()


def test_kafka_to_postgres_end_to_end(pg_schema, tmp_path):
    """Producer -> Kafka -> processor --once -> PostgreSQL, in subprocesses with private topics."""
    _pg_or_skip()
    pytest.importorskip("kafka")
    if not _port_open(c.KAFKA_BOOTSTRAP.split(",")[0]):
        pytest.skip("Kafka unreachable at %s" % c.KAFKA_BOOTSTRAP)
    tag = uuid.uuid4().hex[:8]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), HYPERSPLAT_TOPIC_PREFIX="t%s." % tag,
               HYPERSPLAT_CONSUMER_GROUP="t%s" % tag, HYPERSPLAT_DB_DSN=pg_schema,
               HYPERSPLAT_MODEL_DIR=str(tmp_path / "none"))
    produce = textwrap.dedent("""
        from hypersplat.streaming import contract as c
        from hypersplat.streaming.producer import Sender
        s = Sender()
        for i in range(30):
            s.send(c.CameraPose("kjob", i, 1.0 + i, "f%d.jpg" % i, 1, 0, 0, 0, -0.1 * i, 0, 0, 500, 900))
            s.send(c.TrainingSnapshot("kjob", i, 1.0 + i, 10 * i, 1.0 / (i + 1), 100 + i, 2.0))
        s.close(10)
        assert s.sent == 60, s.sent
    """)
    # The pod's broker has auto-create off (topics come only from the topics step), so create the
    # private prefixed topics the way the stack does; on a dev broker with auto-create this is a no-op.
    subprocess.run([sys.executable, "-m", "hypersplat.streaming.topics", "--wait", "30", "create"],
                   env=env, check=True, timeout=60)
    subprocess.run([sys.executable, "-c", produce], env=env, check=True, timeout=60)
    subprocess.run([sys.executable, "-m", "hypersplat.streaming.processor", "--once", "--max-seconds", "60"],
                   env=env, check=True, timeout=90)
    psycopg = _pg_or_skip()
    with psycopg.connect(pg_schema) as conn:
        assert conn.execute("SELECT count(*), max(seq) FROM pose_history").fetchone() == (30, 29)
        assert conn.execute("SELECT count(*) FROM training_history").fetchone()[0] == 30
        assert conn.execute("SELECT poses_accepted, current_step FROM jobs WHERE job_id = 'kjob'"
                            ).fetchone() == (30, 290)


def test_rising_loss_is_flagged_without_the_forest_and_not_on_a_bump():
    losses = [1.0, 0.8, 0.6, 0.5, 0.5, 0.55, 0.62, 0.7, 0.8, 0.9, 1.0, 1.1]
    store, sender = MemoryStore(), FakeSender()
    proc = P.Processor(store, sender)          # no training model loaded: the trend rule stands alone
    out = proc.process_batch([rec(snap(i, loss=v)) for i, v in enumerate(losses)])
    assert out and all(a.source == "training" and a.reason.startswith("loss_rising=+") for a in out)
    assert out[-1].score < 0 and store.snaps[("job1", 11)]["is_anomaly"]
    assert not store.snaps[("job1", 2)]["is_anomaly"]

    bump = [0.5, 0.49, 0.48, 0.47, 0.9, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42]
    store2 = MemoryStore()
    out2 = P.Processor(store2, FakeSender()).process_batch([rec(snap(i, loss=v, job="job2")) for i, v in enumerate(bump)])
    assert out2 == []


# ---- second-order engineering: identity, dead letters, loss visibility, silent skips -------------

def test_anomaly_id_is_stable_across_replays_and_unique_per_flagged_event(model_dir):
    fit_pose_model(model_dir)
    model = P.load_model(c.POSE_MODEL, c.POSE_FEATURES)
    batch = [rec(pose(i)) for i in range(15)] + [rec(bad(15, x=80.0)), rec(bad(16, x=160.0))]
    first = P.Processor(MemoryStore(), pose_model=model).process_batch(batch)
    again = P.Processor(MemoryStore(), pose_model=model).process_batch(batch)   # replay after a crash
    assert first and [a.anomaly_id for a in first] == [a.anomaly_id for a in again]
    assert all(a.anomaly_id for a in first) and len({a.anomaly_id for a in first}) == len(first)
    wire = c.decode(c.encode(first[0])[2])
    assert wire.anomaly_id == first[0].anomaly_id                          # consumers can dedupe on it
    assert c.anomaly_id("j", "pose", 1, "m") != c.anomaly_id("j", "training", 1, "m")


def test_undecodable_messages_go_to_a_dead_letter_topic_with_their_reason():
    sender = FakeSender()
    proc = P.Processor(MemoryStore(), sender)
    proc.process_batch([(c.TOPIC_POSES.name, b"job1", b"{not json"), rec(pose(0)),
                        (c.TOPIC_TRAINING.name, b"job1", b"x" * 2000)])
    sender.flush()
    dead = [e for e in sender.events() if isinstance(e, c.DeadLetter)]
    assert [d.source_topic for d in dead] == [c.TOPIC_POSES.name, c.TOPIC_TRAINING.name]
    assert all(d.job_id == "job1" and d.reason for d in dead)
    assert len(dead[1].raw) <= c.DEAD_LETTER_RAW_CHARS                      # bounded, never the whole payload
    assert c.TOPIC_DEAD_LETTER in c.TOPICS and proc.drops == 2


def test_a_dead_letter_that_cannot_be_sent_never_stops_the_batch():
    class Broken(FakeSender):
        def send(self, event):
            if isinstance(event, c.DeadLetter):
                raise RuntimeError("broker down")
            return super().send(event)

    store = MemoryStore()
    P.Processor(store, Broken()).process_batch([(c.TOPIC_POSES.name, b"job1", b"junk"), rec(pose(0))])
    assert ("job1", 0) in store.poses


def test_sender_reports_what_it_is_losing():
    s = FakeSender(queue_max=2)
    s._closed.set()                      # stop the sender thread so the queue fills
    s._thread.join(timeout=1)
    s._closed.clear()
    sent = [s.send(pose(i)) for i in range(5)]
    stats = s.stats("job1")
    assert stats["dropped"] == sent.count(False) > 0 and stats["queue_depth"] <= 2
    assert set(stats) >= {"sent", "dropped", "queue_depth", "queue_max", "broker_up"}


def test_out_of_order_events_are_counted_not_silently_lost():
    store = MemoryStore()
    proc = P.Processor(store)
    proc.process_batch([rec(pose(5)), rec(pose(3)), rec(pose(5)), rec(snap(7)), rec(snap(2))])
    assert proc.skipped == {"pose": 2, "training": 1}                      # 3 is late, the second 5 a duplicate
    assert ("job1", 3) not in store.poses


def test_consumer_lag_is_end_offset_minus_position_and_never_raises():
    class TP(str):
        pass

    a, b = TP("a"), TP("b")

    class Consumer:
        def assignment(self):
            return {a, b}

        def end_offsets(self, tps):
            return {a: 100, b: 40}

        def position(self, tp):
            return {a: 90, b: 40}[tp]

    assert P.consumer_lag(Consumer()) == 10

    class Broken(Consumer):
        def end_offsets(self, tps):
            raise RuntimeError("coordinator not available")

    assert P.consumer_lag(Broken()) is None            # unknown, not zero
