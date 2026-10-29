"""Dashboard tests (rows 9, 15, 16, 17, 18): pure KPI/figure shaping on CPU, plus Postgres-backed
reads and Streamlit AppTest runs inside a throwaway schema (skipped when Postgres is down)."""

import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import psycopg
import pytest
from psycopg import sql

from hypersplat.streaming import contract as c
from hypersplat.streaming import dashboard as d

APP = str(Path(d.__file__))
TRAIN_ANOM = (50, 150, 250)
TRAIN_INJ = {150: "blur"}
POSE_ANOM = (30, 90, 160)
POSE_INJ = {90: "shuffle", 120: "shuffle"}


# ---------------------------------------------------------------------------------------------
# Synthetic job: shared by the DB tests and the e2e seed script
# ---------------------------------------------------------------------------------------------

def synth(n_train=300, n_pose=200, start_seq=0, end=None):
    """(training rows, pose rows) as dicts with the processor's columns."""
    end = end or datetime.now(timezone.utc)
    train, losses = [], []
    for i in range(start_seq, start_seq + n_train):
        step = i * c.TRAIN_LOG_EVERY
        loss = 0.4 * math.exp(-step / 1500) + 0.03 + 0.005 * math.sin(i)
        losses.append(loss)
        win = losses[-c.WINDOW:]
        train.append(dict(
            seq=i, ts=end - timedelta(seconds=0.1 * (start_seq + n_train - i)), step=step,
            loss=loss, num_gaussians=50000 + 40 * step, gpu_mem_gb=3.0 + step / 1e4,
            steps_per_s=25.0, loss_roll=sum(win) / len(win), loss_trend=-0.01 - i / 1e5,
            gs_growth=40.0, best_loss=min(losses), peak_mem_gb=3.0 + step / 1e4,
            anomaly_score=-0.2 if i in TRAIN_ANOM else 0.1, is_anomaly=i in TRAIN_ANOM,
            injected=TRAIN_INJ.get(i)))
    poses, confs = [], []
    for i in range(start_seq, start_seq + n_pose):
        a = i / 30
        conf = 900.0 + 50 * math.cos(i)
        confs.append(conf)
        win = confs[-c.WINDOW:]
        poses.append(dict(
            seq=i, ts=end - timedelta(seconds=0.15 * (start_seq + n_pose - i)),
            frame="frame_%05d.jpg" % i, cx=2 * math.cos(a), cy=0.1, cz=2 * math.sin(a),
            focal=1100.0, confidence=conf, step_dist=0.07, turn_deg=1.9,
            conf_roll=sum(win) / len(win), anomaly_score=-0.3 if i in POSE_ANOM else 0.1,
            is_anomaly=i in POSE_ANOM, injected=POSE_INJ.get(i)))
    return train, poses


def _insert(conn, table, rows):
    if not rows:
        return
    cols = list(rows[0])
    q = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, cols)),
        sql.SQL(", ").join(sql.Placeholder() * len(cols)))
    with conn.cursor() as cur:
        cur.executemany(q, [[r[k] for k in cols] for r in rows])


def seed(conn, job_id, n_train=300, n_pose=200, start_seq=0, dropped=7):
    """Insert history rows and upsert the jobs row the way the processor would."""
    train, poses = synth(n_train, n_pose, start_seq)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO jobs (job_id) VALUES (%s) ON CONFLICT DO NOTHING", (job_id,))
    for r in train:
        r["job_id"] = job_id
    for r in poses:
        r["job_id"] = job_id
    _insert(conn, "training_history", train)
    _insert(conn, "pose_history", poses)
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE jobs j SET updated_at = now(),
              current_step = t.step, current_loss = t.loss, num_gaussians = t.num_gaussians,
              best_loss = (SELECT min(loss) FROM training_history WHERE job_id = j.job_id),
              peak_gpu_mem_gb = (SELECT max(peak_mem_gb) FROM training_history
                                 WHERE job_id = j.job_id),
              poses_accepted = (SELECT count(*) FROM pose_history
                                WHERE job_id = j.job_id AND NOT is_anomaly),
              poses_rejected = (SELECT count(*) FROM pose_history
                                WHERE job_id = j.job_id AND is_anomaly),
              anomalies = (SELECT count(*) FROM pose_history WHERE job_id = j.job_id AND is_anomaly)
                        + (SELECT count(*) FROM training_history
                           WHERE job_id = j.job_id AND is_anomaly),
              events_dropped = %s
            FROM (SELECT step, loss, num_gaussians FROM training_history WHERE job_id = %s
                  ORDER BY step DESC LIMIT 1) t
            WHERE j.job_id = %s""", (dropped, job_id, job_id))


def expected_tiles(conn, job_id):
    """KPI tile values computed straight from SQL, independently of dashboard.kpis()."""
    with conn.cursor() as cur:
        cur.execute("SELECT poses_accepted, poses_rejected, events_dropped, anomalies, best_loss,"
                    " num_gaussians, peak_gpu_mem_gb FROM jobs WHERE job_id = %s", (job_id,))
        acc, rej, drop, anom, best, gs, peak = cur.fetchone()
        cur.execute("SELECT loss_roll FROM training_history WHERE job_id = %s "
                    "ORDER BY step DESC LIMIT 1", (job_id,))
        roll = cur.fetchone()[0]
        cur.execute("SELECT conf_roll FROM pose_history WHERE job_id = %s "
                    "ORDER BY seq DESC LIMIT 1", (job_id,))
        conf = cur.fetchone()[0]
    return {"Rolling loss": "%.4f" % roll, "Best loss": "%.4f" % best,
            "Gaussians": "{:,}".format(gs), "Peak GPU memory": "%.2f GB" % peak,
            "Poses accepted": str(acc), "Poses rejected": str(rej),
            "Rolling confidence": "%.1f" % conf, "Dropped events": str(drop),
            "Anomalies": str(anom)}


# ---------------------------------------------------------------------------------------------
# Pure shaping (no database)
# ---------------------------------------------------------------------------------------------

def _frames():
    train, poses = synth()
    return pd.DataFrame(train), pd.DataFrame(poses)


def test_kpis_prefer_job_counters_and_newest_rolling_values():
    training, poses = _frames()
    job = {"current_loss": 0.5, "best_loss": 0.031, "num_gaussians": 123, "peak_gpu_mem_gb": 4.5,
           "poses_accepted": 197, "poses_rejected": 3, "events_dropped": 7, "anomalies": 6}
    k = d.kpis(job, poses, training)
    assert k["rolling_loss"] == pytest.approx(training["loss_roll"].iloc[-1])
    assert k["loss_trend"] == pytest.approx(training["loss_trend"].iloc[-1])
    assert (k["best_loss"], k["gaussians"], k["peak_gpu_mem_gb"]) == (0.031, 123, 4.5)
    assert (k["poses_accepted"], k["poses_rejected"], k["dropped"], k["anomalies"]) == (197, 3, 7, 6)
    assert k["conf_roll"] == pytest.approx(poses["conf_roll"].iloc[-1])
    assert k["gs_growth"] == 40.0


def test_kpis_fall_back_and_tolerate_empty():
    training, poses = _frames()
    training.loc[training.index[-5:], "loss_roll"] = None  # processor still warming up
    k = d.kpis({}, poses, training)
    assert k["rolling_loss"] == pytest.approx(training["loss_roll"].dropna().iloc[-1])
    assert k["best_loss"] == pytest.approx(training["best_loss"].iloc[-1])
    assert k["gaussians"] == training["num_gaussians"].iloc[-1]
    empty = d.kpis({}, pd.DataFrame(), pd.DataFrame())
    assert all(v is None for v in empty.values())
    assert all(t[1] == "—" for t in d.kpi_tiles(empty))


def test_kpi_tiles_include_dropped_events():
    tiles = {t[0]: t[1] for t in d.kpi_tiles(d.kpis({"events_dropped": 12}, *_frames()[::-1]))}
    assert tiles["Dropped events"] == "12"
    assert len(tiles) == 10


def test_liveness():
    assert d.liveness(None)[0] == "none"
    assert d.liveness(1.234) == ("live", "Last event 1.2 s ago")
    level, text = d.liveness(d.STALE_AFTER_S + 30)
    assert level == "stale" and "paused" in text


def _red(fig, name):
    tr = [t for t in fig.data if t.name == name]
    assert len(tr) == 1 and tr[0].marker.color == c.ANOMALY_COLOR and tr[0].mode == "markers"
    return tr[0]


def test_loss_figure_marks_anomalies_red():
    training, _ = _frames()
    fig = d.loss_figure(training)
    red = _red(fig, "anomaly")
    assert list(red.x) == [s * c.TRAIN_LOG_EVERY for s in TRAIN_ANOM]
    inj = [t for t in fig.data if t.name.startswith("injected")][0]
    assert list(inj.x) == [150 * c.TRAIN_LOG_EVERY]
    assert d.loss_figure(pd.DataFrame()) is None


def test_path_figure_top_down_equal_aspect():
    _, poses = _frames()
    fig = d.path_figure(poses)
    red = _red(fig, "flagged")
    assert list(red.x) == pytest.approx([poses.cx[i] for i in POSE_ANOM])
    assert list(red.y) == pytest.approx([poses.cz[i] for i in POSE_ANOM])
    inj = [t for t in fig.data if t.name.startswith("injected")][0]
    assert len(inj.x) == len(POSE_INJ) and inj.marker.symbol == "circle-open"
    assert fig.layout.yaxis.scaleanchor == "x" and fig.layout.yaxis.scaleratio == 1
    assert d.path_figure(pd.DataFrame()) is None


def test_flagged_table():
    training, poses = _frames()
    t = d.flagged_table(poses, training)
    assert len(t) == len(TRAIN_ANOM) + len(POSE_ANOM)
    assert t["ts"].is_monotonic_decreasing
    assert len(d.flagged_table(poses.iloc[:0], training.iloc[:0])) == 0
    assert len(d.flagged_table(pd.DataFrame(), training)) == len(TRAIN_ANOM)


# ---------------------------------------------------------------------------------------------
# PostgreSQL + AppTest, each test in its own schema so it never touches the processor's tables
# ---------------------------------------------------------------------------------------------

def _pg_up():
    try:
        psycopg.connect(c.DB_DSN, connect_timeout=2).close()
        return True
    except psycopg.Error:
        return False


needs_pg = pytest.mark.skipif(not _pg_up(), reason="PostgreSQL unreachable at contract.DB_DSN")


@pytest.fixture
def schema_dsn(monkeypatch):
    """A fresh empty schema; c.DB_DSN points at it so the dashboard sees only this test's data."""
    name = "u6test_" + uuid.uuid4().hex[:10]
    base = c.DB_DSN
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    dsn = psycopg.conninfo.make_conninfo(base, options="-c search_path=%s" % name)
    monkeypatch.setattr(c, "DB_DSN", dsn)
    yield dsn
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))


@pytest.fixture
def db(schema_dsn):
    with psycopg.connect(schema_dsn, autocommit=True) as conn:
        conn.execute(c.DDL)
        yield conn


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _messages(at):
    return " ".join(e.value for e in list(at.info) + list(at.warning) + list(at.success))


def _tiles(at):
    return {m.label: m.value for m in at.metric}


def _plotly_traces(at):
    out = []
    for el in at.get("plotly_chart"):
        spec = json.loads(el.proto.spec)
        out.append({t.get("name"): t for t in spec["data"]})
    return out


@needs_pg
def test_load_jobs_and_job_tail(db):
    seed(db, "old")
    seed(db, "gone")
    seed(db, "new", n_train=20, n_pose=10)
    db.execute("UPDATE jobs SET archived = true WHERE job_id = 'gone'")
    db.execute("UPDATE jobs SET updated_at = now() - interval '1 hour' WHERE job_id = 'old'")
    assert list(d.load_jobs(db)["job_id"]) == ["new", "old"]
    data = d.load_job(db, "old", limit=50)
    assert len(data["poses"]) == 50 and len(data["training"]) == 50
    assert list(data["training"]["seq"]) == list(range(250, 300))  # newest tail, oldest first
    assert data["poses"]["seq"].is_monotonic_increasing
    assert data["job"]["events_dropped"] == 7
    assert 0 <= data["age_s"] < 60
    assert d.load_job(db, "nope")["age_s"] is None


@needs_pg
def test_liveness_uses_database_clock(db):
    """A producer clock 60 s behind must not make a live job look stopped."""
    seed(db, "skewed")
    db.execute("UPDATE pose_history SET ts = ts - interval '60 s'")
    db.execute("UPDATE training_history SET ts = ts - interval '60 s'")
    db.execute("UPDATE jobs SET updated_at = now()")
    assert d.liveness(d.load_job(db, "skewed")["age_s"])[0] == "live"
    db.execute("UPDATE jobs SET updated_at = now() - interval '90 s'")
    assert d.liveness(d.load_job(db, "skewed")["age_s"])[0] == "stale"


@needs_pg
def test_older_ddl_without_metric_columns(db):
    seed(db, "old-ddl")
    db.execute("ALTER TABLE training_history DROP COLUMN loss_roll, DROP COLUMN num_gaussians")
    db.execute("ALTER TABLE pose_history DROP COLUMN conf_roll")
    at = _app()
    assert _tiles(at)["Rolling loss"] == "—" and _tiles(at)["Anomalies"] == "6"


@needs_pg
def test_app_keeps_selected_job_when_order_changes(db):
    seed(db, "job-1")
    seed(db, "job-2")
    db.execute("UPDATE jobs SET updated_at = now() - interval '1 minute' WHERE job_id = 'job-1'")
    at = _app()
    assert at.selectbox[0].value == "job-2"  # newest by default
    at.selectbox[0].select("job-1").run()
    db.execute("UPDATE jobs SET updated_at = now() WHERE job_id = 'job-2'")
    seed(db, "job-3", n_train=5, n_pose=5)  # a new job appears and jumps to the top
    at.run()
    assert not at.exception
    assert at.selectbox[0].options[0] == "job-3" and at.selectbox[0].value == "job-1"


@needs_pg
def test_app_seeded_job(db):
    seed(db, "job-a")
    at = _app()
    assert at.selectbox[0].value == "job-a"
    tiles = _tiles(at)
    for label, value in expected_tiles(db, "job-a").items():
        assert tiles[label] == value, label
    assert "Last event" in _messages(at)
    figs = _plotly_traces(at)
    for name in ("anomaly", "flagged"):
        hits = [f[name] for f in figs if name in f]
        assert len(hits) == 1 and hits[0]["marker"]["color"] == c.ANOMALY_COLOR


@needs_pg
def test_app_rereads_fresh_data(db):
    seed(db, "job-b")
    first = _tiles(_app())
    seed(db, "job-b", n_train=10, n_pose=40, start_seq=300, dropped=11)
    second = _tiles(_app())
    assert second != first
    for label, value in expected_tiles(db, "job-b").items():
        assert second[label] == value, label
    assert second["Dropped events"] == "11"


@needs_pg
def test_app_no_tables(schema_dsn):
    at = _app()
    assert "no streaming tables" in _messages(at)


@needs_pg
def test_app_no_jobs(db):
    at = _app()
    assert "No jobs yet" in _messages(at)


def test_app_database_down(monkeypatch):
    monkeypatch.setattr(c, "DB_DSN", "postgresql://u:secret@127.0.0.1:1/none")
    at = _app()
    msg = _messages(at)
    assert "Waiting for the database at 127.0.0.1:1/none" in msg and "secret" not in msg
