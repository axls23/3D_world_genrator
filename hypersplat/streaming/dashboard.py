"""Streamlit dashboard, the sixth layer of the streaming mini-project (Policy & Process Map rows
9, 15, 16, 17, 18).

    streamlit run hypersplat/streaming/dashboard.py --server.port 8501   # contract.DASHBOARD_PORT

Reads PostgreSQL only (the processor writes ``jobs``, ``pose_history`` and ``training_history``
from contract.DDL) and never talks to Kafka or the trainer. Every DASHBOARD_REFRESH_S seconds it
opens a fresh connection, reads the selected job's row plus the newest DASHBOARD_ROWS rows of
each history table, closes the connection and redraws (row 17: no cached connection or cached
data, so what is shown is what is in the database now).

Shows the row 9 KPIs, the loss curve and the top-down camera path with flagged events as red
markers (row 18), and how long ago the last event arrived so a paused producer is visible.

SQL and dataframe shaping live in plain functions (load_jobs, load_job, kpis, *_figure) that the
tests call without Streamlit.
"""

import html
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# `streamlit run` puts only this file's directory on sys.path; the package import needs the root.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import psycopg  # noqa: E402
from psycopg import errors as pg_errors  # noqa: E402

from hypersplat.streaming import contract as c  # noqa: E402

CONNECT_TIMEOUT_S = 3     # a dead database must not hang a 2 s refresh
STALE_AFTER_S = 3 * c.DASHBOARD_REFRESH_S  # no event for this long: producer paused or stopped

# Series colours; red (contract.ANOMALY_COLOR) is reserved for flagged events.
# Palette and look follow the 8090 telemetry view (PR #34 design): dark navy panels, teal accent.
BG, PANEL, LINE = "#0a0f17", "#111a27", "#253448"
TEXT, MUTED = "#e8eef5", "#8999ad"
ACCENT, BLUE, WARN = "#49d6b3", "#68a9ff", "#f3b95f"
RAW_COLOR = MUTED
MAIN_COLOR = BLUE
OUTLINE_COLOR = TEXT

_POSE_NUMERIC = ("seq", "cx", "cy", "cz", "focal", "confidence", "anomaly_score") + c.POSE_METRICS
_TRAINING_NUMERIC = ("seq", "step", "loss", "num_gaussians", "gpu_mem_gb", "steps_per_s",
                     "anomaly_score") + c.TRAINING_METRICS
_POSE_COLUMNS = _POSE_NUMERIC + ("ts", "frame", "is_anomaly", "injected")
_TRAINING_COLUMNS = _TRAINING_NUMERIC + ("ts", "is_anomaly", "injected")


# ---------------------------------------------------------------------------------------------
# Database reads (rows 15, 17)
# ---------------------------------------------------------------------------------------------

def connect(dsn: Optional[str] = None) -> psycopg.Connection:
    """A new connection per refresh; read at call time so tests can point c.DB_DSN elsewhere."""
    return psycopg.connect(dsn or c.DB_DSN, connect_timeout=CONNECT_TIMEOUT_S, autocommit=True)


def _frame(conn, sql: str, params=(), numeric=(), columns=()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        df = pd.DataFrame.from_records(cur.fetchall(), columns=cols)
    for col in columns:  # tables from an older DDL lack newer columns: show them empty
        if col not in df:
            df[col] = None
    for col in numeric:  # an all-NULL column arrives as object dtype
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_jobs(conn) -> pd.DataFrame:
    """Non-archived jobs, most recently updated first (the selector's default is row 0)."""
    return _frame(conn, "SELECT * FROM jobs WHERE NOT archived ORDER BY updated_at DESC, job_id")


def load_job(conn, job_id: str, limit: int = c.DASHBOARD_ROWS) -> Dict:
    """One job: its jobs row, the newest ``limit`` pose and training rows (oldest first) and the
    age in seconds of its newest event (None when it has none yet)."""
    # Event ts is stamped by the producer's clock (Windows trainer, WSL ACE) while now() is the
    # database's; updated_at is written by the processor on the database clock, so a producer
    # whose clock lags (WSL drift after sleep) still reads as live. A producer clock running
    # ahead can only hide a stop for as long as the skew.
    jobs = _frame(conn, """
        SELECT *, extract(epoch FROM now() - greatest(
                      updated_at,
                      (SELECT max(ts) FROM pose_history WHERE job_id = %s),
                      (SELECT max(ts) FROM training_history WHERE job_id = %s))) AS age_s
        FROM jobs WHERE job_id = %s""", (job_id, job_id, job_id))
    job = jobs.iloc[0].to_dict() if len(jobs) else {}
    age = job.pop("age_s", None)
    # ORDER BY ts DESC LIMIT walks the (job_id, ts DESC) index, never the whole job (row 15).
    poses = _frame(conn, "SELECT * FROM pose_history WHERE job_id = %s ORDER BY ts DESC LIMIT %s",
                   (job_id, limit), _POSE_NUMERIC, _POSE_COLUMNS)
    training = _frame(conn, "SELECT * FROM training_history WHERE job_id = %s "
                            "ORDER BY ts DESC LIMIT %s", (job_id, limit), _TRAINING_NUMERIC,
                      _TRAINING_COLUMNS)
    poses = poses.sort_values("seq", ignore_index=True)
    training = training.sort_values(["step", "seq"], ignore_index=True)
    no_events = not len(poses) and not len(training)
    return {"job": job, "poses": poses, "training": training,
            "age_s": None if no_events or age is None else float(age)}


# ---------------------------------------------------------------------------------------------
# KPIs (row 9)
# ---------------------------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _last(df: pd.DataFrame, col: str) -> Optional[float]:
    """Newest non-null value of a column; the processor leaves metrics NULL until warmed up."""
    if col not in df or not len(df):
        return None
    s = df[col].dropna()
    return _num(s.iloc[-1]) if len(s) else None


def _first(*vals) -> Optional[float]:
    for v in vals:
        v = _num(v)
        if v is not None:
            return v
    return None


def kpis(job: Dict, poses: pd.DataFrame, training: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Row 9 KPIs. Counters come from the jobs row (whole job); rolling values from the newest
    history row that has them ("—" while the processor's window is still filling)."""
    peak_hist = training["peak_mem_gb"].max() if "peak_mem_gb" in training else None
    max_mem = training["gpu_mem_gb"].max() if "gpu_mem_gb" in training else None
    min_loss = training["loss"].min() if "loss" in training else None
    return {
        "rolling_loss": _last(training, "loss_roll"),
        "loss_trend": _last(training, "loss_trend"),
        "best_loss": _first(job.get("best_loss"), _last(training, "best_loss"), min_loss),
        "gaussians": _first(job.get("num_gaussians"), _last(training, "num_gaussians")),
        "gs_growth": _last(training, "gs_growth"),
        "peak_gpu_mem_gb": _first(job.get("peak_gpu_mem_gb"), peak_hist, max_mem),
        "poses_accepted": _first(job.get("poses_accepted")),
        "poses_rejected": _first(job.get("poses_rejected")),
        "conf_roll": _last(poses, "conf_roll"),
        "dropped": _first(job.get("events_dropped")),
        "anomalies": _first(job.get("anomalies")),
    }


def _fmt(v: Optional[float], spec: str) -> str:
    return "—" if v is None else format(v, spec)


def kpi_tiles(k: Dict[str, Optional[float]]) -> List[Tuple[str, str, Optional[str], str, str]]:
    """(label, value, delta, delta_color, help) per tile, in display order."""
    trend = k["loss_trend"]
    growth = k["gs_growth"]
    return [
        ("Rolling loss", _fmt(k["rolling_loss"], ".4f"), None, "off",
         "Mean loss over the last %d snapshots" % c.WINDOW),
        ("Loss trend", _fmt(None if trend is None else trend * 100, "+.2f") + (
            "" if trend is None else " %/snap"), None, "off",
         "Least-squares slope of the window's loss, relative to its mean; negative = improving"),
        ("Best loss", _fmt(k["best_loss"], ".4f"), None, "off", "Lowest loss so far"),
        ("Gaussians", _fmt(k["gaussians"], ",.0f"),
         None if growth is None else "%+.1f / step" % growth, "off",
         "Current Gaussian count; delta = growth per trainer step over the window"),
        ("Peak GPU memory", _fmt(k["peak_gpu_mem_gb"], ".2f") + (
            "" if k["peak_gpu_mem_gb"] is None else " GB"), None, "off",
         "torch max_memory_allocated"),
        ("Poses accepted", _fmt(k["poses_accepted"], ",.0f"), None, "off",
         "Poses the pose model did not flag"),
        ("Poses rejected", _fmt(k["poses_rejected"], ",.0f"), None, "off",
         "Poses flagged by the pose model"),
        ("Rolling confidence", _fmt(k["conf_roll"], ".1f"), None, "off",
         "Mean ACE confidence over the last %d poses" % c.WINDOW),
        ("Dropped events", _fmt(k["dropped"], ",.0f"), None, "off",
         "Producer queue/broker drops plus processor validation drops (row 4)"),
        ("Anomalies", _fmt(k["anomalies"], ",.0f"), None, "off",
         "Events flagged by either Isolation Forest"),
    ]


def liveness(age_s: Optional[float]) -> Tuple[str, str]:
    """('live' | 'stale' | 'none', text) for the last-event indicator."""
    if age_s is None:
        return "none", "No events for this job yet"
    age_s = max(0.0, age_s)  # producer clock slightly ahead of the database's
    shown = "%.1f" % age_s if age_s < 10 else "%.0f" % age_s
    if age_s > STALE_AFTER_S:
        return "stale", "Last event %s s ago: the producer looks paused or stopped" % shown
    return "live", "Last event %s s ago" % shown


# ---------------------------------------------------------------------------------------------
# Figures (rows 16, 18)
# ---------------------------------------------------------------------------------------------

def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    if col == "injected":
        return df[col].notna() & (df[col].astype(str) != "")
    return df[col].fillna(False).astype(bool)


def _layout(fig: go.Figure, title: str, x: str, y: str, height: int) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=13, color=TEXT)), height=height,
                      margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"),
                      hovermode="closest")
    fig.update_xaxes(title=x, gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(title=y, gridcolor=LINE, zerolinecolor=LINE)
    return fig


def loss_figure(training: pd.DataFrame) -> Optional[go.Figure]:
    """Loss and rolling loss against step; flagged snapshots as separate red markers."""
    if not len(training):
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=training["step"], y=training["loss"], name="loss", mode="lines",
                             line=dict(color=RAW_COLOR, width=1)))
    if "loss_roll" in training and training["loss_roll"].notna().any():
        fig.add_trace(go.Scatter(x=training["step"], y=training["loss_roll"], name="rolling loss",
                                 mode="lines", line=dict(color=MAIN_COLOR, width=2)))
    bad = training[_flag(training, "is_anomaly")]
    fig.add_trace(go.Scatter(
        x=bad["step"], y=bad["loss"], name="anomaly", mode="markers",
        marker=dict(color=c.ANOMALY_COLOR, size=9, line=dict(color="white", width=1)),
        customdata=bad[["seq"]].assign(score=bad.get("anomaly_score")).values,
        hovertemplate="step %{x}<br>loss %{y:.4f}<br>seq %{customdata[0]}"
                      "<br>score %{customdata[1]:.3f}<extra>anomaly</extra>"))
    inj = training[_flag(training, "injected")]
    if len(inj):
        fig.add_trace(go.Scatter(
            x=inj["step"], y=inj["loss"], name="injected (ground truth)", mode="markers",
            marker=dict(symbol="circle-open", size=15, color=OUTLINE_COLOR, line=dict(width=2)),
            text=inj["injected"], hovertemplate="step %{x}<br>injected %{text}<extra></extra>"))
    return _layout(fig, "Loss", "step", "loss", 360)


def path_figure(poses: pd.DataFrame) -> Optional[go.Figure]:
    """Camera centres seen from above (x right, z forward), equal aspect so turns look true."""
    poses = poses.dropna(subset=["cx", "cz"]) if len(poses) else poses
    if not len(poses):
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=poses["cx"], y=poses["cz"], name="camera path", mode="lines+markers",
        line=dict(color=MAIN_COLOR, width=2), marker=dict(color=MAIN_COLOR, size=4),
        text=poses["frame"], customdata=poses[["seq"]].values,
        hovertemplate="%{text}<br>seq %{customdata[0]}<br>x %{x:.3f}, z %{y:.3f}<extra></extra>"))
    newest = poses.iloc[[-1]]
    fig.add_trace(go.Scatter(x=newest["cx"], y=newest["cz"], name="newest", mode="markers",
                             marker=dict(color=MAIN_COLOR, size=12, symbol="diamond"),
                             text=newest["frame"], hovertemplate="%{text}<extra>newest</extra>"))
    bad = poses[_flag(poses, "is_anomaly")]
    fig.add_trace(go.Scatter(
        x=bad["cx"], y=bad["cz"], name="flagged", mode="markers",
        marker=dict(color=c.ANOMALY_COLOR, size=10, line=dict(color="white", width=1)),
        text=bad["frame"], customdata=bad[["seq"]].assign(score=bad.get("anomaly_score")).values,
        hovertemplate="%{text}<br>seq %{customdata[0]}<br>score %{customdata[1]:.3f}"
                      "<extra>flagged</extra>"))
    inj = poses[_flag(poses, "injected")]
    if len(inj):
        fig.add_trace(go.Scatter(
            x=inj["cx"], y=inj["cz"], name="injected (ground truth)", mode="markers",
            marker=dict(symbol="circle-open", size=16, color=OUTLINE_COLOR, line=dict(width=2)),
            text=inj["frame"] + " · " + inj["injected"].astype(str),
            hovertemplate="%{text}<extra>injected</extra>"))
    _layout(fig, "Camera path (top-down)", "x", "z", 460)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def gaussians_figure(training: pd.DataFrame) -> Optional[go.Figure]:
    if not len(training) or training["num_gaussians"].isna().all():
        return None
    fig = go.Figure(go.Scatter(x=training["step"], y=training["num_gaussians"], mode="lines",
                               name="Gaussians", line=dict(color=MAIN_COLOR, width=2)))
    return _layout(fig, "Gaussian count", "step", "Gaussians", 240)


def confidence_figure(poses: pd.DataFrame) -> Optional[go.Figure]:
    if not len(poses) or poses["confidence"].isna().all():
        return None
    fig = go.Figure(go.Scatter(x=poses["seq"], y=poses["confidence"], mode="lines",
                               name="confidence", line=dict(color=RAW_COLOR, width=1)))
    if poses["conf_roll"].notna().any():
        fig.add_trace(go.Scatter(x=poses["seq"], y=poses["conf_roll"], mode="lines",
                                 name="rolling confidence", line=dict(color=MAIN_COLOR, width=2)))
    return _layout(fig, "Pose confidence", "pose seq", "confidence", 240)


def flagged_table(poses: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """Flagged rows of both tables, newest first: the table view behind the red markers."""
    cols = ["kind", "seq", "ts", "where", "anomaly_score", "injected"]
    parts = []
    p = poses[_flag(poses, "is_anomaly")]
    if len(p):
        parts.append(p.assign(kind="pose", where=p["frame"]).reindex(columns=cols))
    t = training[_flag(training, "is_anomaly")]
    if len(t):
        parts.append(t.assign(kind="training",
                              where="step " + t["step"].astype("Int64").astype(str))
                     .reindex(columns=cols))
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    # Naive UTC: Streamlit's Arrow conversion of tz-aware columns trips a pandas deprecation.
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.tz_convert(None)
    return out.sort_values("ts", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------------------------

THEME_CSS = f"""<style>
.stApp {{ background: radial-gradient(ellipse at 12% -15%, #17334a 0, transparent 35%), {BG}; color:{TEXT}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
.hs-head {{ display:flex; align-items:center; gap:14px; padding:6px 0 14px; flex-wrap:wrap; }}
.hs-mark {{ width:34px; height:34px; display:grid; place-items:center; border-radius:11px; color:{ACCENT};
            background:#49d6b31a; border:1px solid #49d6b344; font-size:20px; }}
.hs-eyebrow {{ color:{MUTED}; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.13em; }}
.hs-title {{ font-size:15px; font-weight:700; margin:0; }}
.hs-badge {{ display:inline-flex; align-items:center; gap:7px; padding:5px 10px; border-radius:999px; font-size:10px;
             font-weight:800; letter-spacing:.08em; text-transform:uppercase; border:1px solid #49d6b344;
             color:{ACCENT}; background:#49d6b30f; }}
.hs-badge::before {{ content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }}
.hs-badge.stale {{ color:{WARN}; border-color:#f3b95f55; background:#f3b95f12; }}
.hs-badge.none {{ color:{MUTED}; border-color:{LINE}; background:transparent; }}
.hs-hint {{ color:#b6c4d5; font-size:12px; }}
.hs-kpis {{ display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:12px; margin-bottom:6px; }}
.hs-kpi {{ background:linear-gradient(145deg,#142031,#101925 72%); border:1px solid {LINE}; border-radius:14px;
           padding:13px 15px; position:relative; overflow:hidden; box-shadow:0 12px 32px #0003; }}
.hs-kpi::after {{ content:""; position:absolute; right:-24px; top:-32px; width:90px; height:90px; border-radius:50%;
                  background:var(--tone); opacity:.09; }}
.hs-kpi .k {{ color:{MUTED}; font-size:10px; font-weight:750; text-transform:uppercase; letter-spacing:.1em; }}
.hs-kpi .v {{ margin-top:6px; font-size:24px; line-height:1.1; font-weight:720; letter-spacing:-.03em;
              font-variant-numeric:tabular-nums; }}
.hs-kpi .n {{ margin-top:5px; color:{MUTED}; font-size:11px; min-height:14px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:linear-gradient(145deg,#142031,#101925 72%);
    border:1px solid {LINE} !important; border-radius:14px !important; box-shadow:0 12px 32px #0003; }}
.hs-quality {{ display:flex; gap:10px; padding:11px 12px; border-radius:10px; border:1px solid #f3b95f44;
               background:#f3b95f0b; margin-bottom:8px; }}
.hs-quality.ok {{ border-color:#49d6b344; background:#49d6b30b; }}
.hs-quality b {{ display:block; font-size:12px; }}
.hs-quality span {{ color:{MUTED}; font-size:11px; }}
@media (max-width: 900px) {{ .hs-kpis {{ grid-template-columns:repeat(2, minmax(0,1fr)); }} }}
</style>"""

# tile label -> accent tone of its card
_TONE = {"Rolling loss": BLUE, "Loss trend": BLUE, "Best loss": ACCENT, "Gaussians": ACCENT,
         "Peak GPU memory": WARN, "Poses accepted": ACCENT, "Poses rejected": c.ANOMALY_COLOR,
         "Rolling confidence": BLUE, "Dropped events": WARN, "Anomalies": c.ANOMALY_COLOR}


def header_html(job_id: str, level: str, text: str) -> str:
    label = {"live": "live", "stale": "stale", "none": "waiting"}[level]
    return ('<div class="hs-head"><div class="hs-mark">&#9672;</div><div><div class="hs-eyebrow">HyperSplat / streaming'
            '</div><div class="hs-title">Live reconstruction</div></div>'
            '<span class="hs-badge %s">%s</span><span class="hs-hint">%s &middot; job %s</span></div>'
            % (level, label, html.escape(text), html.escape(job_id)))


def cards_html(tiles) -> str:
    """The KPI tiles as the telemetry view's cards (same data as st.metric, different skin)."""
    cards = []
    for label, value, delta, _, help_ in tiles:
        cards.append('<div class="hs-kpi" style="--tone:%s" title="%s"><div class="k">%s</div><div class="v">%s</div>'
                     '<div class="n">%s</div></div>'
                     % (_TONE.get(label, ACCENT), html.escape(help_, quote=True), html.escape(label),
                        html.escape(value), html.escape(delta or "")))
    return '<div class="hs-kpis">%s</div>' % "".join(cards)


def quality_html(poses: pd.DataFrame, training: pd.DataFrame) -> str:
    """The telemetry view's anomaly card: what the models flagged in the rows in view."""
    n_pose = int(_flag(poses, "is_anomaly").sum()) if len(poses) else 0
    n_train = int(_flag(training, "is_anomaly").sum()) if len(training) else 0
    if n_pose + n_train == 0:
        return ('<div class="hs-quality ok"><div><b>No flagged events</b><span>Pose and training models are '
                'quiet in the rows in view.</span></div></div>')
    return ('<div class="hs-quality"><div><b>%d flagged event%s</b><span>%d pose, %d training (isolation '
            'forest or rising-loss trend). Red markers on the charts.</span></div></div>'
            % (n_pose + n_train, "" if n_pose + n_train == 1 else "s", n_pose, n_train))


def _where(dsn: str) -> str:
    """host:port/dbname of the DSN, without the password, for the error message."""
    try:
        d = psycopg.conninfo.conninfo_to_dict(dsn)
    except psycopg.Error:
        return "the configured database"
    return "%s:%s/%s" % (d.get("host", "localhost"), d.get("port", 5432), d.get("dbname", ""))


def _read(st) -> Optional[Tuple[pd.DataFrame, str, Dict]]:
    """One refresh worth of reads on a fresh connection; shows a message and returns None for
    every empty state (no database, no tables, no jobs)."""
    try:
        with connect() as conn:
            jobs = load_jobs(conn)
            if not len(jobs):
                st.info("No jobs yet. Start the processor and a producer (trainer/ACE hooks or "
                        "the replay producer); the dashboard picks the job up on its own.")
                return None
            options = list(jobs["job_id"])
            if st.session_state.get("job") not in options:
                st.session_state.pop("job", None)  # archived or gone: fall back to the newest
            job_id = st.selectbox("Job", options, key="job",
                                  help="Non-archived jobs, most recently updated first")
            return jobs, job_id, load_job(conn, job_id, c.DASHBOARD_ROWS)
    except pg_errors.UndefinedTable:
        st.info("The database is up but has no streaming tables yet. Start the processor "
                "(it applies contract.DDL), then events will appear here.")
    except psycopg.OperationalError:
        st.info("Waiting for the database at %s. Start it with "
                "`bash hypersplat/streaming/ops/infra.sh up`; this page retries every %g s."
                % (_where(c.DB_DSN), c.DASHBOARD_REFRESH_S))
    except psycopg.Error as e:
        st.warning("Database read failed (%s); retrying in %g s."
                   % (type(e).__name__, c.DASHBOARD_REFRESH_S))
    return None


def _chart(st, fig: Optional[go.Figure], key: str, empty: str) -> None:
    if fig is None:
        st.caption(empty)
    else:
        st.plotly_chart(fig, key=key)


def render(st) -> None:
    got = _read(st)
    if got is None:
        return
    _, job_id, data = got
    poses, training = data["poses"], data["training"]

    level, text = liveness(data["age_s"])
    st.markdown(header_html(job_id, level, text), unsafe_allow_html=True)
    st.markdown(cards_html(kpi_tiles(kpis(data["job"], poses, training))), unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            _chart(st, loss_figure(training), "loss", "No training snapshots yet.")
        a, b = st.columns(2)
        with a, st.container(border=True):
            _chart(st, gaussians_figure(training), "gaussians", "No Gaussian counts yet.")
        with b, st.container(border=True):
            _chart(st, confidence_figure(poses), "confidence", "No pose confidences yet.")
    with right:
        st.markdown(quality_html(poses, training), unsafe_allow_html=True)
        with st.container(border=True):
            _chart(st, path_figure(poses), "path", "No camera poses yet.")

    flagged = flagged_table(poses, training)
    with st.expander("Flagged events in view (%d)" % len(flagged)):
        st.dataframe(flagged, hide_index=True)
    st.caption("Job %s · newest %d rows per table · %d poses, %d snapshots in view · refresh %g s"
               % (job_id, c.DASHBOARD_ROWS, len(poses), len(training), c.DASHBOARD_REFRESH_S))


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="HyperSplat stream", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    if hasattr(st, "fragment"):
        # Only the fragment reruns on the timer; each run opens its own connection (row 17).
        st.fragment(run_every=c.DASHBOARD_REFRESH_S)(render)(st)
    else:  # Streamlit < 1.37
        import time

        render(st)
        time.sleep(c.DASHBOARD_REFRESH_S)
        st.rerun()


if __name__ == "__main__":
    main()
