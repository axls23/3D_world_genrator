"""Evidence that streaming never slows training (constraint "training speed comes first",
Policy & Process Map rows 3, 4, 20).

    python -m hypersplat.streaming.bench_overhead --tb-on RUN_A/tb --tb-off RUN_B/tb
    python -m hypersplat.streaming.bench_overhead --micro [--calls 20000]

(a) --tb-on/--tb-off: training throughput of two runs from their TensorBoard event files. Each
    gap between consecutive ``train/loss`` events gives one steps/s sample (Δstep / Δwall_time);
    the first and last 10 % of each event file are cut (warm-up / JIT, final eval and save) and
    the steady-state middle 80 % is summarised as median and IQR. Gaps are never taken across
    event files, so a directory holding restarts is still measured correctly.
(b) --micro: the hot path the trainer / ACE hooks add per event, ``pose_from_event_row``
    (which runs ``contract.validate``) followed by ``Sender.send``, timed per call against a
    dead broker, a live broker and a full queue. Fails (exit 1) unless p99 < 1 ms in every case.

Output is a markdown block for the report, also written to results/overhead.md (git-ignored).
"""

import argparse
import datetime as dt
import glob
import logging
import math
import os
import platform
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from . import contract as c
from .producer import Sender

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "overhead.md")
LOSS_TAG = "train/loss"
TRIM = 0.10            # cut this fraction from each end of every event file
P99_LIMIT_US = 1000.0  # a send must never cost the trainer more than 1 ms
BENCH_JOB = "bench-overhead"
# The live case writes here, never to the processor's input topics, so a benchmark cannot leave
# thousands of fake poses in the DB and on the dashboard.
BENCH_TOPIC = c.TOPIC_PREFIX + "hypersplat.bench-overhead"
BENCH_RETENTION_MS = 3600 * 1000


# ---------------------------------------------------------------------------------------------
# Statistics (pure)
# ---------------------------------------------------------------------------------------------

def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]; NaN for no data."""
    if not xs:
        return math.nan
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def rates_from_points(points: Sequence[Tuple[int, float]]) -> List[float]:
    """steps/s for each gap between consecutive (step, wall_time) points of one event file.

    Gaps where the step does not advance (a resumed run re-logging a step) or the clock does
    not move are skipped instead of producing 0 or inf.
    """
    out = []
    for (s0, t0), (s1, t1) in zip(points, points[1:]):
        if s1 > s0 and t1 > t0:
            out.append((s1 - s0) / (t1 - t0))
    return out


def steady_state(rates: Sequence[float], trim: float = TRIM) -> List[float]:
    """The middle (1 - 2*trim) of a run, in run order; short runs keep at least one sample."""
    n = len(rates)
    k = int(n * trim)
    mid = list(rates[k:n - k])
    return mid or list(rates)


def summarize(rates: Sequence[float]) -> Dict[str, float]:
    return dict(n=len(rates), median=percentile(rates, 50), q1=percentile(rates, 25),
                q3=percentile(rates, 75), iqr=percentile(rates, 75) - percentile(rates, 25))


def pct_diff(on: float, off: float) -> float:
    """Streaming-on throughput relative to off, in %; negative = streaming slowed training."""
    return (on - off) / off * 100.0 if off else math.nan


# ---------------------------------------------------------------------------------------------
# (a) TensorBoard throughput
# ---------------------------------------------------------------------------------------------

def event_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True))


def loss_points(event_file: str, tag: str = LOSS_TAG) -> List[Tuple[int, float]]:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader  # lazy

    pts = []
    for ev in EventFileLoader(event_file).Load():
        if ev.HasField("summary") and any(v.tag == tag for v in ev.summary.value):
            pts.append((int(ev.step), float(ev.wall_time)))
    return pts


def run_throughput(path: str, tag: str = LOSS_TAG) -> Tuple[Dict[str, float], int]:
    files = event_files(path)
    if not files:
        raise SystemExit("no TensorBoard event files under %s" % path)
    rates: List[float] = []
    for f in files:
        rates += steady_state(rates_from_points(loss_points(f, tag)))
    if not rates:
        raise SystemExit("no %s events with advancing steps in %s" % (tag, path))
    return summarize(rates), len(files)


def tb_section(on_path: str, off_path: str) -> str:
    on, n_on = run_throughput(on_path)
    off, n_off = run_throughput(off_path)
    diff = pct_diff(on["median"], off["median"])
    rows = [
        "| run | path | event files | gaps (middle 80 %) | median steps/s | IQR (q1 - q3) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for label, path, s, n in (("streaming on", on_path, on, n_on), ("streaming off", off_path, off, n_off)):
        rows.append("| %s | `%s` | %d | %d | %.2f | %.2f (%.2f - %.2f) |"
                    % (label, path, n, s["n"], s["median"], s["iqr"], s["q1"], s["q3"]))
    # the off run's own spread is the noise floor: a difference inside it is not a slowdown
    noise = off["iqr"] / off["median"] * 100.0 if off["median"] else math.nan
    rows += ["", "Throughput difference (on vs off, medians): **%+.1f %%** "
             "(off-run IQR is %.1f %% of its median; negative = streaming slower)." % (diff, noise)]
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------------
# (b) hot-path micro-benchmark
# ---------------------------------------------------------------------------------------------

def _rows(n: int = 64) -> List[Dict]:
    """ACE EVENT rows with varied rotations, built up front so the loop times only the hook."""
    rows = []
    for i in range(n):
        a = 0.05 * i
        ca, sa = math.cos(a), math.sin(a)
        w2c = [ca, 0.0, sa, 0.1 * i, 0.0, 1.0, 0.0, 0.0, -sa, 0.0, ca, 0.5, 0.0, 0.0, 0.0, 1.0]
        rows.append(dict(kind="pose", frame="frame_%05d.jpg" % i, w2c=w2c, conf=1500.0 + i, focal=525.0))
    return rows


class _Redirect:
    """A real KafkaProducer with the hooks' PRODUCER_CONFIG that writes every event to
    BENCH_TOPIC. Runs on the sender thread, so the redirect costs the timed caller nothing."""

    def __init__(self, bootstrap: str):
        from kafka import KafkaProducer  # lazy

        self._p = KafkaProducer(bootstrap_servers=bootstrap, **c.PRODUCER_CONFIG)

    def send(self, topic, key=None, value=None):
        return self._p.send(BENCH_TOPIC, key=key, value=value)

    def flush(self, timeout=None):
        self._p.flush(timeout=timeout)

    def close(self, timeout=None):
        self._p.close(timeout=timeout)


def _bench_topic(bootstrap: str, create: bool) -> None:
    """Create (or delete) BENCH_TOPIC; best effort, a failure shows up as an unmeasured case."""
    admin = None
    try:
        from kafka.admin import KafkaAdminClient, NewTopic

        admin = KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=5000)
        if create:
            if BENCH_TOPIC not in admin.list_topics():
                admin.create_topics([NewTopic(BENCH_TOPIC, 1, 1, topic_configs={
                    "retention.ms": str(BENCH_RETENTION_MS)})], raise_errors=False)
        else:
            admin.delete_topics([BENCH_TOPIC], raise_errors=False)
    except Exception:
        pass
    finally:
        if admin is not None:
            try:
                admin.close()
            except Exception:
                pass


class _Blocked:
    """Producer whose send() parks the sender thread, so the queue fills and stays full."""

    def __init__(self, gate: threading.Event):
        self.gate = gate

    def send(self, *a, **k):
        self.gate.wait()

    def flush(self, timeout=None):
        pass

    def close(self, timeout=None):
        pass


def time_hot_path(sender: Sender, calls: int, warmup: int = 500) -> Dict[str, float]:
    rows = _rows()
    clock = time.perf_counter_ns
    build, send, total = [], [], []
    drops0 = sender.dropped(BENCH_JOB)
    t_start = 0
    for i in range(warmup + calls):
        if i == warmup:
            t_start = clock()
            drops0 = sender.dropped(BENCH_JOB)
        t0 = clock()
        ev = c.pose_from_event_row(rows[i % len(rows)], BENCH_JOB, seq=i, focal=525.0, ts=1.0)
        t1 = clock()
        sender.send(ev)
        t2 = clock()
        if i >= warmup:
            build.append((t1 - t0) / 1e3)
            send.append((t2 - t1) / 1e3)
            total.append((t2 - t0) / 1e3)
    elapsed = (clock() - t_start) / 1e9
    return dict(calls=calls, build_p50=percentile(build, 50), send_p50=percentile(send, 50),
                p50=percentile(total, 50), p99=percentile(total, 99), max=max(total),
                rate=calls / elapsed if elapsed > 0 else math.nan,
                dropped=sender.dropped(BENCH_JOB) - drops0)


def micro_cases(calls: int, bootstrap: str) -> List[Tuple[str, str, Optional[Dict[str, float]]]]:
    """(case, note, result or None when the case could not run)."""
    # drops are the point of two cases; the table counts them, the per-100 warning is noise here
    logging.getLogger("hypersplat.streaming.producer").setLevel(logging.ERROR)
    out = []

    dead = Sender("localhost:1")
    out.append(("dead broker", "`localhost:1`", time_hot_path(dead, calls)))
    dead.close(timeout=0.1)  # the sender thread may still be stuck bootstrapping; it is a daemon

    _bench_topic(bootstrap, create=True)
    live = Sender(bootstrap, producer_factory=_Redirect)
    live.send(c.pose_from_event_row(_rows(1)[0], BENCH_JOB, seq=0, focal=525.0, ts=1.0))
    live.flush(timeout=10.0)  # connect before timing so the case measures a live producer
    if live.sent:
        out.append(("live broker", "`%s`, topic `%s`" % (bootstrap, BENCH_TOPIC),
                    time_hot_path(live, calls)))
    else:
        out.append(("live broker", "`%s` did not take the warm-up event (broker or topic "
                    "unavailable), not measured" % bootstrap, None))
    live.close(timeout=10.0)
    _bench_topic(bootstrap, create=False)

    gate = threading.Event()
    full = Sender("blocked", queue_max=1, producer_factory=lambda _: _Blocked(gate))
    row = _rows(1)[0]
    deadline = time.time() + 5
    # fill until send() reports a drop: one event parked in the producer, one in the queue
    while full.send(c.pose_from_event_row(row, BENCH_JOB, seq=0, focal=525.0, ts=1.0)) \
            and time.time() < deadline:
        time.sleep(0.001)
    out.append(("full queue", "sender thread parked, every call drops", time_hot_path(full, calls)))
    gate.set()
    full.close(timeout=1.0)
    return out


def micro_section(cases) -> Tuple[str, bool]:
    rows = [
        "| case | setup | calls | build p50 µs | send p50 µs | p50 µs | p99 µs | max µs | calls/s | dropped | p99 < 1 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ok = True
    skipped = [name for name, _, r in cases if r is None]
    for name, note, r in cases:
        if r is None:
            rows.append("| %s | %s | - | - | - | - | - | - | - | - | n/a |" % (name, note))
            continue
        good = r["p99"] < P99_LIMIT_US
        ok &= good
        rows.append("| %s | %s | %d | %.1f | %.1f | %.1f | %.1f | %.0f | %.0f | %d | %s |" % (
            name, note, r["calls"], r["build_p50"], r["send_p50"], r["p50"], r["p99"], r["max"],
            r["rate"], r["dropped"], "yes" if good else "**NO**"))
    rows += ["", "Per call = `contract.pose_from_event_row` (incl. `contract.validate`) + "
             "`Sender.send`, back to back on one thread (a burst far above the trainer's rate of one "
             "snapshot per %d steps, so live-broker drops are queue overrun, not slowness). "
             "Verdict: **%s** (p99 < %.0f µs in every measured case%s)." % (
                 c.TRAIN_LOG_EVERY, "PASS" if ok else "FAIL", P99_LIMIT_US,
                 "; NOT measured: " + ", ".join(skipped) if skipped else "")]
    return "\n".join(rows), ok


# ---------------------------------------------------------------------------------------------

def _header() -> str:
    try:
        import kafka
        kv = kafka.__version__
    except Exception:
        kv = "not installed"
    return ("_%s, %s, Python %s, kafka-python %s_" % (
        dt.datetime.now().strftime("%Y-%m-%d %H:%M"), platform.node(), platform.python_version(), kv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypersplat.streaming.bench_overhead",
                                 description="Streaming overhead on training: TB throughput and hot-path micro-benchmark.")
    ap.add_argument("--tb-on", help="TB dir (or event file) of a run with streaming on")
    ap.add_argument("--tb-off", help="TB dir (or event file) of a run with streaming off")
    ap.add_argument("--micro", action="store_true", help="time the per-event hook hot path")
    ap.add_argument("--calls", type=int, default=20000, help="timed calls per micro case")
    ap.add_argument("--bootstrap", default=c.KAFKA_BOOTSTRAP, help="live broker for --micro")
    ap.add_argument("--out", default=DEFAULT_OUT, help="markdown file to write ('' = do not write)")
    args = ap.parse_args(argv)
    if bool(args.tb_on) != bool(args.tb_off):
        ap.error("--tb-on and --tb-off go together")
    if not (args.tb_on or args.micro):
        ap.error("give --tb-on/--tb-off, --micro, or both")
    if args.calls < 1:
        ap.error("--calls must be >= 1")

    parts = ["### Streaming overhead on training", "", _header()]
    ok = True
    if args.tb_on:
        parts += ["", "#### Training throughput from TensorBoard (`train/loss` gaps)", "",
                  tb_section(args.tb_on, args.tb_off)]
    if args.micro:
        table, ok = micro_section(micro_cases(args.calls, args.bootstrap))
        parts += ["", "#### Hook hot path", "", table]
    md = "\n".join(parts) + "\n"
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(md)
        print("written to %s" % args.out, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
