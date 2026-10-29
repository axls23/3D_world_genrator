"""One-command health check for the streaming pipeline (Policy & Process Map row 20).

    python -m hypersplat.streaming.health [--interval 5] [--job JOB] [--json]

Samples the database (contract.HEALTH_QUERY) and the Kafka topic end offsets twice,
``--interval`` seconds apart, prints event and anomaly counts, and judges the flow:

    PASS  counts climb, anomalies > 0, broker and DB reachable
    IDLE  nothing climbed and no job was updated between the samples (finished)
    FAIL  anything else, one line per reason

Exit code 0 on PASS or IDLE, 1 on FAIL. Every probe degrades to a FAIL line, never a traceback,
so the check is safe to run while some layers are down. The verdict is the pure function
evaluate(), tested with fixture numbers in tests/test_health.py.
"""

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import contract as c

PROBE_TIMEOUT_S = 3.0  # per probe; a down layer must not stall the check


@dataclass(frozen=True)
class JobCounts:
    poses: int
    snapshots: int
    anomalies: int
    dropped: int
    updated_at: dt.datetime


@dataclass
class Sample:
    """One look at the DB and the broker. ``*_error`` set means that probe failed."""
    taken_at: dt.datetime                       # DB clock when reachable, so ages ignore skew
    jobs: Dict[str, JobCounts] = field(default_factory=dict)
    db_error: Optional[str] = None
    topics: Dict[str, Optional[int]] = field(default_factory=dict)  # None = topic missing
    kafka_error: Optional[str] = None


@dataclass
class Report:
    verdict: str                                # PASS | IDLE | FAIL
    lines: List[Tuple[str, str]]                # (PASS | IDLE | FAIL | INFO, message)
    job_state: Dict[str, str]


def _err(e: BaseException) -> str:
    msg = str(e).strip().splitlines()
    first = msg[0] if msg else ""
    name = type(e).__name__
    return first if first.startswith(name) else "%s: %s" % (name, first)  # kafka repeats the name


# ---------------------------------------------------------------------------------------------
# Probes (lazy imports: kafka / psycopg are only needed when the check actually runs)
# ---------------------------------------------------------------------------------------------

def probe_db(dsn: str, job: Optional[str]) -> Tuple[Dict[str, JobCounts], Optional[dt.datetime], Optional[str]]:
    try:
        import psycopg

        # statement_timeout too: a locked jobs table must give a FAIL line, not a hung check
        with psycopg.connect(dsn, connect_timeout=max(1, int(PROBE_TIMEOUT_S)),
                             options="-c statement_timeout=%d" % int(PROBE_TIMEOUT_S * 1000)) as conn:
            now = conn.execute("SELECT now()").fetchone()[0]
            rows = conn.execute(c.HEALTH_QUERY).fetchall()
    except Exception as e:
        return {}, None, _err(e)
    jobs = {r[0]: JobCounts(int(r[1]), int(r[2]), int(r[3]), int(r[4]), r[5])
            for r in rows if job is None or r[0] == job}
    return jobs, now, None


def probe_kafka(bootstrap: str, topics: Sequence[str]) -> Tuple[Dict[str, Optional[int]], Optional[str]]:
    consumer = None
    try:
        from kafka import KafkaConsumer, TopicPartition

        ms = int(PROBE_TIMEOUT_S * 1000)
        consumer = KafkaConsumer(bootstrap_servers=bootstrap, bootstrap_timeout_ms=ms,
                                 request_timeout_ms=ms + 1000)
        # topics() lists without the per-topic metadata request that can auto-create a topic
        existing = consumer.topics()
        out: Dict[str, Optional[int]] = {}
        for name in topics:
            parts = consumer.partitions_for_topic(name) if name in existing else None
            if not parts:
                out[name] = None
                continue
            ends = consumer.end_offsets([TopicPartition(name, p) for p in parts], timeout_ms=ms)
            out[name] = int(sum(ends.values()))
        return out, None
    except Exception as e:
        return {}, _err(e)
    finally:
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                pass


def take_sample(dsn: str, bootstrap: str, topics: Sequence[str], job: Optional[str]) -> Sample:
    # Kafka first: events counted on a topic then get the DB probe's time to reach the history
    offsets, kafka_error = probe_kafka(bootstrap, topics)
    jobs, now, db_error = probe_db(dsn, job)
    return Sample(taken_at=now or dt.datetime.now(dt.timezone.utc), jobs=jobs, db_error=db_error,
                  topics=offsets, kafka_error=kafka_error)


# ---------------------------------------------------------------------------------------------
# Verdict (pure)
# ---------------------------------------------------------------------------------------------

INPUT_TOPICS = (c.TOPIC_POSES.name, c.TOPIC_TRAINING.name)  # what the processor consumes
TOPICS_MODULE = next((l.module for l in c.LAYERS if l.name == "topics"), None)


def _age_s(s: Sample, job: JobCounts) -> float:
    return (s.taken_at - job.updated_at).total_seconds()


def evaluate(before: Sample, after: Sample, interval: float, bootstrap: str = c.KAFKA_BOOTSTRAP,
             job: Optional[str] = None) -> Report:
    lines: List[Tuple[str, str]] = []
    state: Dict[str, str] = {}

    db_ok = before.db_error is None and after.db_error is None
    if db_ok:
        lines.append(("PASS", "database reachable"))
    else:
        lines.append(("FAIL", "database unreachable (%s)" % (after.db_error or before.db_error)))

    kafka_ok = before.kafka_error is None and after.kafka_error is None
    if kafka_ok:
        lines.append(("PASS", "broker %s reachable" % bootstrap))
        missing = [t for t, n in after.topics.items() if n is None]
        if missing:
            lines.append(("FAIL", "topic(s) missing: %s (create with %s)"
                          % (", ".join(missing), TOPICS_MODULE)))
    else:
        lines.append(("FAIL", "broker %s unreachable (%s)"
                      % (bootstrap, after.kafka_error or before.kafka_error)))

    flowing = False
    if db_ok:
        if not after.jobs:
            lines.append(("FAIL", "no job %r in the database" % job if job else
                          "no active jobs in the database (nothing processed yet)"))
        for jid, a in after.jobs.items():
            b = before.jobs.get(jid)
            climbed = b is None or a.poses > b.poses or a.snapshots > b.snapshots
            # comparing the two updated_at values is exact; an age-vs-interval test would also
            # count the probes' own time and call a just-stalled job finished
            touched = b is not None and a.updated_at > b.updated_at
            age = _age_s(after, a)
            if climbed:
                state[jid] = "PASS"
                flowing = True
            elif not touched:
                state[jid] = "IDLE"
                lines.append(("IDLE", "%s: no new rows, last update %.0f s ago (finished)" % (jid, age)))
            else:
                state[jid] = "FAIL"
                lines.append(("FAIL", "%s: updated %.0f s ago but no new history rows in %.0f s "
                              "(processor dropping everything?)" % (jid, age, interval)))
        if after.jobs:
            # live jobs must show the model scoring; old finished jobs cannot vouch for them
            scope = [j for jid, j in after.jobs.items() if state.get(jid) == "PASS"] or after.jobs.values()
            total = sum(j.anomalies for j in scope)
            if total > 0:
                lines.append(("PASS", "%d anomal%s flagged" % (total, "y" if total == 1 else "ies")))
            else:
                lines.append(("FAIL", "no anomalies flagged (anomaly model not loaded or not scoring?)"))
        if flowing:
            lines.append(("PASS", "history rows climbing"))

    if kafka_ok:
        fed = [t for t in INPUT_TOPICS
               if (after.topics.get(t) or 0) > (before.topics.get(t) or 0)]
        # end offsets cover every job, so this cross-check only means something unfiltered
        if fed and db_ok and after.jobs and not flowing and job is None:
            lines.append(("FAIL", "events arrive on %s but no history rows climb (processor down?)"
                          % ", ".join(fed)))
        elif flowing and not fed:
            lines.append(("INFO", "DB climbs without new input events (processor catching up)"))

    if any(s == "FAIL" for s, _ in lines):
        verdict = "FAIL"
    elif flowing:
        verdict = "PASS"
    else:
        verdict = "IDLE"
    return Report(verdict, lines, state)


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

def _delta(a: Optional[int], b: Optional[int]) -> str:
    if a is None or b is None:
        return ""
    return "%+d" % (a - b)


def render(before: Sample, after: Sample, report: Report, interval: float) -> str:
    out = []
    if after.jobs:
        hdr = "%-24s %14s %14s %9s %8s %12s  %s" % (
            "job", "poses", "snapshots", "anomalies", "dropped", "last update", "state")
        out += [hdr, "-" * len(hdr)]
        for jid, a in after.jobs.items():
            b = before.jobs.get(jid)
            out.append("%-24s %14s %14s %9d %8d %12s  %s" % (
                jid[:24],
                "%d (%s)" % (a.poses, _delta(a.poses, b.poses if b else 0)),
                "%d (%s)" % (a.snapshots, _delta(a.snapshots, b.snapshots if b else 0)),
                a.anomalies, a.dropped, "%.0f s ago" % _age_s(after, a),
                report.job_state.get(jid, "")))
        out.append("")
    if after.topics:
        hdr = "%-44s %10s %8s" % ("topic", "messages", "delta")
        out += [hdr, "-" * len(hdr)]
        for name, n in after.topics.items():
            out.append("%-44s %10s %8s" % (name, "missing" if n is None else n,
                                           _delta(n, before.topics.get(name))))
        out.append("")
    out += ["%-4s  %s" % (s, m) for s, m in report.lines]
    out.append("VERDICT: %s  (interval %.0f s)" % (report.verdict, interval))
    return "\n".join(out)


def as_json(before: Sample, after: Sample, report: Report, interval: float) -> str:
    jobs = {}
    for jid, a in after.jobs.items():
        b = before.jobs.get(jid)
        jobs[jid] = dict(poses=a.poses, snapshots=a.snapshots, anomalies=a.anomalies,
                         dropped=a.dropped, updated_at=a.updated_at.isoformat(),
                         poses_delta=a.poses - (b.poses if b else 0),
                         snapshots_delta=a.snapshots - (b.snapshots if b else 0),
                         state=report.job_state.get(jid))
    topics = {name: dict(messages=n, delta=None if n is None or before.topics.get(name) is None
                         else n - before.topics[name]) for name, n in after.topics.items()}
    return json.dumps(dict(verdict=report.verdict, interval=interval, jobs=jobs, topics=topics,
                           checks=[dict(status=s, message=m) for s, m in report.lines]), indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypersplat.streaming.health", description=__doc__.split("\n")[0])
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between the two samples")
    ap.add_argument("--job", help="check only this job_id")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    if args.interval <= 0:
        ap.error("--interval must be > 0")

    topics = [t.name for t in c.TOPICS]
    before = take_sample(c.DB_DSN, c.KAFKA_BOOTSTRAP, topics, args.job)
    time.sleep(args.interval)
    after = take_sample(c.DB_DSN, c.KAFKA_BOOTSTRAP, topics, args.job)
    report = evaluate(before, after, args.interval, c.KAFKA_BOOTSTRAP, args.job)
    print(as_json(before, after, report, args.interval) if args.json
          else render(before, after, report, args.interval))
    return 0 if report.verdict in ("PASS", "IDLE") else 1


if __name__ == "__main__":
    sys.exit(main())
