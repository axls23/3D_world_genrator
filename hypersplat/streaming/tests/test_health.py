"""Health verdict (pure, fixture numbers) and overhead-bench statistics; live probes skip when
Kafka or PostgreSQL is unreachable."""

import datetime as dt
import math
import socket
from urllib.parse import urlparse

import pytest

from hypersplat.streaming import bench_overhead as bo
from hypersplat.streaming import contract as c
from hypersplat.streaming import health as h
from hypersplat.streaming.producer import FakeSender

T0 = dt.datetime(2026, 10, 29, 12, 0, 0, tzinfo=dt.timezone.utc)
POSES, TRAIN, ANOM, *_ = (t.name for t in c.TOPICS)


def _job(poses, snaps, anomalies=3, age_s=1.0, at=T0):
    return h.JobCounts(poses, snaps, anomalies, 0, at - dt.timedelta(seconds=age_s))


def _sample(jobs, topics=None, at=T0, db_error=None, kafka_error=None):
    topics = {POSES: 10, TRAIN: 5, ANOM: 1} if topics is None else topics
    return h.Sample(at, jobs, db_error, topics, kafka_error)


def _pair(before_jobs, after_jobs, before_topics=None, after_topics=None, interval=5.0, **kw):
    t1 = T0 + dt.timedelta(seconds=interval)
    return (_sample(before_jobs, before_topics, T0),
            _sample(after_jobs, after_topics, t1, **kw), interval)


def _fails(report):
    return [m for s, m in report.lines if s == "FAIL"]


def test_climbing_counts_pass():
    b, a, i = _pair({"j": _job(10, 5)}, {"j": _job(14, 6, at=T0 + dt.timedelta(seconds=5))},
                    after_topics={POSES: 14, TRAIN: 6, ANOM: 1})
    r = h.evaluate(b, a, i)
    assert r.verdict == "PASS" and r.job_state == {"j": "PASS"} and not _fails(r)


def test_finished_job_is_idle_not_fail():
    old = T0 - dt.timedelta(minutes=10)
    b, a, i = _pair({"j": _job(100, 50, at=old)}, {"j": _job(100, 50, at=old)})
    r = h.evaluate(b, a, i)
    assert r.verdict == "IDLE" and r.job_state == {"j": "IDLE"} and not _fails(r)


def test_recent_update_without_new_rows_fails():
    b, a, i = _pair({"j": _job(10, 5)}, {"j": _job(10, 5, age_s=1.0, at=T0 + dt.timedelta(seconds=5))})
    r = h.evaluate(b, a, i)
    assert r.verdict == "FAIL" and any("no new history rows" in m for m in _fails(r))


def test_no_anomalies_fails_even_when_flowing():
    b, a, i = _pair({"j": _job(10, 5, anomalies=0)}, {"j": _job(20, 9, anomalies=0)},
                    after_topics={POSES: 20, TRAIN: 9, ANOM: 1})
    r = h.evaluate(b, a, i)
    assert r.verdict == "FAIL" and any("no anomalies" in m for m in _fails(r))


def test_empty_db_fails_with_reason():
    b, a, i = _pair({}, {})
    r = h.evaluate(b, a, i)
    assert r.verdict == "FAIL" and any("no active jobs" in m for m in _fails(r))
    r = h.evaluate(b, a, i, job="nope")
    assert any("no job 'nope'" in m for m in _fails(r))


def test_unreachable_broker_and_db_are_fail_lines():
    b, a, i = _pair({"j": _job(10, 5)}, {"j": _job(14, 6)}, kafka_error="KafkaTimeoutError: x")
    r = h.evaluate(b, a, i, bootstrap="localhost:1")
    assert r.verdict == "FAIL" and any("localhost:1 unreachable" in m for m in _fails(r))
    b, a, i = _pair({}, {}, db_error="OperationalError: refused")
    r = h.evaluate(b, a, i)
    assert any("database unreachable" in m for m in _fails(r))
    assert not any("no active jobs" in m for m in _fails(r))  # no follow-on noise


def test_missing_topic_and_stuck_processor():
    b, a, i = _pair({"j": _job(10, 5, age_s=60)}, {"j": _job(10, 5, age_s=65, at=T0 + dt.timedelta(seconds=5))},
                    after_topics={POSES: 30, TRAIN: 5, ANOM: None})
    r = h.evaluate(b, a, i)
    fails = _fails(r)
    assert r.verdict == "FAIL"
    assert any(ANOM in m and "missing" in m for m in fails)
    assert any("processor down" in m for m in fails)


def test_new_job_between_samples_counts_as_climbing():
    old = T0 - dt.timedelta(minutes=10)
    b, a, i = _pair({"old": _job(5, 5, at=old)},
                    {"old": _job(5, 5, at=old), "new": _job(1, 0)},
                    after_topics={POSES: 11, TRAIN: 5, ANOM: 1})
    r = h.evaluate(b, a, i)
    assert r.verdict == "PASS" and r.job_state == {"old": "IDLE", "new": "PASS"}


def test_job_filter_skips_topic_cross_check():
    # another job feeds the topic; the filtered, finished job must not read as "processor down"
    old = T0 - dt.timedelta(minutes=10)
    b, a, i = _pair({"done": _job(5, 5, at=old)}, {"done": _job(5, 5, at=old)},
                    after_topics={POSES: 50, TRAIN: 5, ANOM: 1})
    assert h.evaluate(b, a, i, job="done").verdict == "IDLE"
    assert h.evaluate(b, a, i).verdict == "FAIL"


def test_anomalies_counted_on_live_jobs_only():
    old = T0 - dt.timedelta(minutes=10)
    b, a, i = _pair({"old": _job(5, 5, anomalies=9, at=old), "live": _job(1, 0, anomalies=0)},
                    {"old": _job(5, 5, anomalies=9, at=old), "live": _job(9, 3, anomalies=0)},
                    after_topics={POSES: 18, TRAIN: 8, ANOM: 1})
    r = h.evaluate(b, a, i)
    assert r.verdict == "FAIL" and any("no anomalies" in m for m in _fails(r))


def test_render_and_json_do_not_crash():
    b, a, i = _pair({"j": _job(10, 5)}, {"j": _job(14, 6)}, after_topics={POSES: 14, TRAIN: 6, ANOM: None})
    r = h.evaluate(b, a, i)
    text = h.render(b, a, r, i)
    assert "VERDICT: FAIL" in text and "missing" in text and "+4" in text
    assert '"verdict": "FAIL"' in h.as_json(b, a, r, i)


# --- bench_overhead statistics --------------------------------------------------------------

def test_percentile_and_rates():
    assert bo.percentile([1, 2, 3, 4], 50) == 2.5
    assert math.isnan(bo.percentile([], 50))
    pts = [(0, 0.0), (100, 10.0), (100, 11.0), (300, 20.0), (250, 21.0), (350, 21.0)]
    assert bo.rates_from_points(pts) == [10.0, pytest.approx(200 / 9)]


def test_steady_state_trims_ends():
    rates = list(range(20))
    assert bo.steady_state(rates) == list(range(2, 18))
    assert bo.steady_state([5.0]) == [5.0]
    s = bo.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["median"] == 3.0 and s["iqr"] == 2.0
    assert bo.pct_diff(9.5, 10.0) == pytest.approx(-5.0)


def test_hot_path_with_fake_sender():
    # the 1 ms p99 gate lives in the bench run; on a shared box a unit-test timing is flaky
    r = bo.time_hot_path(FakeSender(), calls=2000, warmup=100)
    assert r["calls"] == 2000 and r["dropped"] == 0 and 0 < r["p50"] <= r["p99"] <= r["max"]


# --- live probes (skip when infra is down) --------------------------------------------------

def _reachable(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def test_probe_kafka_dead_broker_degrades():
    offsets, err = h.probe_kafka("localhost:1", [POSES])
    assert offsets == {} and err


@pytest.mark.skipif(not _reachable(*c.KAFKA_BOOTSTRAP.split(":")), reason="Kafka unreachable")
def test_probe_kafka_live():
    pytest.importorskip("kafka")
    offsets, err = h.probe_kafka(c.KAFKA_BOOTSTRAP, [POSES, "hypersplat.no-such-topic-u7"])
    assert err is None and offsets["hypersplat.no-such-topic-u7"] is None


@pytest.mark.skipif(not _reachable(urlparse(c.DB_DSN).hostname, urlparse(c.DB_DSN).port or 5432),
                    reason="PostgreSQL unreachable")
def test_probe_db_live():
    pytest.importorskip("psycopg")
    jobs, now, err = h.probe_db(c.DB_DSN, None)
    # a DB without the DDL applied reports an error string instead of raising
    assert (err is None and now is not None) or "UndefinedTable" in err
