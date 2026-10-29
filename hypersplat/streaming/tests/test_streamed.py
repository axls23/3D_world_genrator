"""The 8090 telemetry view reads what the processor wrote: flagged poses and training snapshots of the
newest job, nothing hard-coded."""
import pytest

from hypersplat.stream.streamed import StreamedResults
from hypersplat.streaming import contract as c
from hypersplat.streaming.tests.test_dashboard import db, needs_pg, schema_dsn  # noqa: F401  (fixtures)


def _job(db, job, poses, snaps=(), archived=False):
    db.execute("INSERT INTO jobs (job_id, archived) VALUES (%s, %s)", (job, archived))
    for seq, (frame, flagged, score) in enumerate(poses):
        db.execute("INSERT INTO pose_history (job_id, seq, ts, frame, anomaly_score, is_anomaly, injected) "
                   "VALUES (%s, %s, now(), %s, %s, %s, %s)", (job, seq, frame, score, flagged, "shuffle" if flagged else None))
    for seq, (step, flagged, score) in enumerate(snaps):
        db.execute("INSERT INTO training_history (job_id, seq, ts, step, loss, anomaly_score, is_anomaly) "
                   "VALUES (%s, %s, now(), %s, 0.1, %s, %s)", (job, seq, step, score, flagged))


def test_unavailable_when_the_database_cannot_be_reached():
    r = StreamedResults(dsn="postgresql://x:y@127.0.0.1:1/none", ttl=0).snapshot()
    assert r["available"] is False and "database" in r["reason"].lower() and r["flags"] == []


@needs_pg
def test_no_job_yet_is_reported_as_waiting_not_as_zero_flags(db):
    r = StreamedResults(ttl=0).snapshot()
    assert r["available"] is False and "no job" in r["reason"].lower()


@needs_pg
def test_snapshot_carries_the_processors_flags_for_the_newest_job(db):
    _job(db, "old", [("o0.jpg", True, -0.2)], archived=True)
    _job(db, "live", [("f0.jpg", False, 0.1), ("f1.jpg", True, -0.3), ("f2.jpg", False, 0.2)],
         snaps=[(10, False, 0.1), (20, True, -0.05)])
    r = StreamedResults(ttl=0).snapshot()
    assert r["available"] and r["job_id"] == "live"
    assert (r["poses_scored"], r["poses_flagged"], r["training_flagged"]) == (3, 1, 1)
    assert r["flagged_frames"] == ["f1.jpg"]
    kinds = {(f["source"], f.get("frame") or f.get("step")) for f in r["flags"]}
    assert kinds == {("pose", "f1.jpg"), ("training", 20)}
    assert all(f["reason"] for f in r["flags"])


@needs_pg
def test_a_pinned_job_wins_over_the_newest(db):
    _job(db, "a", [("a0.jpg", True, -0.2)])
    _job(db, "b", [("b0.jpg", False, 0.2)])
    assert StreamedResults(job_id="a", ttl=0).snapshot()["flagged_frames"] == ["a0.jpg"]


@needs_pg
def test_results_are_cached_for_the_ttl(db):
    _job(db, "c", [("c0.jpg", False, 0.2)])
    s = StreamedResults(ttl=60)
    assert s.snapshot()["poses_flagged"] == 0
    db.execute("UPDATE pose_history SET is_anomaly = true WHERE job_id = 'c'")
    assert s.snapshot()["poses_flagged"] == 0           # inside the ttl: no second query
    s.ttl = 0
    assert s.snapshot()["poses_flagged"] == 1


@needs_pg
def test_the_live_run_pins_its_own_job_through_the_hooks_env(db, monkeypatch):
    """live.sh exports HYPERSPLAT_JOB_ID for the hooks; the 8090 view must read that same job even when
    another job was updated more recently."""
    _job(db, "mine", [("m0.jpg", True, -0.2)])
    _job(db, "other-newer", [("o0.jpg", False, 0.2)])
    monkeypatch.setenv("HYPERSPLAT_JOB_ID", "mine")
    assert StreamedResults(ttl=0).snapshot()["job_id"] == "mine"
