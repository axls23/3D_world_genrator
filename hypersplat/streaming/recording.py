"""Recorder: one real run -> one JSONL of contract events in time order (rows 3, 7, 12, 21).

The replay producer (replay.py) streams these files so the processor, models and dashboard can
be exercised without a GPU. Sources the pipeline already writes are converted, not re-recorded:

    poses     stream_events.arrows (hypersplat.stream.record) or a batch run's poses_final.txt
    training  the run's TensorBoard dir (train/* tags, contract.TB_TAGS)

Each line is exactly the value contract.encode() puts on Kafka ({"kind", "v", ...fields}), so
replay decodes it with contract.decode(). Poses and snapshots are interleaved by trainer step.

    python -m hypersplat.streaming.recording export \\
        --events results/stream_3dgs/stream_events.arrows --tb results/stream_3dgs/tb \\
        --out hypersplat/streaming/recordings/record1.jsonl

--tb is optional (poses only). Output belongs under the git-ignored recordings/ dir.
"""

import argparse
import bisect
import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from hypersplat.streaming import contract as c

# Batch runs: ACE registers every frame before 3DGS training starts.
PRE_TRAINING_STEP = -1


def read_tb_scalars(tb_dir: str) -> Dict[int, Tuple[float, Dict[str, float]]]:
    """{step: (wall_time, {tag: value})} for the TB_TAGS present in a TensorBoard dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # lazy

    acc = EventAccumulator(tb_dir, size_guidance={"scalars": 0})  # 0 = keep every point
    acc.Reload()
    out: Dict[int, Tuple[float, Dict[str, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        if tag not in c.TB_TAGS:
            continue
        for ev in acc.Scalars(tag):
            wall, vals = out.get(ev.step, (0.0, {}))
            vals[tag] = ev.value
            out[ev.step] = (max(wall, ev.wall_time), vals)
    return out


class _Clock:
    """Trainer step -> wall time, interpolated from the TB points (poses carry no wall time)."""

    def __init__(self, tb: Dict[int, Tuple[float, Dict]], fallback: float):
        self.steps = sorted(tb)
        self.walls = [tb[s][0] for s in self.steps]
        self.fallback = fallback

    def __call__(self, step: int) -> float:
        if not self.steps:
            return self.fallback
        i = bisect.bisect_left(self.steps, step)
        if i == 0:
            return self.walls[0]
        if i == len(self.steps):
            return self.walls[-1]
        s0, s1, w0, w1 = self.steps[i - 1], self.steps[i], self.walls[i - 1], self.walls[i]
        return w0 + (w1 - w0) * (step - s0) / (s1 - s0)


def poses_from_events(path: str, job_id: str, clock, stats: Dict) -> List[Tuple[int, c.CameraPose]]:
    """[(trainer_step, CameraPose)] from a stream recording; Unregistered rows are counted."""
    from hypersplat.stream.record import load_recording  # lazy: pyarrow + numpy

    out = []
    focal = None  # pose rows without a focal reuse the last map's
    for step, _kind, rows in load_recording(path):
        for row in rows:
            try:
                p = c.pose_from_event_row(row, job_id, seq=0, focal=focal, ts=clock(step))
            except c.Unregistered:
                stats["unregistered"] += 1
                continue
            except c.InvalidEvent:
                stats["invalid"] += 1
                continue
            focal = p.focal
            out.append((step, p))
    return out


def poses_from_file(path: str, job_id: str, ts: float, stats: Dict) -> List[Tuple[int, c.CameraPose]]:
    """[(PRE_TRAINING_STEP, CameraPose)] from a poses_final.txt (batch runs)."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                out.append((PRE_TRAINING_STEP, c.pose_from_ace_line(line, job_id, 0, ts=ts, source="file")))
            except c.InvalidEvent:
                stats["invalid"] += 1
    return out


def snapshots_from_tb(tb: Dict[int, Tuple[float, Dict[str, float]]], job_id: str,
                      stats: Dict) -> List[Tuple[int, c.TrainingSnapshot]]:
    out = []
    for step in sorted(tb):
        wall, scalars = tb[step]
        try:
            out.append((step, c.snapshot_from_scalars(step, scalars, job_id, 0, ts=wall)))
        except c.InvalidEvent:  # e.g. a step logged without train/mem or train/num_GS
            stats["invalid"] += 1
    return out


def build(job_id: str, events: Optional[str] = None, pose_file: Optional[str] = None,
          tb_dir: Optional[str] = None) -> Tuple[List, Dict]:
    """(events in time order with seq 0..n-1, stats). Exactly one of events / pose_file."""
    if (events is None) == (pose_file is None):
        raise ValueError("give exactly one of events / pose_file")
    stats = {"poses": 0, "training": 0, "unregistered": 0, "invalid": 0}
    tb = read_tb_scalars(tb_dir) if tb_dir else {}
    clock = _Clock(tb, fallback=time.time())
    if events is not None:
        poses = poses_from_events(events, job_id, clock, stats)
    else:
        poses = poses_from_file(pose_file, job_id, clock(PRE_TRAINING_STEP), stats)
    snaps = snapshots_from_tb(tb, job_id, stats)
    # Stable sort: recorded order within a step, and a step's poses before its snapshot (the
    # trainer logs after applying the events that arrived during the step).
    merged = sorted([(s, 0, i, e) for i, (s, e) in enumerate(poses)]
                    + [(s, 1, i, e) for i, (s, e) in enumerate(snaps)], key=lambda t: t[:3])
    out = [replace(e, seq=n) for n, (_, _, _, e) in enumerate(merged)]
    stats["poses"], stats["training"] = len(poses), len(snaps)
    return out, stats


def write_jsonl(events: List, out: str) -> None:
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    tmp = "%s.%d.tmp" % (out, os.getpid())
    try:
        with open(tmp, "wb") as f:
            for e in events:
                f.write(c.encode(e)[2] + b"\n")
        os.replace(tmp, out)  # a half-written recording never looks complete
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hypersplat.streaming.recording", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="one run -> one JSONL of contract events")
    src = ex.add_mutually_exclusive_group(required=True)
    src.add_argument("--events", help="stream_events.arrows from a streaming run")
    src.add_argument("--pose-file", help="poses_final.txt from a batch run")
    ex.add_argument("--tb", help="TensorBoard dir of the same run (optional: poses only without it)")
    ex.add_argument("--out", required=True, help="recordings/<name>.jsonl")
    ex.add_argument("--job-id", help="job_id inside the file (default: the --out file stem); "
                                     "replay rewrites it anyway")
    a = ap.parse_args(argv)

    job_id = a.job_id or os.path.splitext(os.path.basename(a.out))[0]
    if a.tb and not os.path.isdir(a.tb):
        ap.error("--tb %s is not a directory" % a.tb)
    events, stats = build(job_id, events=a.events, pose_file=a.pose_file, tb_dir=a.tb)
    if not events:
        print("[recording] nothing to export (%s)" % stats, file=sys.stderr)
        return 1
    write_jsonl(events, a.out)
    print("[recording] %s: %d events (%d poses, %d training), skipped %d unregistered, %d invalid%s"
          % (a.out, len(events), stats["poses"], stats["training"], stats["unregistered"],
             stats["invalid"], "" if a.tb else "; no --tb, poses only"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
