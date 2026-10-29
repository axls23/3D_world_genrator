"""Offline training of the two anomaly models (Policy & Process Map rows 10, 11).

    past runs on disk --features (contract.py)--> IsolationForest x2 --joblib--> models/

Run once, before the processor starts; the processor only loads the saved files and never
refits inside the stream:

    python -m hypersplat.streaming.train_models [--runs DIR ...] [--holdout PATTERN ...]

Sources, all read-only:
  * TensorBoard event files under ``**/tb/`` -> TrainingSnapshot sequences (contract.TB_TAGS),
    one sequence per event file because a tb dir reused across restarts holds several runs.
  * ACE ``poses_final.txt`` -> CameraPose sequences in frame order.
  * Stream recordings ``stream_events.arrows`` -> CameraPose sequences; the live mapper sweeps
    the keyframes again after every relocalization, so a recording splits into one frame-ordered
    sequence per sweep. pose and reloc rows count as source "pose" here; map bursts are skipped
    (contract.SCORED_SOURCES).

Features come from contract.pose_features / training_features over the same sliding window the
processor keeps (last contract.WINDOW items, scored from contract.MIN_WINDOW_TO_SCORE on), so
the model sees exactly what it will be scored on (row 11, no train/serve skew). Whole runs
matching ``--holdout`` form the test split (row 11 report: train/test sizes in meta.json).
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import contract as c

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
# A whole capture the models never see (its ACE poses and its TensorBoard runs). The drone
# capture is in-domain for the stream recording the pose model mostly learns from; holding
# out the handheld phone run instead (--holdout whatsapp_run) measures cross-scene transfer,
# where raw step_dist / confidence shift with scene scale and the pose model over-flags.
DEFAULT_HOLDOUT = ("drone_demo_run",)
SCAN_TIMEOUT_S = 60.0
MAX_EVENT_FILE_BYTES = 256 << 20   # image-heavy event files cost minutes for a few scalars
SKIP_DIRS = {".git", ".claude", "node_modules", "__pycache__", ".cache", "wandb"}

POSE = "pose"
TRAINING = "training"
MODEL_OF = {POSE: c.POSE_MODEL, TRAINING: c.TRAINING_MODEL}
FEATURES_OF = {POSE: c.POSE_FEATURES, TRAINING: c.TRAINING_FEATURES}


@dataclass
class Run:
    """One event sequence in producer order: CameraPose (kind pose) or TrainingSnapshot."""
    id: str
    kind: str
    items: list = field(default_factory=list)


def meta_path() -> str:
    return os.path.join(c.MODEL_DIR, "meta.json")


# ---------------------------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------------------------

def _classify(path: str) -> Optional[str]:
    name = os.path.basename(path)
    if name.startswith("events.out.tfevents") and os.path.basename(os.path.dirname(path)) == "tb":
        return "tb"
    if name == "poses_final.txt":
        return "ace"
    if name == "stream_events.arrows":
        return "recording"
    return None


def discover(roots: Sequence[str], timeout_s: float = SCAN_TIMEOUT_S) -> Dict[str, List[str]]:
    """{"tb": [...], "ace": [...], "recording": [...]} under roots (files are taken as given).

    Virtualenvs, git internals and agent worktrees (copies of the same runs) are pruned; the
    walk stops at timeout_s so a huge mount cannot hang training."""
    found: Dict[str, List[str]] = {"tb": [], "ace": [], "recording": []}
    deadline = time.monotonic() + timeout_s
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            kind = _classify(root)
            if kind:
                found[kind].append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if time.monotonic() > deadline:
                logger.warning("scan stopped after %.0f s in %s; pass narrower --runs", timeout_s,
                               dirpath)
                break
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in SKIP_DIRS and not d.startswith(".venv"))
            for name in sorted(filenames):
                kind = _classify(os.path.join(dirpath, name))
                if kind:
                    found[kind].append(os.path.join(dirpath, name))
    return {k: sorted(set(v)) for k, v in found.items()}


def _run_id(path: str, bases: Sequence[str] = ()) -> str:
    """Path relative to the scanned directory (else the repo) that holds it: readable in meta
    and the same for a run whether it is scanned from the main checkout or a worktree."""
    p = os.path.abspath(path)
    for base in [b for b in bases if os.path.isdir(b)] + [str(REPO_ROOT)]:
        try:
            rel = os.path.relpath(p, os.path.abspath(base))
        except ValueError:  # other drive on Windows
            continue
        if not rel.startswith(".."):
            return rel
    return p


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_tb(path: str, job_id: Optional[str] = None) -> List[c.TrainingSnapshot]:
    """TrainingSnapshots of one event file, one per step that logged train/loss."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if os.path.getsize(path) > MAX_EVENT_FILE_BYTES:
        logger.warning("skip %s: %d MB event file", path, os.path.getsize(path) >> 20)
        return []
    ea = EventAccumulator(path, size_guidance={"scalars": 0, "images": 1, "tensors": 1,
                                               "histograms": 1, "audio": 1})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    by_step: Dict[int, Dict[str, float]] = {}
    wall: Dict[int, float] = {}
    for tag in c.TB_TAGS:
        if tag in tags:
            for e in ea.Scalars(tag):  # a restart re-logs a step: the later value wins
                by_step.setdefault(e.step, {})[tag] = e.value
                wall[e.step] = e.wall_time
    job_id = job_id or _run_id(path)
    out = []
    for step in sorted(by_step):
        scalars = by_step[step]
        if "train/loss" not in scalars:
            continue
        try:
            out.append(c.snapshot_from_scalars(step, scalars, job_id, len(out), ts=wall[step]))
        except c.InvalidEvent as e:  # e.g. num_GS never logged: not a usable snapshot
            logger.debug("skip %s step %d: %s", path, step, e)
    return out


def load_ace_poses(path: str, job_id: Optional[str] = None) -> List[c.CameraPose]:
    """poses_final.txt in frame order (ACE writes them in its own processing order)."""
    job_id = job_id or _run_id(path)
    poses = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                poses.append(c.pose_from_ace_line(line, job_id, 0, ts=0.0))
            except c.InvalidEvent as e:
                logger.debug("skip line in %s: %s", path, e)
    poses.sort(key=lambda p: _natural_key(p.frame))
    return [replace(p, seq=i, ts=float(i)) for i, p in enumerate(poses)]


def split_sweeps(poses: Sequence[c.CameraPose]) -> List[List[c.CameraPose]]:
    """Cut a recording's pose stream wherever the frame order restarts (a new mapper sweep)."""
    sweeps: List[List[c.CameraPose]] = []
    for p in poses:
        if not sweeps or _natural_key(p.frame) <= _natural_key(sweeps[-1][-1].frame):
            sweeps.append([])
        sweeps[-1].append(p)
    return sweeps


def load_recording_sweeps(path: str, job_id: Optional[str] = None) -> List[List[c.CameraPose]]:
    """Frame-ordered sweeps of one recording, also cut at every map event: poses after a
    refit live in the new map's frame, so a window must not mix both sides of it."""
    from hypersplat.stream.record import load_recording

    job_id = job_id or _run_id(path)
    segments: List[List[c.CameraPose]] = [[]]
    n = 0
    focal = None
    for step, kind, rows in load_recording(path):
        for row in rows:
            f = row.get("focal")
            if f is not None and math.isfinite(float(f)) and f > 0:  # pose rows may omit it
                focal = float(f)
        if kind == "map":   # a refit/load burst moves every camera at once: not scored
            if segments[-1]:
                segments.append([])
            continue
        for row in rows:
            try:
                # pose and reloc rows are both single-frame localizations: train them as "pose"
                segments[-1].append(c.pose_from_event_row(row, job_id, n, focal=focal,
                                                          ts=float(n), source="pose"))
                n += 1
            except c.InvalidEvent:   # Unregistered included: nothing to score
                continue
    return [sweep for seg in segments for sweep in split_sweeps(seg)]


def collect_runs(roots: Sequence[str], timeout_s: float = SCAN_TIMEOUT_S) -> List[Run]:
    found = discover(roots, timeout_s)
    runs: List[Run] = []
    for path in found["tb"]:
        rid = _run_id(path, roots)
        try:
            snaps = load_tb(path, rid)
        except Exception as e:  # truncated / foreign event file
            logger.warning("skip %s: %s", path, e)
            continue
        runs.append(Run(rid, TRAINING, snaps))
    for path in found["ace"]:
        rid = _run_id(path, roots)
        runs.append(Run(rid, POSE, load_ace_poses(path, rid)))
    for path in found["recording"]:
        rid = _run_id(path, roots)
        try:
            sweeps = load_recording_sweeps(path, rid)
        except Exception as e:
            logger.warning("skip %s: %s", path, e)
            continue
        for k, sweep in enumerate(sweeps):
            runs.append(Run("%s#sweep%02d" % (rid, k), POSE, sweep))
    for run in runs:
        if run.kind == POSE:
            run.items = [p for p in run.items if p.source in c.SCORED_SOURCES]
    return [r for r in runs if len(r.items) >= c.MIN_WINDOW_TO_SCORE]


# ---------------------------------------------------------------------------------------------
# Features, split, fit
# ---------------------------------------------------------------------------------------------

def window_at(seq: Sequence, i: int) -> Sequence:
    """The processor's window when item i arrives: the last WINDOW items up to and incl. i."""
    return seq[max(0, i - c.WINDOW + 1):i + 1]


def feature_rows(kind: str, seq: Sequence, positions: Optional[Iterable[int]] = None
                 ) -> Tuple[List[int], List[List[float]]]:
    """(positions scored, feature rows in contract column order); positions whose window is
    shorter than MIN_WINDOW_TO_SCORE are skipped, as the processor skips them."""
    fn = c.pose_features if kind == POSE else c.training_features
    names = FEATURES_OF[kind]
    if positions is None:
        positions = range(len(seq))
    used, rows = [], []
    for i in positions:
        w = window_at(seq, i)
        if len(w) < c.MIN_WINDOW_TO_SCORE:
            continue
        f = fn(w)
        row = [float(f[n]) for n in names]
        if all(math.isfinite(v) for v in row):
            used.append(i)
            rows.append(row)
    return used, rows


def matches(run_id: str, patterns: Sequence[str]) -> bool:
    return any(p in run_id for p in patterns)


def _source(run_id: str) -> str:
    """The capture a run came from: sweeps of one recording and the event files of one tb dir
    share it, so a fallback holdout never leaves near-duplicates of the test run in training."""
    return os.path.dirname(run_id.split("#")[0])


def split_runs(runs: Sequence[Run], holdout: Sequence[str]) -> Tuple[List[Run], List[Run]]:
    """(train, test) runs of both kinds: whole runs matching a holdout pattern are test.

    A kind with no matching run holds out the source of its last run so both splits exist; a
    kind whose every run matches keeps them all for training (a model fitted on nothing is
    worse)."""
    train, test = [], []
    for kind in (POSE, TRAINING):
        mine = [r for r in runs if r.kind == kind]
        held = [r for r in mine if matches(r.id, holdout)]
        if not held and len(mine) > 1:
            src = _source(mine[-1].id)
            held = [r for r in mine if _source(r.id) == src]
            logger.warning("holdout %s matches no %s run: holding out %s instead",
                           list(holdout), kind, src or mine[-1].id)
        if len(held) == len(mine):
            if mine and holdout:
                logger.warning("every %s run matches holdout %s: no %s test split",
                               kind, list(holdout), kind)
            held = []
        test += held
        train += [r for r in mine if r not in held]
    return train, test


def matrix(kind: str, runs: Sequence[Run]):
    import numpy as np

    rows: List[List[float]] = []
    for r in runs:
        if r.kind == kind:
            rows += feature_rows(kind, r.items)[1]
    return np.asarray(rows, dtype=float).reshape(-1, len(FEATURES_OF[kind]))


def fit(X, contamination: float):
    from sklearn.ensemble import IsolationForest

    return IsolationForest(contamination=contamination, **c.IFOREST_PARAMS).fit(X)


def train(runs: Sequence[Run], contamination: float = c.CONTAMINATION,
          holdout: Sequence[str] = DEFAULT_HOLDOUT) -> Dict[str, Dict]:
    """{kind: {"model", "meta"}} for each kind with training windows."""
    import sklearn

    train_runs, test_runs = split_runs(runs, holdout)
    out = {}
    for kind in (POSE, TRAINING):
        X = matrix(kind, train_runs)
        Xt = matrix(kind, test_runs)
        if len(X) < 2:
            logger.warning("no %s training windows: %s model not written", kind, MODEL_OF[kind])
            continue
        model = fit(X, contamination)
        names = FEATURES_OF[kind]
        out[kind] = {"model": model, "meta": {
            "model": MODEL_OF[kind],
            "features": list(names),
            "n_train": int(len(X)),
            "n_test": int(len(Xt)),
            "train_runs": sorted(r.id for r in train_runs if r.kind == kind),
            "test_runs": sorted(r.id for r in test_runs if r.kind == kind),
            "feature_mean": dict(zip(names, map(float, X.mean(axis=0)))),
            "feature_std": dict(zip(names, map(float, X.std(axis=0)))),
            "train_flag_rate": float((model.predict(X) == -1).mean()),
            "test_flag_rate": float((model.predict(Xt) == -1).mean()) if len(Xt) else None,
            "sklearn_version": sklearn.__version__,
        }}
    return out


def save(fitted: Dict[str, Dict], contamination: float, holdout: Sequence[str]) -> Dict:
    import joblib
    import sklearn

    os.makedirs(c.MODEL_DIR, exist_ok=True)
    for kind, name in MODEL_OF.items():
        if kind in fitted:
            joblib.dump(fitted[kind]["model"], c.model_path(name))
        elif os.path.exists(c.model_path(name)):
            # a model meta.json no longer describes must not be loaded by the processor
            os.remove(c.model_path(name))
            logger.warning("removed stale %s: no %s windows this time", c.model_path(name), kind)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sklearn_version": sklearn.__version__,
        "contamination": contamination,
        "iforest_params": c.IFOREST_PARAMS,
        "window": c.WINDOW,
        "min_window_to_score": c.MIN_WINDOW_TO_SCORE,
        "holdout": list(holdout),
        "models": {MODEL_OF[k]: f["meta"] for k, f in fitted.items()},
    }
    with open(meta_path(), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", default=[str(REPO_ROOT)],
                    help="directories to scan (or event / pose / recording files)")
    ap.add_argument("--holdout", nargs="+", default=list(DEFAULT_HOLDOUT),
                    help="run-id substrings kept out of training as the test split")
    ap.add_argument("--contamination", type=float, default=c.CONTAMINATION)
    ap.add_argument("--scan-timeout", type=float, default=SCAN_TIMEOUT_S)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    t0 = time.monotonic()
    runs = collect_runs(args.runs, args.scan_timeout)
    logger.info("%d pose runs, %d training runs in %.1f s",
                sum(r.kind == POSE for r in runs), sum(r.kind == TRAINING for r in runs),
                time.monotonic() - t0)
    fitted = train(runs, args.contamination, args.holdout)
    if not fitted:
        logger.error("no usable runs under %s", args.runs)
        return 1
    meta = save(fitted, args.contamination, args.holdout)
    for name, m in meta["models"].items():
        print("%-17s train %5d windows from %3d runs | test %4d windows from %d runs (%s) | "
              "flagged train %.1f%% test %s"
              % (name, m["n_train"], len(m["train_runs"]), m["n_test"], len(m["test_runs"]),
                 ", ".join(m["test_runs"])[:80] or "-", 100 * m["train_flag_rate"],
                 "-" if m["test_flag_rate"] is None else "%.1f%%" % (100 * m["test_flag_rate"])))
    print("wrote %s" % ", ".join([c.model_path(MODEL_OF[k]) for k in fitted] + [meta_path()]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
