"""Contamination experiment for the two anomaly models (Policy & Process Map row 12).

    held-out clean runs --inject known-bad events--> refit per contamination --> markdown table

    python -m hypersplat.streaming.experiment [--runs DIR ...] [--holdout PATTERN ...]

Uses the same discovery, split and features as train_models (it never touches the saved
models). For each held-out sequence, every injection is placed at several positions, each in
its own copy of the sequence; only the windows ending on an injected event are scored as
positives, and the clean sequence's windows are scored once as negatives. Keeping each
injection in its own copy stops one bad event's echo in the next window (the step back to
the true path) from counting as a false alarm or as a second catch.

Pose injections (contract.INJECTIONS):
  shuffle  swap the poses of two frames SHUFFLE_GAP apart (both frames labelled)
  blur     BLUR_LEN consecutive poses: confidence x BLUR_CONF_SCALE plus a camera-centre
           jitter of about one median step and ~BLUR_ROT_DEG of rotation jitter
Training injections ("training going wrong"):
  loss_spike    one snapshot's loss x SPIKE_SCALE
  loss_rise     loss stops falling and climbs RISE_PER_SNAPSHOT per snapshot for RISE_LEN
  gs_explosion  Gaussian count x EXPLODE_BASE per snapshot for EXPLODE_LEN, memory following

Contamination only moves IsolationForest's threshold (the trees are seeded by
contract.IFOREST_PARAMS), so the grid trades false alarms for recall on one fixed score.
"""

import argparse
import logging
import math
import os
import random
import statistics
import sys
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import contract as c
from . import train_models as tm

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                            "contamination.md")

SHUFFLE_GAP = 10
BLUR_LEN = 5
BLUR_CONF_SCALE = 0.1
BLUR_ROT_DEG = 3.0
SPIKE_SCALE = 4.0
RISE_LEN = 8
RISE_PER_SNAPSHOT = 0.2
EXPLODE_LEN = 6
EXPLODE_BASE = 1.5
# contract.INJECTIONS names only the pose kinds; the training kinds live here until it grows
TRAINING_INJECTIONS = ("loss_spike", "loss_rise", "gs_explosion")
PLACEMENTS = 8   # positions per injection kind per held-out sequence


# ---------------------------------------------------------------------------------------------
# Injections: (sequence, position, rng) -> (new sequence, injected positions)
# ---------------------------------------------------------------------------------------------

Injected = Tuple[List, List[int]]


def inject_shuffle(seq: Sequence[c.CameraPose], i: int, rng: random.Random,
                   gap: int = SHUFFLE_GAP) -> Injected:
    out = list(seq)
    a, b = out[i], out[i + gap]
    keys = ("qw", "qx", "qy", "qz", "tx", "ty", "tz")
    out[i] = replace(a, injected="shuffle", **{k: getattr(b, k) for k in keys})
    out[i + gap] = replace(b, injected="shuffle", **{k: getattr(a, k) for k in keys})
    return out, [i, i + gap]


def _median_step(seq: Sequence[c.CameraPose]) -> float:
    steps = [c.pose_features([a, b])["step_dist"] for a, b in zip(seq, seq[1:])]
    m = statistics.median(steps) if steps else 0.0
    return m if m > 0 else 0.01


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def inject_blur(seq: Sequence[c.CameraPose], i: int, rng: random.Random,
                k: int = BLUR_LEN) -> Injected:
    out = list(seq)
    sigma = _median_step(seq) / math.sqrt(3)   # per axis: about one step in 3-D
    for j in range(i, i + k):
        p = out[j]
        axis = [rng.gauss(0, 1) for _ in range(3)]
        n = math.sqrt(sum(v * v for v in axis)) or 1.0
        half = math.radians(rng.gauss(0, BLUR_ROT_DEG)) / 2
        dq = (math.cos(half),) + tuple(math.sin(half) * v / n for v in axis)
        q = _qmul(dq, (p.qw, p.qx, p.qy, p.qz))
        q = tuple(v / math.sqrt(sum(u * u for u in q)) for v in q)
        # Jitter the camera centre, then rebuild t = -R C: rotating R alone would swing the
        # centre by angle x |t|, far more than one step when the map origin is distant.
        centre = [v + rng.gauss(0, sigma) for v in c.camera_center(p)]
        R = c._rotmat(q)
        t = [-sum(R[r][k] * centre[k] for k in range(3)) for r in range(3)]
        out[j] = replace(p, qw=q[0], qx=q[1], qy=q[2], qz=q[3], tx=t[0], ty=t[1], tz=t[2],
                         confidence=p.confidence * BLUR_CONF_SCALE, injected="blur")
    return out, list(range(i, i + k))


def inject_spike(seq: Sequence[c.TrainingSnapshot], i: int, rng: random.Random) -> Injected:
    out = list(seq)
    out[i] = replace(out[i], loss=out[i].loss * SPIKE_SCALE, injected="loss_spike")
    return out, [i]


def inject_rise(seq: Sequence[c.TrainingSnapshot], i: int, rng: random.Random,
                k: int = RISE_LEN) -> Injected:
    out = list(seq)
    base = out[i - 1].loss
    for j in range(k):
        out[i + j] = replace(out[i + j], loss=base * (1 + RISE_PER_SNAPSHOT * (j + 1)),
                             injected="loss_rise")
    return out, list(range(i, i + k))


def inject_explosion(seq: Sequence[c.TrainingSnapshot], i: int, rng: random.Random,
                     k: int = EXPLODE_LEN) -> Injected:
    out = list(seq)
    n0, m0 = out[i - 1].num_gaussians, out[i - 1].gpu_mem_gb
    for j in range(k):
        f = EXPLODE_BASE ** (j + 1)
        out[i + j] = replace(out[i + j], num_gaussians=int(max(n0, 1) * f),
                             gpu_mem_gb=m0 * (1 + 0.5 * (f - 1)), injected="gs_explosion")
    return out, list(range(i, i + k))


# label -> (injector, events it spans from its start position)
INJECTORS: Dict[str, Dict[str, Tuple[Callable[..., Injected], int]]] = {
    tm.POSE: {"shuffle": (inject_shuffle, SHUFFLE_GAP + 1), "blur": (inject_blur, BLUR_LEN)},
    tm.TRAINING: {"loss_spike": (inject_spike, 1), "loss_rise": (inject_rise, RISE_LEN),
                  "gs_explosion": (inject_explosion, EXPLODE_LEN)},
}
assert tuple(INJECTORS[tm.POSE]) == c.INJECTIONS
assert tuple(INJECTORS[tm.TRAINING]) == TRAINING_INJECTIONS


def placements(n: int, span: int, count: int = PLACEMENTS) -> List[int]:
    """Up to count start positions spread over the sequence. The first leaves a full
    MIN_WINDOW_TO_SCORE of clean history so the injected event is actually scored."""
    lo, hi = c.MIN_WINDOW_TO_SCORE, n - span
    if hi < lo:
        return []
    if count < 2 or hi == lo:
        return [lo]
    return sorted({lo + k * (hi - lo) // (count - 1) for k in range(count)})


def eval_set(kind: str, runs: Sequence[tm.Run], seed: int = 0):
    """(X_clean, X_bad, labels of X_bad rows) from held-out runs of one kind."""
    import numpy as np

    rng = random.Random(seed)
    clean, bad, labels = [], [], []
    for run in runs:
        if run.kind != kind:
            continue
        clean += tm.feature_rows(kind, run.items)[1]
        for label, (inject, span) in INJECTORS[kind].items():
            for i in placements(len(run.items), span):
                seq, pos = inject(run.items, i, rng)
                rows = tm.feature_rows(kind, seq, pos)[1]
                bad += rows
                labels += [label] * len(rows)
    width = len(tm.FEATURES_OF[kind])
    return (np.asarray(clean, dtype=float).reshape(-1, width),
            np.asarray(bad, dtype=float).reshape(-1, width), labels)


def evaluate(X_train, X_clean, X_bad, labels: Sequence[str], contamination: float) -> Dict:
    model = tm.fit(X_train, contamination)
    bad_flag = model.predict(X_bad) == -1 if len(X_bad) else []
    clean_flag = model.predict(X_clean) == -1 if len(X_clean) else []
    caught = int(sum(bad_flag))
    false = int(sum(clean_flag))
    by_label: Dict[str, List[int]] = {}
    for lab, hit in zip(labels, bad_flag):
        by_label.setdefault(lab, [0, 0])
        by_label[lab][0] += int(hit)
        by_label[lab][1] += 1
    return {
        "contamination": contamination, "injected": len(labels), "caught": caught,
        "missed": len(labels) - caught, "false_alarms": false, "clean": len(clean_flag),
        "precision": caught / (caught + false) if caught + false else float("nan"),
        "recall": caught / len(labels) if labels else float("nan"),
        "by_label": by_label,
        "score_clean": float(model.decision_function(X_clean).mean()) if len(X_clean) else None,
        "score_bad": float(model.decision_function(X_bad).mean()) if len(X_bad) else None,
    }


def run_experiment(runs: Sequence[tm.Run], holdout: Sequence[str] = tm.DEFAULT_HOLDOUT,
                   grid: Sequence[float] = c.CONTAMINATION_GRID, seed: int = 0) -> Dict[str, Dict]:
    train_runs, test_runs = tm.split_runs(runs, holdout)
    out = {}
    for kind in (tm.POSE, tm.TRAINING):
        X = tm.matrix(kind, train_runs)
        X_clean, X_bad, labels = eval_set(kind, test_runs, seed)
        if len(X) < 2 or not labels:
            logger.warning("%s: %d training windows, %d injected: skipped", kind, len(X),
                           len(labels))
            continue
        out[kind] = {
            "n_train": len(X), "n_train_runs": sum(r.kind == kind for r in train_runs),
            "test_runs": sorted(r.id for r in test_runs if r.kind == kind),
            "rows": [evaluate(X, X_clean, X_bad, labels, g) for g in grid],
        }
    return out


def _pct(x: float) -> str:
    return "n/a" if x != x else "%.2f" % x


def to_markdown(results: Dict[str, Dict]) -> str:
    lines = ["# Contamination experiment (row 12)", ""]
    for kind, res in results.items():
        r0 = res["rows"][0]
        lines += [
            "## %s model (`%s`)" % (kind.capitalize(), tm.MODEL_OF[kind]), "",
            "Train: %d windows from %d runs. Test: %d clean windows + %d injected from %s."
            % (res["n_train"], res["n_train_runs"], r0["clean"], r0["injected"],
               ", ".join("`%s`" % t for t in res["test_runs"])), "",
            "| contamination | injected | caught | missed | false alarms | precision | recall |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in res["rows"]:
            lines.append("| %.2f | %d | %d | %d | %d / %d | %s | %s |"
                         % (r["contamination"], r["injected"], r["caught"], r["missed"],
                            r["false_alarms"], r["clean"], _pct(r["precision"]),
                            _pct(r["recall"])))
        lines.append("")
        lines.append("Recall by injection: " + "; ".join(
            "%.2f: %s" % (r["contamination"], ", ".join(
                "%s %d/%d" % (lab, hit, tot) for lab, (hit, tot) in sorted(r["by_label"].items())))
            for r in res["rows"]))
        lines.append("")
        lines.append("Mean decision_function at contamination %.2f (lower = more anomalous): "
                     "clean %.4f, injected %.4f."
                     % (r0["contamination"], r0["score_clean"], r0["score_bad"]))
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", default=[str(tm.REPO_ROOT)])
    ap.add_argument("--holdout", nargs="+", default=list(tm.DEFAULT_HOLDOUT))
    ap.add_argument("--grid", nargs="+", type=float, default=list(c.CONTAMINATION_GRID))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=RESULTS_PATH)
    ap.add_argument("--scan-timeout", type=float, default=tm.SCAN_TIMEOUT_S)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results = run_experiment(tm.collect_runs(args.runs, args.scan_timeout), args.holdout,
                             args.grid, args.seed)
    if not results:
        logger.error("nothing to evaluate under %s", args.runs)
        return 1
    md = to_markdown(results)
    print(md)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md + "\n")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
