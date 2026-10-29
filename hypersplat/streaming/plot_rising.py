"""Report figure for the rising-loss trend rule: the loss of one held-out run with a planted slow climb,
what the Isolation Forest flags (nothing) and what the trend rule flags.

    python -m hypersplat.streaming.plot_rising --runs <dirs or files> [--out results/rising_loss.png]

Needs models/training_iforest.joblib (train_models). Output goes under the git-ignored results/ dir.
"""

import argparse
import os
import random
from typing import Optional, Sequence

from hypersplat.streaming import contract as c
from hypersplat.streaming import experiment as ex
from hypersplat.streaming import train_models as tm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "rising_loss.png")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", default=[str(tm.REPO_ROOT)])
    ap.add_argument("--holdout", nargs="+", default=list(tm.DEFAULT_HOLDOUT))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    _, test = tm.split_runs(tm.collect_runs(args.runs), args.holdout)
    runs = sorted((r for r in test if r.kind == tm.TRAINING), key=lambda r: -len(r.items))
    if not runs:
        print("no held-out training run found")
        return 1
    run = runs[0]
    start = len(run.items) // 2
    seq, planted = ex.inject_rise(run.items, start, random.Random(0))
    model = joblib.load(c.model_path(c.TRAINING_MODEL))
    est = model["model"] if isinstance(model, dict) else model

    steps = [s.step for s in seq]
    losses = [s.loss for s in seq]
    forest = [i for i in range(len(seq)) if len(tm.window_at(seq, i)) >= c.MIN_WINDOW_TO_SCORE
              and est.predict([[c.training_features(tm.window_at(seq, i))[n] for n in c.TRAINING_FEATURES]])[0] == -1]
    trend = [i for i in range(len(seq)) if c.rising_loss(tm.window_at(seq, i))[0]]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.plot(steps, losses, color="#1f6fb4", lw=1.6, label="training loss")
    ax.axvspan(steps[planted[0]], steps[planted[-1]], color="#f3b95f", alpha=0.25, label="planted slow climb")
    ax.scatter([steps[i] for i in forest], [losses[i] for i in forest], marker="x", color="#7a7a7a", s=40,
               label="Isolation Forest flags (%d)" % len(forest), zorder=3)
    ax.scatter([steps[i] for i in trend], [losses[i] for i in trend], color="#d62728", s=26,
               label="trend rule flags (%d)" % len(trend), zorder=4)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Rising-loss detection on a held-out run")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out)
    print("wrote %s (planted %d snapshots, forest flagged %d, trend rule flagged %d)"
          % (args.out, len(planted), len(forest), len(trend)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
