"""Difix feedback loop: closed-loop orbit angle, pseudo-view trust, view count and loop exit.

Signals (all recorded during the run):
    difix.camera_spread_deg  angular extent of the real cameras seen from the scene pivot
                             (augment.measure_camera_spread, recorded before loop 1)
    difix.rounds             one entry per Difix round, appended by augment.run:
                             {loop, max_angle, mean_change_psnr, min_change_psnr, threshold,
                              n_views, n_kept}
    difix.loop_index         index of the loop about to run (set by the manager)
    train.val                held-out val rows, each carrying its result_dir
    frames.count             real training frames

"change PSNR" is PSNR(raw 3DGS render, Difix output) for one pseudo view: how much Difix had
to change the render. Validated on the drone scene with nvidia/difix_ref: 22.1 dB @3 deg,
20.6 @6, 19.7 @12 (faithful), while large changes (< ~15 dB) were hallucinated structure.
So a high change PSNR means "safe to go wider", a low one means "back off / trust less".

The per-view helpers below are also used by hypersplat/beta/difix/augment.py.
Stdlib-only and py3.8-compatible.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..resolve import Decision
from . import Param

# --- Difix trust bands (dB of change PSNR) ---
HALLUCINATION_FLOOR = 15.0  # below this Difix is inventing content, whatever the distribution
REL_MARGIN = 3.0            # a view this far below the round's median is an outlier
FULL_TRUST_PSNR = 21.0      # changes smaller than this are treated as pure artifact cleanup
MIN_VIEW_TRUST = 0.2        # lowest trust factor of a view that survives the threshold
WIDEN_ABOVE = 20.0          # last round's mean >= this: widen the orbit
SHRINK_BELOW = 17.0         # last round's mean < this: shrink the orbit
MIN_KEPT_FRACTION = 0.5     # fewer surviving views than this: the angle is too far out

# --- Orbit angle (degrees) ---
START_ANGLE = 6.0           # validated faithful for difix_ref; used when geometry is unknown
MIN_ANGLE = 2.0
FIRST_MIN_ANGLE = 3.0       # smallest validated-faithful angle; floor for round 1
ANGLE_CAP = 20.0            # previous hard-coded DIFIX_MAX_ANGLE
WIDEN_FACTOR = 1.5
SHRINK_FACTOR = 0.6

# --- Other knobs ---
BASE_PSEUDO_WEIGHT = 0.5    # previous hard-coded DIFIX_PSEUDO_WEIGHT
VIEWS_PER_FRAME = 0.5
MIN_VIEWS, MAX_VIEWS = 8, 48
MAX_LOOPS = 3               # previous hard-coded REFINE_LOOPS (upper bound when auto)
MIN_VAL_GAIN = 0.1          # dB of held-out PSNR a loop must add to justify another one


# ----------------------------------------------------------------------------
# Small stdlib statistics helpers
# ----------------------------------------------------------------------------

def _finite(values: Iterable) -> List[float]:
    out = []
    for v in values:
        if v is None:
            continue
        v = float(v)
        if math.isfinite(v):
            out.append(v)
    return out


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default), q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        raise ValueError("percentile of empty sequence")
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------------------------------------------------------
# Per-view helpers (used by augment.py)
# ----------------------------------------------------------------------------

def hallucination_threshold(change_psnrs: Iterable) -> float:
    """Change PSNR below which a pseudo view is dropped.

    Relative to the round: median minus max(2 robust sigma, REL_MARGIN), so only outliers in
    this round are dropped; never below the absolute HALLUCINATION_FLOOR.
    """
    xs = _finite(change_psnrs)
    if not xs:
        return HALLUCINATION_FLOOR
    med = percentile(xs, 50)
    sigma = 1.4826 * percentile([abs(x - med) for x in xs], 50)
    return max(HALLUCINATION_FLOOR, med - max(2.0 * sigma, REL_MARGIN))


def change_trust(change_psnr: Optional[float], threshold: float) -> float:
    """Trust factor of one pseudo view: 0 below the threshold, then a ramp from MIN_VIEW_TRUST
    at the threshold to 1 at FULL_TRUST_PSNR (more change -> less trust). Unmeasured (None or
    inf, e.g. raw renders without Difix) -> 1."""
    if change_psnr is None or not math.isfinite(float(change_psnr)):
        return 1.0
    c = float(change_psnr)
    if c < threshold:
        return 0.0
    if FULL_TRUST_PSNR <= threshold:
        return 1.0
    return _clip((c - threshold) / (FULL_TRUST_PSNR - threshold), MIN_VIEW_TRUST, 1.0)


def pseudo_view_weights(base_weights: Sequence[float], change_psnrs: Sequence,
                        threshold: Optional[float] = None) -> Tuple[List[float], float]:
    """Scale each pseudo view's base weight by its change trust; dropped views get 0.
    Returns (weights, threshold)."""
    if threshold is None:
        threshold = hallucination_threshold(change_psnrs)
    weights = [float(w) * change_trust(c, threshold) for w, c in zip(base_weights, change_psnrs)]
    return weights, threshold


def round_stats(loop, max_angle: float, angles: Sequence[float], change_psnrs: Sequence,
                threshold: float, n_kept: int) -> Dict:
    """The difix.rounds entry for one round."""
    xs = _finite(change_psnrs)
    return {
        "loop": loop,
        "max_angle": round(float(max_angle), 3),
        "mean_change_psnr": round(sum(xs) / len(xs), 3) if xs else None,
        "min_change_psnr": round(min(xs), 3) if xs else None,
        "threshold": round(float(threshold), 3),
        "n_views": len(change_psnrs),
        "n_kept": int(n_kept),
        "max_view_angle": round(max(angles), 3) if angles else None,
    }


def confidence_ramp(conf_values: Iterable, user_min_conf: Optional[float] = None,
                    user_floor: Optional[float] = None) -> Tuple[float, float]:
    """(min_conf, floor) for augment's ACE-Zero confidence ramp (min_conf -> floor, median -> 1).

    min_conf: the lower of the 5th percentile and half the median, so only the unreliable tail
        is floored and a tight distribution (all frames equally good) stays ~flat.
    floor: P5/median clipped to [0.1, 0.5] - a scene whose worst frames are close to typical
        keeps them at half weight, one with a long bad tail pushes them down to 0.1.
    Explicit user values win; (1000, 0.2) - the old constants - when there is no confidence.
    """
    xs = _finite(conf_values)
    min_conf, floor = 1000.0, 0.2
    if xs:
        med = percentile(xs, 50)
        p5 = percentile(xs, 5)
        min_conf = min(p5, 0.5 * med)
        if med > 0:
            floor = _clip(p5 / med, 0.1, 0.5)
    if user_min_conf is not None:
        min_conf = float(user_min_conf)
    if user_floor is not None:
        floor = float(user_floor)
    return min_conf, floor


# ----------------------------------------------------------------------------
# Profile readers
# ----------------------------------------------------------------------------

def _rounds(profile) -> List[dict]:
    rounds = profile.get("difix.rounds") if profile is not None else None
    return [r for r in rounds if isinstance(r, dict)] if isinstance(rounds, list) else []


def final_val_psnrs(profile) -> List[Tuple[str, float]]:
    """(result_dir, psnr at its last val step) per result dir, in the order they were trained."""
    rows = profile.get("train.val") if profile is not None else None
    last: Dict[str, Tuple[int, float]] = {}
    order: List[str] = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict) or r.get("psnr") is None:
            continue
        key = str(r.get("result_dir"))
        if key not in last:
            order.append(key)
        step = int(r.get("step", 0))
        if key not in last or step >= last[key][0]:
            last[key] = (step, float(r["psnr"]))
    return [(k, last[k][1]) for k in order]


def _trust_of_round(mean_change: float) -> float:
    return _clip((mean_change - HALLUCINATION_FLOOR) / (FULL_TRUST_PSNR - HALLUCINATION_FLOOR),
                 MIN_VIEW_TRUST, 1.0)


def _hallucinating(last: dict) -> Optional[str]:
    """Why the last round says Difix is hallucinating at its angle, or None."""
    mean = last.get("mean_change_psnr")
    if mean is not None and mean < HALLUCINATION_FLOOR:
        return f"mean change PSNR {mean:.1f} dB < {HALLUCINATION_FLOOR:.0f} dB floor"
    n, kept = last.get("n_views") or 0, last.get("n_kept")
    if n and kept is not None and kept < MIN_KEPT_FRACTION * n:
        return f"only {kept}/{n} pseudo views passed the hallucination threshold"
    return None


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------

def derive_max_angle(profile, config) -> Decision:
    rounds = _rounds(profile)
    if not rounds:
        spread = profile.get("difix.camera_spread_deg") if profile is not None else None
        if spread:
            angle = round(_clip(float(spread) / 2.0, FIRST_MIN_ANGLE, START_ANGLE), 1)
            return Decision(angle, "camera-spread",
                            f"first round: camera spread {float(spread):.1f}/2 deg, "
                            f"clamped [{FIRST_MIN_ANGLE:.0f}, {START_ANGLE:.0f}]")
        return Decision(START_ANGLE, "faithful-start",
                        f"first round, camera spread unknown: {START_ANGLE:.0f} deg is validated faithful")
    last = rounds[-1]
    prev = float(last.get("max_angle") or START_ANGLE)
    mean = last.get("mean_change_psnr")
    if mean is None:
        return Decision(prev, "hold", "last round has no change PSNR (Difix off?)")
    halluc = _hallucinating(last)
    if mean < SHRINK_BELOW or halluc:
        angle = round(max(prev * SHRINK_FACTOR, MIN_ANGLE), 1)
        return Decision(angle, "shrink",
                        f"last round {prev:.1f} deg: {halluc or f'mean change PSNR {mean:.1f} dB < {SHRINK_BELOW:.0f}'}")
    if mean >= WIDEN_ABOVE:
        angle = round(min(prev * WIDEN_FACTOR, ANGLE_CAP), 1)
        return Decision(angle, "widen",
                        f"last round {prev:.1f} deg: mean change PSNR {mean:.1f} dB >= {WIDEN_ABOVE:.0f} (faithful)")
    return Decision(round(prev, 1), "hold",
                    f"last round {prev:.1f} deg: mean change PSNR {mean:.1f} dB in "
                    f"[{SHRINK_BELOW:.0f}, {WIDEN_ABOVE:.0f})")


def derive_num_views(profile, config) -> Optional[Decision]:
    frames = profile.get("frames.count") if profile is not None else None
    if not frames:
        return None
    n = int(_clip(round(VIEWS_PER_FRAME * frames), MIN_VIEWS, MAX_VIEWS))
    return Decision(n, "frames-fraction",
                    f"{VIEWS_PER_FRAME} x {frames} train frames, clamped [{MIN_VIEWS}, {MAX_VIEWS}]")


def derive_pseudo_weight(profile, config) -> Decision:
    rounds = _rounds(profile)
    mean = rounds[-1].get("mean_change_psnr") if rounds else None
    if mean is None:
        return Decision(BASE_PSEUDO_WEIGHT, "no-round-yet",
                        "no Difix change PSNR measured yet; base weight")
    w = round(BASE_PSEUDO_WEIGHT * _trust_of_round(float(mean)), 3)
    return Decision(w, "change-trust",
                    f"{BASE_PSEUDO_WEIGHT} x trust of last round's mean change PSNR {mean:.1f} dB "
                    f"(ramp {HALLUCINATION_FLOOR:.0f}->{FULL_TRUST_PSNR:.0f} dB)")


def derive_refine_loops(profile, config) -> Decision:
    """Total number of loops: MAX_LOOPS unless the loop should stop now, in which case the
    value is the index of the loop about to run (so the manager's `while i < REFINE_LOOPS`
    exits)."""
    rounds = _rounds(profile)
    idx = profile.get("difix.loop_index") if profile is not None else None
    done = int(idx) if idx is not None else len(rounds)
    psnrs = final_val_psnrs(profile)
    if done > 0 and len(psnrs) >= 2:
        gain = psnrs[-1][1] - psnrs[-2][1]
        if gain < MIN_VAL_GAIN:
            return Decision(done, "val-plateau",
                            f"held-out PSNR {psnrs[-2][1]:.2f} -> {psnrs[-1][1]:.2f} dB "
                            f"(gain {gain:+.2f} < {MIN_VAL_GAIN})")
    if rounds:
        halluc = _hallucinating(rounds[-1])
        if halluc:
            return Decision(done, "difix-hallucinating", halluc)
    if len(psnrs) >= 2:
        reason = f"held-out PSNR still improving ({psnrs[-1][1] - psnrs[-2][1]:+.2f} dB)"
    else:
        reason = "no refinement result to compare yet"
    return Decision(MAX_LOOPS, "improving", f"{reason}; up to {MAX_LOOPS} loops")


PARAMS = [
    Param("REFINE_LOOPS", "loop", derive_refine_loops),
    Param("DIFIX_NUM_VIEWS", "loop", derive_num_views),
    Param("DIFIX_MAX_ANGLE", "loop", derive_max_angle),
    Param("DIFIX_PSEUDO_WEIGHT", "loop", derive_pseudo_weight),
]
