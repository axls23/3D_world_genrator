"""Gauge lock for ACE refits: keep every map version in the bootstrap map's frame.

ACE-Zero reconstructs up to a similarity (Brachmann et al., ECCV 2024): the seed image sits at
identity only in the first iteration, and later mapping iterations fine-tune the scene model
with nothing pinning rotation, translation or scale. Offline that is harmless - only the final
poses matter. In a stream the 3DGS trainer keeps training in the frame it froze at bootstrap,
so a refit whose whole reconstruction drifted by a similarity moves every camera under the
Gaussians. A refit is aligned back with the Sim(3) that best maps its poses onto the previous
ones; what remains after alignment is the refit's actual pose correction.

Convention: w2c 4x4 (OpenCV), X_cam = R w2c X_world + t. A Sim(3) S = (s, R, t) maps the new
map's world into the reference world: X_ref = s R X_new + t.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class Sim3:
    s: float = 1.0
    R: np.ndarray = None
    t: np.ndarray = None

    def __post_init__(self):
        self.R = np.eye(3) if self.R is None else np.asarray(self.R, np.float64)
        self.t = np.zeros(3) if self.t is None else np.asarray(self.t, np.float64)

    def points(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64).reshape(-1, 3)
        return self.s * X @ self.R.T + self.t

    def w2c(self, w2c: np.ndarray) -> np.ndarray:
        """The same camera expressed in the reference world. Camera coordinates scale by s,
        which leaves the projection unchanged."""
        w2c = np.asarray(w2c, np.float64).reshape(4, 4)
        out = np.eye(4)
        out[:3, :3] = w2c[:3, :3] @ self.R.T
        out[:3, 3] = self.s * w2c[:3, 3] - out[:3, :3] @ self.t
        return out

    def compose(self, other: "Sim3") -> "Sim3":
        """self after other: X -> self(other(X))."""
        return Sim3(self.s * other.s, self.R @ other.R, self.s * self.R @ other.t + self.t)

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(self.R) - 1) / 2, -1, 1))))


def _centers(w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def _project_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


@dataclass
class AlignResult:
    sim3: Sim3
    inliers: int
    used: int
    # Residuals after alignment, over the inlier set: camera centre distance (reference units)
    # and viewing-rotation angle (degrees)
    center_median: float
    center_p90: float
    center_max: float
    rot_median_deg: float
    rot_max_deg: float
    # Same distances without alignment, for comparison with the raw "poses moved"
    raw_center_median: float
    raw_center_max: float
    # Aligned centre distance of every frame (inliers and trimmed ones), same order as input
    all_res: np.ndarray = None


def _fit_on(ca, cb, Ra, Rb, idx) -> Sim3:
    """Least-squares Sim(3) on frames ``idx``: rotation from orientations (chordal mean), then
    scale and translation from the centres."""
    R = _project_rotation(np.einsum("nij,nkj->ik", Rb[idx], Ra[idx]))
    a, b = ca[idx] @ R.T, cb[idx]
    da, db = a - a.mean(0), b - b.mean(0)
    var = (da * da).sum()
    s = float((da * db).sum() / var) if var > 1e-12 else 1.0
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return Sim3(s, R, b.mean(0) - s * a.mean(0))


def _residuals(S: Sim3, ca, cb, Ra, Rb):
    res = np.linalg.norm(S.points(ca) - cb, axis=1)
    Rn = np.einsum("ij,njk->nik", S.R, Ra)
    cosang = (np.einsum("nij,nij->n", Rn, Rb) - 1) / 2
    return res, np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def fit_sim3(w2c_new: np.ndarray, w2c_ref: np.ndarray, tol: float = 0.01, rot_tol_deg: float = 2.0,
             hypotheses: int = 400, candidates: Sequence[Sim3] = (), seed: int = 0) -> Optional[AlignResult]:
    """Robust Sim(3) mapping the new map's poses onto the reference poses (same frames).

    A refit legitimately re-poses some frames by a lot - often a whole cluster of recently
    registered ones - and those must not pull the gauge. Least squares with a trim relative to
    the median got captured by such clusters (a 9% false shrink on Kenilworth), so this is
    RANSAC with an absolute tolerance: hypotheses from frame pairs (rotation from their
    orientations, scale and translation from their centres), scored by the frames within
    `tol` x path extent and `rot_tol_deg`, refined by least squares on the winner's inliers.

    Rotation comes from orientations, not centres: a drone or a walk is close to a line,
    where centres alone leave the roll about that line free.

    `candidates` (e.g. identity, the previous gauge) compete with the fit on the median aligned
    distance over all frames; the best one is returned, so the alignment never makes the
    cameras agree less with the reference than a candidate does.
    """
    A = np.asarray(w2c_new, np.float64).reshape(-1, 4, 4)
    B = np.asarray(w2c_ref, np.float64).reshape(-1, 4, 4)
    n = len(A)
    if n < 3:
        return None
    ca, cb = _centers(A), _centers(B)
    Ra = np.transpose(A[:, :3, :3], (0, 2, 1))  # c2w rotations
    Rb = np.transpose(B[:, :3, :3], (0, 2, 1))
    extent = float(np.linalg.norm(cb.max(0) - cb.min(0))) or 1.0
    thr = tol * extent

    def score(S):
        res, rot = _residuals(S, ca, cb, Ra, Rb)
        inl = (res < thr) & (rot < rot_tol_deg)
        return inl, inl.sum(), np.median(res[inl]) if inl.any() else np.inf

    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > hypotheses:
        pairs = [pairs[k] for k in rng.choice(len(pairs), hypotheses, replace=False)]
    best = (None, -1, np.inf, None)
    for i, j in pairs:
        if np.linalg.norm(cb[i] - cb[j]) < 0.05 * extent:
            continue  # too short a baseline to fix the scale
        S = _fit_on(ca, cb, Ra, Rb, np.array([i, j]))
        inl, k, med = score(S)
        if k > best[1] or (k == best[1] and med < best[2]):
            best = (S, k, med, inl)
    S, keep = (best[0], best[3]) if best[0] is not None and best[1] >= 3 else (_fit_on(ca, cb, Ra, Rb, np.arange(n)), None)
    for _ in range(3):  # refine on the consensus set
        if keep is None or keep.sum() < 3:
            break
        S = _fit_on(ca, cb, Ra, Rb, np.where(keep)[0])
        new_keep, _, _ = score(S)
        if new_keep.sum() < 3 or (new_keep == keep).all():
            break
        keep = new_keep
    # Never worse than a candidate (identity, previous gauge) on all frames
    for C in candidates:
        if np.median(_residuals(C, ca, cb, Ra, Rb)[0]) < np.median(_residuals(S, ca, cb, Ra, Rb)[0]):
            S = C
    all_res, all_rot = _residuals(S, ca, cb, Ra, Rb)
    keep = (all_res < thr) & (all_rot < rot_tol_deg)
    if keep.sum() == 0:
        keep = np.ones(n, bool)
    res, rot = all_res[keep], all_rot[keep]
    raw = np.linalg.norm(ca - cb, axis=1)
    return AlignResult(S, int(keep.sum()), n, float(np.median(res)), float(np.percentile(res, 90)),
                       float(res.max()), float(np.median(rot)), float(rot.max()),
                       float(np.median(raw)), float(raw.max()), all_res)
