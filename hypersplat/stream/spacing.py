"""Per-source keyframe spacing: how far apart (in rotation) the accepted keyframes are.

Measured on a handheld phone stream the adjacent-keyframe rotation was ~1.3 deg median and
~4.4 deg p90, against ~0.4-0.6 deg on the drone clip. A fixed neighbour radius tuned on one
source is wrong on the other, so each stream measures its own spacing and derives the radius
from it. Pairs with few inliers are not trusted: a rotation estimated from 3 matches is noise.

Everything here is numpy / OpenCV and runs on CPU.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

DEFAULT_MIN_INLIERS = 30
MIN_SAMPLES = 5           # below this many trusted pairs the radius falls back to the default


Features = Tuple[np.ndarray, Optional[np.ndarray]]   # (N x 2 keypoint positions, N x 128 descriptors)


def sift_features(img: np.ndarray) -> Features:
    """SIFT keypoint positions and descriptors of one RGB frame (descriptors None if none found).

    Split out of ``relative_rotation_deg`` so a stream can keep the previous keyframe's
    features instead of recomputing them for every new pair.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kps, desc = cv2.SIFT_create(nfeatures=2000).detectAndCompute(gray, None)
    return np.asarray(cv2.KeyPoint_convert(kps), np.float32).reshape(-1, 2), desc


def relative_rotation_from_features(fa: Features, fb: Features, size: Tuple[int, int],
                                    focal: Optional[float] = None) -> Tuple[Optional[float], int]:
    """``relative_rotation_deg`` on precomputed ``sift_features``; ``size`` is the (w, h) of
    the first frame, which sets the intrinsics."""
    w, h = size
    if focal is None:
        focal = max(w, h) / (2.0 * np.tan(np.radians(35.0)))  # the 70 deg HFOV rule
    pp = (w / 2.0, h / 2.0)
    K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1.0]])
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return None, 0
    try:
        raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    except cv2.error:
        return None, 0
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, 0
    pa = ka[[m.queryIdx for m in good]]
    pb = kb[[m.trainIdx for m in good]]
    E, mask = cv2.findEssentialMat(pa, pb, focal=focal, pp=pp, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None or mask is None or E.shape != (3, 3):
        return None, 0
    n, R, _t = cv2.recoverPose(E, pa, pb, K, distanceThresh=1e6, mask=mask.copy())[:3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return angle, int(n)


def relative_rotation_deg(img_a: np.ndarray, img_b: np.ndarray,
                          focal: Optional[float] = None) -> Tuple[Optional[float], int]:
    """Rotation angle (deg) between two RGB frames, and the inlier count behind it.

    SIFT + ratio test + essential matrix, as in ``init_all_poses``. recoverPose is called with
    a large distanceThresh: the default (50) drops far points, which on near rotation-only
    motion flips the estimate to ~180 deg. Returns (None, 0) when the pair cannot be solved.
    Identical frames give ~0 deg with 0 inliers (zero baseline): the tracker rejects them.
    """
    h, w = img_a.shape[:2]
    return relative_rotation_from_features(sift_features(img_a), sift_features(img_b), (w, h), focal)


@dataclass
class SpacingTracker:
    """Rolling record of the rotation between adjacent accepted keyframes.

    ``eps()`` is the p90 of the trusted pairs (a neighbour radius that most true adjacent
    keyframes fall inside), or ``default`` until ``MIN_SAMPLES`` trusted pairs exist.
    """
    min_inliers: int = DEFAULT_MIN_INLIERS
    history: int = 200
    default_eps: float = 1.0

    def __post_init__(self):
        self._angles: deque = deque(maxlen=self.history)
        self.rejected = 0

    def add(self, angle_deg: Optional[float], inliers: int) -> bool:
        """Record one adjacent-keyframe pair. Returns whether it was trusted."""
        if angle_deg is None or inliers < self.min_inliers:
            self.rejected += 1
            return False
        self._angles.append(float(angle_deg))
        return True

    @property
    def samples(self) -> int:
        return len(self._angles)

    def eps(self) -> float:
        if self.samples < MIN_SAMPLES:
            return self.default_eps
        return float(np.percentile(np.asarray(self._angles), 90))

    def median(self) -> Optional[float]:
        return float(np.median(self._angles)) if self._angles else None

    def status(self) -> dict:
        return {"spacing_samples": self.samples, "spacing_rejected": self.rejected,
                "spacing_median_deg": self.median(), "spacing_eps_deg": round(self.eps(), 3)}
