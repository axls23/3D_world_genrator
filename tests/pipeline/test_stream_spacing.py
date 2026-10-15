"""Keyframe spacing (hypersplat/stream/spacing.py): CPU-only, synthetic textured frames."""

import cv2
import numpy as np
import pytest

from hypersplat.stream.spacing import MIN_SAMPLES, SpacingTracker, relative_rotation_deg


def _texture(h=270, w=480, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h, w), dtype=np.uint8)
    blurred = cv2.GaussianBlur(base, (0, 0), 1.2)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)


def _camera_rotation_warp(img, deg, focal):
    """Warp for a pure camera rotation of ``deg`` about the vertical axis: H = K R K^-1."""
    h, w = img.shape[:2]
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    H = K @ R @ np.linalg.inv(K)
    return cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_identical_frames_have_no_rotation():
    img = _texture()
    angle, inliers = relative_rotation_deg(img, img.copy())
    assert angle is not None and angle < 0.2
    # zero baseline: nothing triangulates, so no inliers; the tracker rejects such pairs
    assert inliers == 0


def test_recovers_known_camera_rotation():
    img = _texture()
    w = img.shape[1]
    focal = max(img.shape[:2]) / (2.0 * np.tan(np.radians(35.0)))
    moved = _camera_rotation_warp(img, 2.0, focal)
    angle, inliers = relative_rotation_deg(img, moved, focal=focal)
    assert angle is not None
    assert angle == pytest.approx(2.0, abs=0.5)
    assert inliers >= 30


def test_unsolvable_pair_returns_none():
    blank = np.zeros((270, 480, 3), np.uint8)
    angle, inliers = relative_rotation_deg(blank, _texture())
    assert angle is None and inliers == 0


def test_tracker_default_until_enough_trusted_pairs():
    t = SpacingTracker(min_inliers=30, default_eps=1.5)
    assert t.eps() == 1.5
    for _ in range(MIN_SAMPLES - 1):
        t.add(0.5, 100)
    assert t.eps() == 1.5


def test_tracker_rejects_weak_pairs_and_uses_p90():
    t = SpacingTracker(min_inliers=30, default_eps=1.0)
    assert t.add(50.0, 3) is False          # 3 inliers: noise, never trusted
    assert t.add(None, 0) is False
    for a in [1.0, 1.2, 1.1, 1.3, 4.0, 1.25]:
        assert t.add(a, 200)
    assert t.samples == 6 and t.rejected == 2
    assert t.eps() == pytest.approx(np.percentile([1.0, 1.2, 1.1, 1.3, 4.0, 1.25], 90))
    st = t.status()
    assert st["spacing_samples"] == 6 and st["spacing_rejected"] == 2
