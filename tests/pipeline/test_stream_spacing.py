"""Keyframe spacing (hypersplat/stream/spacing.py): CPU-only, synthetic textured frames."""

import cv2
import numpy as np
import pytest

import hypersplat.stream.spacing as spacing_mod
from hypersplat.stream.spacing import (MIN_SAMPLES, SpacingTracker, relative_rotation_deg,
                                       relative_rotation_from_features, sift_features)


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


def test_wrapper_matches_split_functions():
    img = _texture()
    focal = max(img.shape[:2]) / (2.0 * np.tan(np.radians(35.0)))
    moved = _camera_rotation_warp(img, 2.0, focal)
    size = (img.shape[1], img.shape[0])
    split = relative_rotation_from_features(sift_features(img), sift_features(moved), size)
    assert split[0] is not None
    assert relative_rotation_deg(img, moved) == split
    assert relative_rotation_deg(img, moved, focal=focal) == \
        relative_rotation_from_features(sift_features(img), sift_features(moved), size, focal=focal)
    blank = np.zeros_like(img)
    assert relative_rotation_from_features(sift_features(blank), sift_features(img), size) == (None, 0)


def _spacing_session(tmp_path, **kw):
    from hypersplat.stream.session import StreamConfig, StreamSession
    src = tmp_path / "src.mp4"
    src.touch()   # a file source is only opened on start()
    s = StreamSession(StreamConfig(source=str(src), work_dir=str(tmp_path / "w"), **kw))
    s.frames_dir.mkdir(parents=True)
    return s


def test_session_describes_each_keyframe_once(tmp_path, monkeypatch):
    from hypersplat.stream import session as session_mod

    calls, resized = [], []
    real_sift, real_res = spacing_mod.sift_features, session_mod.StreamSession._ace_res

    def counting_sift(img):
        calls.append(img.shape)
        return real_sift(img)

    def counting_res(self, image):
        resized.append(image.shape)
        return real_res(self, image)

    monkeypatch.setattr(session_mod, "sift_features", counting_sift)
    monkeypatch.setattr(session_mod.StreamSession, "_ace_res", counting_res)
    # Bootstrap target out of reach: _add_keyframe only stores and writes the ACE copy
    s = _spacing_session(tmp_path, adaptive_spacing=True, ace_short_side=135, bootstrap_frames=100)
    base = _texture()
    focal = 240 / (2.0 * np.tan(np.radians(35.0)))   # 70 deg HFOV at the ACE res (240 x 135);
    # the 480 x 270 source frames are warped with twice that focal
    n = 6
    for i in range(n):
        s._add_keyframe(_camera_rotation_warp(base, 1.5 * i, 2 * focal), float(i), 1.0)
    # One resize and one SIFT pass per keyframe, at ACE res; one rotation per adjacent pair
    assert len(resized) == n and len(calls) == n and all(c[:2] == (135, 240) for c in calls)
    assert s.file_size == (240, 135)
    assert s.spacing.samples + s.spacing.rejected == n - 1
    assert s.spacing.samples >= 1 and s.spacing.median() == pytest.approx(1.5, abs=0.5)
    s.stop()


def test_stop_closes_offers_log(tmp_path):
    s = _spacing_session(tmp_path, debug_offered_dir=str(tmp_path / "offers"))
    log = s._offer_log
    assert log is not None and not log.closed
    s.stop()
    assert log.closed and s._offer_log is None
    s.stop()   # idempotent
