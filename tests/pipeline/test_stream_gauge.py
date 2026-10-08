"""Gauge lock for ACE refits (hypersplat/stream/gauge.py)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hypersplat.stream.gauge import Sim3, fit_sim3


def _line_w2c(n=30, noise=0.0, seed=0):
    """Cameras along a line (drone-like), slowly yawing."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler("yxz", [i * 0.5, 2.0, 0.0], degrees=True).as_matrix()
        c2w[:3, 3] = [0.3 * i, 0.0, 0.05 * i] + noise * rng.standard_normal(3)
        out.append(np.linalg.inv(c2w))
    return np.stack(out)


def _sim3(seed=1):
    rng = np.random.default_rng(seed)
    return Sim3(1.7, Rotation.from_rotvec(rng.standard_normal(3) * 0.4).as_matrix(),
                rng.standard_normal(3) * 2)


def test_recovers_exact_similarity_on_a_line():
    ref = _line_w2c()
    S = _sim3()
    # The new map is the reference seen through S^-1: aligning must give S back
    Sinv = Sim3(1 / S.s, S.R.T, -S.R.T @ S.t / S.s)
    new = np.stack([Sinv.w2c(w) for w in ref])
    r = fit_sim3(new, ref)
    assert r.sim3.s == pytest.approx(S.s, rel=1e-6)
    np.testing.assert_allclose(r.sim3.R, S.R, atol=1e-6)
    np.testing.assert_allclose(r.sim3.t, S.t, atol=1e-6)
    assert r.center_max < 1e-6 and r.rot_max_deg < 1e-4
    assert r.raw_center_median > 1.0  # the raw "poses moved" would have reported a big drift


def test_outliers_do_not_pull_the_gauge():
    ref = _line_w2c(noise=0.002)
    S = _sim3(2)
    Sinv = Sim3(1 / S.s, S.R.T, -S.R.T @ S.t / S.s)
    new = np.stack([Sinv.w2c(w) for w in ref])
    # A refit fixed three poses by a lot
    for k in (3, 11, 20):
        c2w = np.linalg.inv(new[k])
        c2w[:3, 3] += [1.0, -0.5, 0.3]
        new[k] = np.linalg.inv(c2w)
    r = fit_sim3(new, ref)
    assert r.inliers == len(ref) - 3
    assert r.sim3.s == pytest.approx(S.s, rel=1e-3)
    assert r.center_median < 0.01


def test_points_and_cameras_stay_consistent():
    S = _sim3(3)
    w2c = _line_w2c(n=1)[0]
    X = np.array([[0.4, -0.2, 3.0], [1.0, 0.5, 5.0]])
    cam = X @ w2c[:3, :3].T + w2c[:3, 3]
    w2 = S.w2c(w2c)
    cam2 = S.points(X) @ w2[:3, :3].T + w2[:3, 3]
    # Same pixels: camera coordinates only scale by s
    np.testing.assert_allclose(cam2, S.s * cam, atol=1e-9)


def test_compose_and_too_few():
    a, b = _sim3(4), _sim3(5)
    X = np.random.default_rng(0).standard_normal((5, 3))
    np.testing.assert_allclose(a.compose(b).points(X), a.points(b.points(X)), atol=1e-9)
    assert fit_sim3(_line_w2c(n=2), _line_w2c(n=2)) is None


def test_a_reposed_cluster_does_not_capture_the_fit():
    """A refit re-poses a whole cluster of recently registered frames together; the fit must
    follow the majority that kept their poses. (Synthetic: the median-trim fit that reported
    a 9% false shrink on Kenilworth v3 passes this too - that failure needs the real pairs.)"""
    ref = _line_w2c(n=60, noise=0.001)
    S = _sim3(6)
    Sinv = Sim3(1 / S.s, S.R.T, -S.R.T @ S.t / S.s)
    new = np.stack([Sinv.w2c(w) for w in ref])
    # The last 40% of frames moved coherently: a shrink towards their own centroid
    c2w = np.linalg.inv(new[36:])
    cen = c2w[:, :3, 3].mean(0)
    c2w[:, :3, 3] = cen + 0.7 * (c2w[:, :3, 3] - cen) + [0.0, 0.4, 0.0]
    new[36:] = np.linalg.inv(c2w)
    r = fit_sim3(new, ref)
    assert r.sim3.s == pytest.approx(S.s, rel=2e-3)
    assert r.inliers >= 36
    assert np.median(r.all_res[:36]) < 0.01


def test_candidates_guard_against_a_worse_fit():
    ref = _line_w2c(n=20, noise=0.001)
    new = ref.copy()  # nothing moved: identity is the answer
    rng = np.random.default_rng(1)
    for k in range(0, 20, 2):  # half the frames re-posed at random
        c2w = np.linalg.inv(new[k])
        c2w[:3, 3] += rng.standard_normal(3)
        new[k] = np.linalg.inv(c2w)
    r = fit_sim3(new, ref, candidates=(Sim3(),))
    assert np.median(r.all_res) <= np.median(np.linalg.norm(
        np.linalg.inv(new)[:, :3, 3] - np.linalg.inv(ref)[:, :3, 3], axis=1)) + 1e-12
