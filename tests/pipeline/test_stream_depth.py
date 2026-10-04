"""CPU tests for dense stream depth (hypersplat/stream/depth.py): alignment of a relative
disparity prediction to sparse ACE points, sharpening, flying-pixel masking, unprojection."""

import cv2
import numpy as np

from hypersplat.stream.depth import (align_disparity, dense_depth, flying_pixel_mask,
                                     guided_filter, unproject)

H, W = 120, 160
K = np.array([[140.0, 0, W / 2], [0, 140.0, H / 2], [0, 0, 1]])


def scene():
    """Floor-ish slanted plane with a box in front: true depth, and an RGB image whose
    edges coincide with the depth edges."""
    ys, xs = np.mgrid[0:H, 0:W]
    z = 4.0 + 2.0 * ys / H
    box = (xs > 60) & (xs < 110) & (ys > 30) & (ys < 90)
    z = np.where(box, 2.0, z).astype(np.float32)
    rgb = np.full((H, W, 3), 90, np.uint8)
    rgb[box] = (230, 230, 230)
    rgb[~box] += (np.arange(W)[None, :] % 20 < 10).astype(np.uint8)[..., None].repeat(H, 0)[~box] * 30
    return z, rgb, box


def network_like(z, a=0.7, b=-0.05, blur=5, seed=0):
    """Relative disparity as a network outputs it: affine in 1/z, soft edges, a little noise."""
    d = (1.0 / z - b) / a
    d = cv2.GaussianBlur(d.astype(np.float32), (0, 0), blur)
    return d + np.random.default_rng(seed).normal(0, 0.002, d.shape).astype(np.float32)


def ace_points(z, n=400, noise=0.01, seed=1, outliers=0.0):
    rng = np.random.default_rng(seed)
    ys, xs = rng.integers(5, H - 5, n), rng.integers(5, W - 5, n)
    zz = z[ys, xs] * (1 + rng.normal(0, noise, n))
    k = int(outliers * n)
    zz[:k] *= rng.uniform(2, 4, k)  # grossly wrong scene coordinates
    return np.stack([(xs + 0.5) / W, (ys + 0.5) / H], 1).astype(np.float32), zz


def test_align_recovers_metric_depth_and_ignores_outliers():
    z, rgb, box = scene()
    disp = network_like(z, blur=0.01)
    pix, zz = ace_points(z, outliers=0.2)
    xs, ys = (pix[:, 0] * W).astype(int), (pix[:, 1] * H).astype(int)
    al = align_disparity(disp[ys, xs], zz)
    assert al.ok, al.reason
    assert abs(al.a - 0.7) < 0.05 and abs(al.b + 0.05) < 0.01 and al.rel_err < 0.05
    assert 0.7 < al.inliers < 0.85  # the 20 % injected outliers are rejected
    pred = 1.0 / (al.a * disp + al.b)
    assert np.median(np.abs(pred - z) / z) < 0.03


def test_align_rejects_bad_frames():
    z, _, _ = scene()
    pix, zz = ace_points(z, n=20)
    xs, ys = (pix[:, 0] * W).astype(int), (pix[:, 1] * H).astype(int)
    assert "ACE points" in align_disparity(network_like(z)[ys, xs], zz).reason
    pix, zz = ace_points(z)
    xs, ys = (pix[:, 0] * W).astype(int), (pix[:, 1] * H).astype(int)
    noise = np.random.default_rng(3).uniform(0, 1, len(zz))  # prediction unrelated to the scene
    assert not align_disparity(noise, zz).ok


def test_guided_filter_sharpens_blurred_edges():
    z, rgb, box = scene()
    true = 1.0 / z
    blurred = cv2.GaussianBlur(true, (0, 0), 3)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sharp = guided_filter(grey, blurred, radius=4, eps=1e-4)
    edge = cv2.dilate((cv2.Canny((box * 255).astype(np.uint8), 50, 150) > 0).astype(np.uint8), np.ones((5, 5)))
    err = lambda d: np.abs(d - true)[edge > 0].mean()
    assert err(sharp) < 0.7 * err(blurred)


def test_dense_depth_masks_flying_pixels_and_is_accurate():
    z, rgb, box = scene()
    pix, zz = ace_points(z)
    res = dense_depth(network_like(z, blur=2), rgb, pix, zz)
    assert res.align.ok, res.align.reason
    v = res.valid
    assert 0.6 < v.mean() < 0.99
    boundary = cv2.dilate((cv2.Canny((box * 255).astype(np.uint8), 50, 150) > 0).astype(np.uint8), np.ones((3, 3))) > 0
    assert v[boundary].mean() < 0.2            # discontinuity pixels are dropped
    interior = v & ~cv2.dilate(boundary.astype(np.uint8), np.ones((9, 9))).astype(bool)
    assert np.median(np.abs(res.depth[interior] - z[interior]) / z[interior]) < 0.03
    assert (res.depth[~v] == 0).all()


def test_flying_pixel_mask_flat_is_all_valid():
    assert flying_pixel_mask(np.full((40, 50), 3.0, np.float32)).all()


def test_unproject_round_trip():
    z, rgb, _ = scene()
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, -2.0, 0.5]
    pts, cols = unproject(z, np.ones_like(z, bool), K, c2w, rgb, stride=10)
    cam = pts - c2w[:3, 3]
    uv = (cam @ K.T)
    uv = uv[:, :2] / uv[:, 2:]
    assert np.allclose(cam[:, 2], z[5::10, 5::10].ravel(), atol=1e-4)
    assert np.allclose((uv[:, 0] - 0.5) % 10, 5, atol=1e-3)
    assert cols.shape == pts.shape and cols.max() <= 1.0


def test_depth_preview_publishes_new_frames_only_and_stops():
    import time as _time

    import cv2 as _cv2

    from hypersplat.stream.depth import DepthPreview, depth_preview_jpeg

    jpg = depth_preview_jpeg(np.linspace(0, 1, 120 * 160, dtype=np.float32).reshape(120, 160))
    assert _cv2.imdecode(np.frombuffer(jpg, np.uint8), _cv2.IMREAD_COLOR).shape == (120, 160, 3)

    class FakeEstimator:
        calls = 0

        def predict(self, rgb):
            FakeEstimator.calls += 1
            return rgb[..., 0].astype(np.float32)

    frame = {"item": (np.zeros((60, 80, 3), np.uint8), 0.0, False, 1)}
    got = []
    prev = DepthPreview(FakeEstimator(), lambda: frame["item"], got.append, fps=50).start()
    _time.sleep(0.2)
    assert FakeEstimator.calls == 1 and len(got) == 1          # same frame: not recomputed
    frame["item"] = (np.full((60, 80, 3), 50, np.uint8), 0.1, False, 2)
    _time.sleep(0.2)
    prev.stop()
    assert FakeEstimator.calls == 2 and prev.frames == 2
    n = FakeEstimator.calls
    frame["item"] = (np.zeros((60, 80, 3), np.uint8), 0.2, False, 3)
    _time.sleep(0.1)
    assert FakeEstimator.calls == n                              # stopped


def test_bootstrap_cooldown_caps_first_map_only():
    import types

    from hypersplat.stream.session import StreamConfig, StreamSession

    def schedule(cooldown):
        s = StreamSession.__new__(StreamSession)
        s.cfg = StreamConfig(source="x.mp4", bootstrap_cooldown=cooldown)
        s.file_size = (854, 480)
        s.gate = types.SimpleNamespace(avg_motion=8.0)
        return s._ace_schedule(24)

    base = schedule(None)
    capped = schedule(500)
    assert base["cooldown_iterations"] > 500 and capped["cooldown_iterations"] == 500
    assert capped["hybrid_train_iterations"] == base["hybrid_train_iterations"]


def _depth_session():
    import threading
    import types

    from hypersplat.stream.session import StreamConfig, StreamSession

    s = StreamSession.__new__(StreamSession)
    s.cfg = StreamConfig(source="x.mp4", depth=True)
    s._cond = threading.Condition()
    s.transform = np.eye(4)
    s._depth, s._disp, s._ace_pts, s._aligned_version = {}, {}, {}, {}
    s._depth_jobs, s._refit_requested = [], 0
    s.depth_counters = {"maps": 0, "rejected": 0, "ms": 0.0, "rel_err": None, "last_reject": None, "realigned": 0}
    kfs = {k: types.SimpleNamespace(kid=k, width=160, height=120) for k in (1, 2)}
    s.store = types.SimpleNamespace(get=kfs.get, frames=lambda: list(kfs.values()))
    return s


def test_realign_after_ace_update_and_keep_map_on_failure():
    s = _depth_session()
    z, rgb, _ = scene()
    disp = network_like(z)
    s.set_depth(1, z.copy(), {"ms": 5.0, "rel_err": 0.05}, disp=disp, version=1)
    assert s.depth_map(1).shape == (60, 80) and s._disp[1].shape == (60, 80)
    assert s.realign_jobs() == []                                   # nothing newer yet
    pix, zz = ace_points(z)
    s._ace_pts[1] = (pix, zz * 1.1, 2)                             # a refit rescaled the map
    (kid, d, p, zj, ver), = s.realign_jobs()
    res = dense_depth(d, None, p, zj, guide_radius=0)
    assert res.align.ok
    s.set_depth(kid, res.depth, {"realign": True, "rel_err": res.align.rel_err}, version=ver)
    assert s.depth_counters["realigned"] == 1 and s.depth_counters["maps"] == 1
    v = s.depth_map(1) > 0
    ratio = np.median(s.depth_map(1)[v].astype(np.float32) / cv2.resize(z, (80, 60))[v])
    assert abs(ratio - 1.1) < 0.05                                  # follows the new ACE scale
    assert s.realign_jobs() == []
    # a failed re-alignment keeps the previous map and is not retried for that version
    before = s.depth_map(1).copy()
    s._ace_pts[1] = (pix[:10], zz[:10], 3)
    (kid, d, p, zj, ver), = s.realign_jobs()
    s.set_depth(kid, dense_depth(d, None, p, zj, guide_radius=0).depth, {"realign": True}, version=ver)
    assert np.array_equal(s.depth_map(1), before) and s.realign_jobs() == []


def test_request_refit_only_when_idle_and_changed():
    import threading

    s = _depth_session()
    s.map_ready = threading.Event()
    s.failed, s._job, s._new_registered, s._new_failed = None, None, 3, 0
    started = []
    s._maybe_refit = lambda force=False: (started.append(force), setattr(s, "_job", "refit"))
    assert not s.request_refit(4)                 # map not ready
    s.map_ready.set()
    assert not s.request_refit(4)                 # only 3 keyframes changed
    s._new_failed = 1
    assert s.request_refit(4) and started == [True] and s._refit_requested == 1
    assert not s.request_refit(4)                 # a refit is already running


def test_align_disparity_render_anchors_guarded_by_ace():
    """Closed loop: render anchors join the fit only when held-out ACE points agree."""
    from hypersplat.stream.depth import align_disparity

    rng = np.random.default_rng(0)
    z = rng.uniform(2.0, 8.0, 400)
    a_true, b_true = 0.5, 0.05
    disp = (1.0 / z - b_true) / a_true
    noisy = disp + rng.normal(0, 0.01, disp.shape)  # ACE-consistent, a little noisy
    # Consistent anchors (from a render that agrees with ACE) are used
    zr = rng.uniform(2.0, 8.0, 2000)
    dr = (1.0 / zr - b_true) / a_true
    al = align_disparity(noisy, z, anchors=(dr, zr))
    assert al.ok and al.anchored
    assert abs(al.a - a_true) < 0.02 and abs(al.b - b_true) < 0.01
    # Anchors from a render whose scale drifted (even 2 %) would pull the fit off ACE: rejected
    al_bad = align_disparity(noisy, z, anchors=(dr, zr * 1.02))
    base = align_disparity(noisy, z)
    assert not al_bad.anchored
    assert al_bad.a == base.a and al_bad.b == base.b
