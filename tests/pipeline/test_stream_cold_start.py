"""Cold start (no bootstrap): seed depth from relative disparity, one-camera normalization."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hypersplat.stream.depth import seed_depth
from hypersplat.stream.session import cold_normalization_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from datasets.normalize import transform_cameras, transform_points  # noqa: E402


def test_seed_depth_inverts_disparity_and_sets_median():
    h, w = 60, 80
    z_true = np.linspace(1.0, 4.0, w)[None].repeat(h, 0)    # a slanted plane
    disp = 3.0 / z_true                                       # shift-free, unknown scale
    z, st = seed_depth(disp.astype(np.float32), None, median_depth=2.0, guide_radius=0)
    valid = z > 0
    assert st["valid"] > 0.9
    assert abs(np.median(z[valid]) - 2.0) < 0.05
    # shape is kept: depth ratios match the true ones
    ratio = z[valid] / z_true[valid]
    assert ratio.std() / ratio.mean() < 1e-3


def test_seed_depth_drops_sky_and_edges():
    h, w = 60, 80
    disp = np.full((h, w), 1.0, np.float32)
    disp[:20] = 1e-4                 # sky: ~no disparity
    disp[40:, 40:] = 4.0             # near box -> discontinuity at its border
    z, _ = seed_depth(disp, None, median_depth=2.0, guide_radius=0)
    assert (z[:18] == 0).all()                      # far beyond 8x median
    assert (z[38:42, 45:] == 0).all()               # flying pixels on the box edge
    assert (z[25:35, 5:35] > 0).all()               # plain surface kept


def test_seed_depth_empty():
    z, st = seed_depth(np.zeros((10, 10), np.float32), None, guide_radius=0)
    assert st["valid"] == 0.0 and not z.any()


def test_cold_normalization_one_camera():
    rng = np.random.default_rng(0)
    # seed camera looking along +z (ACE identity), tilted a little
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_euler("x", 10, degrees=True).as_matrix()
    pts_cam = np.c_[rng.uniform(-1, 1, (500, 2)), rng.uniform(1.5, 2.5, 500)]
    pts = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
    T = cold_normalization_transform(c2w, pts)
    cam = transform_cameras(T, c2w[None])[0]
    # scene centre (median of the seed points) at the origin, camera at unit distance from it
    assert np.linalg.norm(transform_points(T, np.median(pts, 0)[None])) < 1e-9
    assert abs(np.linalg.norm(cam[:3, 3]) - 1.0) < 1e-6
    # camera up (-y in OpenCV) maps to world +z
    up = cam[:3, :3] @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(up, [0, 0, 1], atol=1e-6)
    # a similarity: one uniform scale
    s = np.linalg.svd(T[:3, :3], compute_uv=False)
    assert np.allclose(s, s[0])


def test_cold_normalization_upside_down_camera():
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    pts = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.1]])
    T = cold_normalization_transform(c2w, pts)
    cam = transform_cameras(T, c2w[None])[0]
    up = cam[:3, :3] @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(up, [0, 0, 1], atol=1e-6)
