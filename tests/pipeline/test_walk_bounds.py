"""Walk-mode bounds for the live viewer (hypersplat/stream/walk_bounds.py)."""
import numpy as np
import pytest

from hypersplat.stream.walk_bounds import WalkBounds, WalkConfig, compute_walk_bounds


def _poses(centers, fwd=(1.0, 0.0, 0.0)):
    """Level OpenCV cameras (+Z forward) at the given centers, all looking along `fwd`."""
    f = np.asarray(fwd, float) / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(f, up)  # camera right
    y = np.cross(f, x)   # camera down
    out = np.tile(np.eye(4), (len(centers), 1, 1))
    out[:, :3, 0], out[:, :3, 1], out[:, :3, 2] = x, y, f
    out[:, :3, 3] = centers
    return out


def _line(n=20, step=0.1, z=1.5, slope=0.0):
    t = np.arange(n) * step
    return np.stack([t, np.zeros(n), z + slope * t], -1)


def test_straight_path_is_a_corridor():
    c = _line()
    b = compute_walk_bounds(_poses(c), scene_scale=1.0)
    r = WalkConfig().radius_mult * b.spacing
    assert b.spacing == pytest.approx(0.1)
    # Whole path is walkable, including midpoints between keyframes
    mids = (c[1:] + c[:-1]) / 2
    assert b.contains(c).all() and b.contains(mids).all()
    # Inside the radius beside the path yes, well outside no
    assert b.contains(np.array([[1.0, 0.8 * r]])).all()
    assert not b.contains(np.array([[1.0, 1.5 * r], [1.0, -1.5 * r], [-1.5 * r, 0.0]])).any()


def test_heights_follow_capture_height_on_a_slope():
    c = _line(slope=0.2)
    b = compute_walk_bounds(_poses(c), scene_scale=1.0)
    ij = np.round((c[:, :2] - np.asarray(b.origin)) / b.cell).astype(int)
    h = b.heights[ij[:, 1], ij[:, 0]]
    # Interior cameras: weighted mean of a linear ramp is ~the ramp itself
    np.testing.assert_allclose(h[3:-3], c[3:-3, 2], atol=0.02)


def test_relocalization_jump_is_not_bridged():
    a = _line(n=10)
    b_ = _line(n=10) + np.array([5.0, 0.0, 0.0])
    b = compute_walk_bounds(_poses(np.concatenate([a, b_])), scene_scale=5.0)
    assert not b.contains(np.array([[3.0, 0.0]])).any()
    assert b.contains(np.array([[0.5, 0.0], [5.5, 0.0]])).all()


def test_l_shaped_path_turns_the_corner():
    leg1 = _line(n=10)
    leg2 = np.stack([np.full(10, 0.9), 0.1 * np.arange(1, 11), np.full(10, 1.5)], -1)
    b = compute_walk_bounds(_poses(np.concatenate([leg1, leg2])), scene_scale=1.0)
    assert b.contains(np.array([[0.9, 0.0], [0.9, 0.5], [0.45, 0.0]])).all()
    assert not b.contains(np.array([[0.0, 1.0]])).any()  # far side of the L stays closed


@pytest.mark.parametrize("n", [1, 2])
def test_few_cameras(n):
    b = compute_walk_bounds(_poses(_line(n=n)), scene_scale=2.0)
    assert b.spacing == pytest.approx(0.1)  # 0.05 * scene_scale fallback
    assert b.contains(_line(n=n)).all()
    assert b.mask.sum() > 0


def test_empty_and_start_pose():
    assert compute_walk_bounds(np.zeros((0, 4, 4)), 1.0) is None
    c = _line()
    b = compute_walk_bounds(_poses(c, fwd=(0.0, 1.0, -0.3)), scene_scale=1.0)
    assert b.start_position == pytest.approx(tuple(c[0]))
    look = np.asarray(b.start_look_at) - c[0]
    assert look[2] == pytest.approx(0.0)  # levelled to the horizon
    assert look[1] > 0


def test_grid_is_capped():
    c = _line(n=400, step=0.05)
    c[:, 1] = np.sin(np.arange(400) * 0.05) * 5
    b = compute_walk_bounds(_poses(c), scene_scale=10.0)
    assert max(b.mask.shape) <= 256


def test_npz_round_trip(tmp_path):
    b = compute_walk_bounds(_poses(_line()), scene_scale=1.0)
    b.save_npz(tmp_path / "w.npz")
    b2 = WalkBounds.load_npz(tmp_path / "w.npz")
    assert b2.origin == pytest.approx(b.origin) and b2.num_views == b.num_views
    np.testing.assert_array_equal(b2.mask, b.mask)
    np.testing.assert_allclose(b2.heights, b.heights)
    assert b2.start_look_at == pytest.approx(b.start_look_at)
