"""CPU tests for per-keyframe pose / exposure corrections (hypersplat/stream/training.py)."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from hypersplat.stream.training import SlotMap, StreamViewAdjust  # noqa: E402
from utils import CameraOptModule  # noqa: E402


def pose(t=(0, 0, 0), yaw_deg=0.0):
    c = np.eye(4)
    a = math.radians(yaw_deg)
    c[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    c[:3, 3] = t
    return c


def test_slotmap_assign_reuse_and_lru():
    m = SlotMap(3)
    s1, fresh, moved = m.assign(10, pose())
    assert fresh and not moved
    assert m.assign(10, pose()) == (s1, False, False)
    s2, _, _ = m.assign(11, pose())
    s3, _, _ = m.assign(12, pose())
    assert len({s1, s2, s3}) == 3
    m.assign(10, pose())                      # 11 is now least recently used
    s4, fresh, _ = m.assign(13, pose())
    assert fresh and s4 == s2 and 11 not in m.slot_of
    assert m.release_missing({10, 13}) == 1   # 12 evicted from the store
    s5, fresh, _ = m.assign(14, pose())
    assert fresh and s5 == s3


def test_slotmap_detects_refit_moves():
    m = SlotMap(4, move_tol=0.01, angle_tol_deg=0.5)
    m.assign(1, pose())
    assert not m.assign(1, pose((0.005, 0, 0)))[2]      # below tolerance
    assert m.assign(1, pose((0.05, 0, 0)))[2]            # moved
    assert not m.assign(1, pose((0.05, 0, 0)))[2]        # reference updated
    assert m.assign(1, pose((0.05, 0, 0), yaw_deg=1.0))[2]


def make(capacity=8):
    pm = CameraOptModule(capacity)
    pm.zero_init()
    va = StreamViewAdjust(capacity, pm, exposure=True)
    va.pose_optimizers = [torch.optim.Adam(pm.parameters(), lr=1e-2)]
    va.exposure_optimizers = [torch.optim.Adam(va.exposure.parameters(), lr=1e-2)]
    return va


def test_identity_at_init_and_no_correction_for_pseudo():
    va = make()
    c2w = torch.tensor(np.stack([pose((1, 0, 0)), pose((0, 1, 0))]), dtype=torch.float32)
    slots = va.slot_ids(torch.tensor([5, -1]), c2w, torch.tensor([False, True]))
    assert slots.tolist()[1] == -1 and slots[0] >= 0
    assert torch.allclose(va.adjust_pose(c2w, slots), c2w, atol=1e-6)
    colors = torch.rand(2, 4, 5, 3)
    assert torch.allclose(va.adjust_colors(colors, slots), colors)
    # a learned correction applies to the real view only (exposure relative to the live mean)
    other = va.slot_ids(torch.tensor([6]), c2w[:1], None)
    with torch.no_grad():
        va.pose.embeds.weight[slots[0], :3] = torch.tensor([0.1, 0.0, 0.0])
        va.exposure.weight[slots[0]] = torch.tensor([math.log(2.0)] * 3 + [0.0] * 3)
    out = va.adjust_pose(c2w, slots)
    assert not torch.allclose(out[0], c2w[0]) and torch.allclose(out[1], c2w[1])
    cols = va.adjust_colors(colors, slots)
    assert torch.allclose(cols[0], math.sqrt(2) * colors[0], atol=1e-5) and torch.allclose(cols[1], colors[1])
    both = torch.cat([slots[:1], other])
    c2 = va.adjust_colors(colors, both)
    assert torch.allclose(c2[0] * c2[1], colors[0] * colors[1], atol=1e-5)  # corrections cancel on average


def test_exposure_learns_relative_gain_and_reset_on_reassign():
    va = make(capacity=3)
    c2w = torch.tensor(np.stack([pose(), pose()]), dtype=torch.float32)
    # two keyframes of the same surface: one exposed 2x brighter than the other
    target = torch.stack([torch.full((4, 4, 3), 0.8), torch.full((4, 4, 3), 0.4)])
    render = torch.full((2, 4, 4, 3), 0.4 * math.sqrt(2))   # the model's neutral colour
    for _ in range(300):
        slots = va.slot_ids(torch.tensor([7, 9]), c2w, None)
        loss = (va.adjust_colors(render, slots) - target).abs().mean()
        loss.backward()
        va.exposure_optimizers[0].step()
        va.exposure_optimizers[0].zero_grad()
    assert torch.allclose(va.adjust_colors(render, slots), target, atol=0.02)
    slot = int(slots[0])
    va.slots.release_missing({9})                    # keyframe 7 evicted
    new = va.slot_ids(torch.tensor([8]), c2w[:1], None)
    assert int(new[0]) == slot                        # slot reused ...
    assert torch.count_nonzero(va.exposure.weight[slot]) == 0   # ... and zeroed
    st = va.exposure_optimizers[0].state[va.exposure.weight]
    assert torch.count_nonzero(st["exp_avg"][slot]) == 0


def test_exposure_reg_penalises_mean_only():
    va = make()
    c2w = torch.tensor(np.stack([pose(), pose()]), dtype=torch.float32)
    slots = va.slot_ids(torch.tensor([1, 2]), c2w, None)
    with torch.no_grad():
        va.exposure.weight[slots[0], 0] = 0.3
        va.exposure.weight[slots[1], 0] = -0.3
    assert float(va.exposure_reg()) < 1e-8           # opposite exposures: mean is neutral
    with torch.no_grad():
        va.exposure.weight[slots[1], 0] = 0.3
    assert float(va.exposure_reg()) > 0.05
