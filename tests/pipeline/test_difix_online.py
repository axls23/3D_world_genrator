"""Reproducible online Difix: per-view seeds and pooled round control (CPU only)."""

import pytest

from hypersplat.pipeline.params import SceneProfile
from hypersplat.pipeline.params.strategies.difix import (derive_max_angle, derive_pseudo_weight, judged,
                                                         round_hallucinating, round_stats)
from hypersplat.stream.difix_online import OnlineDifix


def _difix(**kw):
    return OnlineDifix(profile=SceneProfile(), **kw)


def _round(d, changes, kept, angle=6.0, loop=0):
    stats = round_stats(loop, angle, [angle] * len(changes), changes, 15.0, kept)
    return d._pooled(stats, changes, kept, len(changes), angle)


def test_view_seeds_are_fixed_and_distinct():
    d = _difix(seed=0)
    seeds = {d.view_seed(r, k) for r in range(20) for k in range(16)}
    assert len(seeds) == 320
    assert _difix(seed=0).view_seed(3, 1) == d.view_seed(3, 1)
    assert _difix(seed=1).view_seed(3, 1) != d.view_seed(3, 1)
    assert _difix(seed=None).view_seed(3, 1) is None


def test_one_bad_two_view_round_no_longer_stops_difix():
    # Per round (legacy): 0 of 2 views kept -> "hallucinating", Difix off for the rest of the run
    legacy = round_stats(0, 6.0, [6.0, 6.0], [14.0, 14.5], 15.0, 0)
    assert round_hallucinating(legacy)
    d = _difix(pool_views=8)
    s = _round(d, [14.0, 14.5], 0)
    assert s["judge"] == "pooling" and judged(s) == (None, 0, None)
    assert s["n_views"] == 2 and s["n_kept"] == 0  # the round's own stats are unchanged
    assert round_hallucinating(s) is None
    for _ in range(3):  # three good rounds fill the pool: the bad pair is outvoted
        s = _round(d, [22.0, 23.0], 2)
    assert judged(s)[1:] == (8, 6) and round_hallucinating(s) is None
    assert s["mean_change_psnr"] == pytest.approx(22.5)


def test_pool_slides_and_restarts_on_angle_change():
    d = _difix(pool_views=4)
    for _ in range(5):
        s = _round(d, [20.0, 20.0], 2)
    assert s["pooled_rounds"] == 2 and judged(s)[1] == 4
    s = _round(d, [20.0, 20.0], 2, angle=9.0)
    assert s["judge"] == "pooling"  # change PSNR depends on the angle: start over


def test_weight_and_angle_hold_while_pooling():
    prof = SceneProfile()
    prof.append("difix.rounds", {"max_angle": 6.0, "mean_change_psnr": 22.0, "n_views": 8, "n_kept": 8}, "t")
    w = derive_pseudo_weight(prof, None).value
    prof.append("difix.rounds", {"max_angle": 9.0, "mean_change_psnr": 12.0, "n_views": 2, "n_kept": 0,
                                 "judge": "pooling"}, "t")
    assert derive_pseudo_weight(prof, None).value == w
    assert derive_max_angle(prof, None).value == 9.0
