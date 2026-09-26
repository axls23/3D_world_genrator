"""Closed-loop Difix strategies (CPU only, no Difix / gsplat imports)."""

import math

import pytest

from hypersplat.pipeline.params.strategies import apply_stage
from hypersplat.pipeline.params.strategies import difix as S


def _round(loop, angle, mean, n=20, kept=20, low=None):
    return {"loop": loop, "max_angle": angle, "mean_change_psnr": mean,
            "min_change_psnr": low if low is not None else mean - 1.0, "n_views": n, "n_kept": kept}


def _val(profile, *psnrs):
    profile.set("train.val", [{"step": 6999, "psnr": p, "result_dir": f"r{k}"}
                              for k, p in enumerate(psnrs)])


# --- DIFIX_MAX_ANGLE ---------------------------------------------------------

def test_first_angle_from_camera_spread(drone_profile, make_config):
    drone_profile.set("difix.camera_spread_deg", 40.0)
    assert S.derive_max_angle(drone_profile, make_config()).value == 6.0
    drone_profile.set("difix.camera_spread_deg", 2.4)   # measured drone spread -> 3 deg floor
    assert S.derive_max_angle(drone_profile, make_config()).value == 3.0
    drone_profile.set("difix.camera_spread_deg", 7.0)
    d = S.derive_max_angle(drone_profile, make_config())
    assert d.value == 3.5 and d.strategy == "camera-spread"


def test_first_angle_without_geometry_is_faithful_start(drone_profile, make_config):
    d = S.derive_max_angle(drone_profile, make_config())
    assert d.value == S.START_ANGLE and d.value < 20.0


def test_angle_widens_when_difix_is_faithful(drone_profile, make_config):
    drone_profile.set("difix.rounds", [_round(0, 6.0, 22.0)])
    d = S.derive_max_angle(drone_profile, make_config())
    assert d.strategy == "widen" and d.value > 6.0


def test_angle_holds_in_middle_band(drone_profile, make_config):
    # Validated drone data: 19.7 dB at 12 deg -> still faithful but don't push further
    drone_profile.set("difix.rounds", [_round(0, 12.0, 19.7)])
    assert S.derive_max_angle(drone_profile, make_config()).value == 12.0


def test_angle_shrinks_and_weight_drops_when_hallucinating(drone_profile, make_config):
    cfg = make_config()
    drone_profile.set("difix.rounds", [_round(0, 9.0, 22.0)])
    w_good = S.derive_pseudo_weight(drone_profile, cfg).value
    drone_profile.set("difix.rounds", [_round(0, 9.0, 22.0), _round(1, 13.5, 14.0)])
    d = S.derive_max_angle(drone_profile, cfg)
    assert d.strategy == "shrink" and d.value < 13.5
    w_bad = S.derive_pseudo_weight(drone_profile, cfg).value
    assert w_bad < w_good and w_bad < 0.5


def test_angle_never_exceeds_cap(drone_profile, make_config):
    drone_profile.set("difix.rounds", [_round(0, 18.0, 23.0)])
    assert S.derive_max_angle(drone_profile, make_config()).value == S.ANGLE_CAP


def test_pseudo_weight_before_any_round(drone_profile, make_config):
    assert S.derive_pseudo_weight(drone_profile, make_config()).value == 0.5


# --- DIFIX_NUM_VIEWS -----------------------------------------------------------

def test_num_views_scale_with_frames(drone_profile, whatsapp_profile, make_config):
    assert S.derive_num_views(drone_profile, make_config()).value == 20      # 40 frames
    assert S.derive_num_views(whatsapp_profile, make_config()).value == 48   # 139 -> capped
    drone_profile.set("frames.count", 6)
    assert S.derive_num_views(drone_profile, make_config()).value == 8       # floor
    drone_profile.set("frames.count", None)
    assert S.derive_num_views(drone_profile, make_config()) is None


# --- REFINE_LOOPS / early exit -------------------------------------------------

def test_loop_stops_on_val_plateau(drone_profile, make_config):
    # Observed: Difix round 1 took held-out PSNR 23.53 -> 23.41
    base, r1 = 23.53, drone_profile.get("_observed.difix_r1.val_psnr")
    _val(drone_profile, base, r1)
    drone_profile.set("difix.rounds", [_round(0, 6.0, 20.6)])
    drone_profile.set("difix.loop_index", 1)
    d = S.derive_refine_loops(drone_profile, make_config())
    assert d.strategy == "val-plateau" and d.value == 1


def test_loop_continues_while_improving(drone_profile, make_config):
    _val(drone_profile, 23.0, 23.5)
    drone_profile.set("difix.rounds", [_round(0, 6.0, 21.0)])
    drone_profile.set("difix.loop_index", 1)
    assert S.derive_refine_loops(drone_profile, make_config()).value == S.MAX_LOOPS


def test_first_loop_always_runs(drone_profile, make_config):
    drone_profile.set("difix.loop_index", 0)
    assert S.derive_refine_loops(drone_profile, make_config()).value == S.MAX_LOOPS


def test_loop_stops_when_hallucinating(drone_profile, make_config):
    _val(drone_profile, 23.0, 23.5)
    drone_profile.set("difix.rounds", [_round(0, 6.0, 21.0, n=20, kept=6)])
    drone_profile.set("difix.loop_index", 1)
    d = S.derive_refine_loops(drone_profile, make_config())
    assert d.strategy == "difix-hallucinating" and d.value == 1


def test_final_val_psnrs_uses_last_step_per_result_dir(drone_profile):
    drone_profile.set("train.val", [
        {"step": 2999, "psnr": 20.0, "result_dir": "a"}, {"step": 6999, "psnr": 23.0, "result_dir": "a"},
        {"step": 6999, "psnr": 23.2, "result_dir": "b"}, {"step": 2999, "psnr": 21.0, "result_dir": "b"}])
    assert S.final_val_psnrs(drone_profile) == [("a", 23.0), ("b", 23.2)]


def test_apply_loop_stage_respects_user_values(drone_profile, make_config):
    drone_profile.set("difix.rounds", [_round(0, 6.0, 22.0)])
    cfg = make_config(USER_SET={"DIFIX_MAX_ANGLE"}, DIFIX_MAX_ANGLE=10.0)
    apply_stage("loop", drone_profile, cfg)
    assert cfg.DIFIX_MAX_ANGLE == 10.0
    assert cfg.DIFIX_NUM_VIEWS == 20
    assert drone_profile.get("decisions.DIFIX_PSEUDO_WEIGHT")["strategy"] == "change-trust"


# --- per-view weighting (used by augment.py) -----------------------------------

def test_threshold_is_relative_with_absolute_floor():
    assert S.hallucination_threshold([22, 21, 20.5, 21.5, 12]) == pytest.approx(18.0)
    assert S.hallucination_threshold([14.0, 14.5, 14.2]) == S.HALLUCINATION_FLOOR
    assert S.hallucination_threshold([None, float("inf")]) == S.HALLUCINATION_FLOOR


def test_pseudo_view_weights_drop_and_scale():
    changes = [22.0, 21.0, 20.5, 21.5, 12.0, 19.0]
    weights, thr = S.pseudo_view_weights([0.5] * 6, changes)
    assert thr == pytest.approx(17.75)  # median 20.75 - 3 dB margin
    assert weights[4] == 0.0                       # hallucinated view dropped
    assert weights[0] == pytest.approx(0.5)        # small change -> full trust
    assert 0.0 < weights[5] < weights[2] < weights[0]  # more change -> less trust
    # Unmeasured change (no Difix) keeps the base weight
    assert S.pseudo_view_weights([0.3], [None])[0] == [0.3]


def test_round_stats_ignores_unmeasured():
    st = S.round_stats(1, 6.0, [3.0, 6.1], [20.0, float("inf")], 17.0, 2)
    assert st["mean_change_psnr"] == 20.0 and st["n_views"] == 2 and st["loop"] == 1
    assert math.isclose(st["max_view_angle"], 6.1)


def test_confidence_ramp_from_distribution(drone_profile, whatsapp_profile):
    # Drone: tight distribution -> nearly flat weights, floor high
    dmin, dfloor = S.confidence_ramp(drone_profile.get("ace.conf").values())
    assert dmin < 0.6 * 6350 and dfloor == 0.5
    # WhatsApp: long low tail -> threshold within the data, lower floor
    conf = list(whatsapp_profile.get("ace.conf").values())
    wmin, wfloor = S.confidence_ramp(conf)
    assert min(conf) < wmin < S.percentile(conf, 50)
    assert 0.1 <= wfloor <= 0.5
    # User values win; no data -> old constants
    assert S.confidence_ramp(conf, 1000.0, 0.2) == (1000.0, 0.2)
    assert S.confidence_ramp([]) == (1000.0, 0.2)
