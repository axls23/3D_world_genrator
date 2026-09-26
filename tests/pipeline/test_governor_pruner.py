"""Governor state (profile-fed, relative normalizers) and percentile-budget pruner. CPU only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

DQN_DIR = Path(__file__).resolve().parents[2] / "scripts" / "post" / "dqn_pruner"
if str(DQN_DIR) not in sys.path:
    sys.path.insert(0, str(DQN_DIR))

import governor  # noqa: E402
from agent import ContextAwareAgent, HeuristicAgent  # noqa: E402
from config import PrunerConfig  # noqa: E402
from state_extractor import StateExtractor  # noqa: E402

from hypersplat.pipeline.params import SceneProfile  # noqa: E402


# ---------------------------------------------------------------------------- pruner
def _cfg(**kw):
    cfg = PrunerConfig()
    cfg.USE_GPU = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _scene(seed=0, n_dense=3000, n_floaters=20, red=True):
    """Dense surface-like cloud + a saturated red patch + far, faint floaters.

    Mimics PLY fields: opacity logits, log-scales, SH DC colours.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, size=(n_dense, 3)).astype(np.float32)
    pos[:, 2] *= 0.05  # a slab
    opac = rng.normal(1.0, 1.0, size=n_dense).astype(np.float32)
    scales = np.log(np.full((n_dense, 3), 0.03, np.float32)) + rng.normal(0, 0.2, (n_dense, 3)).astype(np.float32)
    scales[:, 2] -= 2.0  # flat discs: legitimate, must not count as needles
    colors = rng.normal(0.0, 0.15, size=(n_dense, 3)).astype(np.float32)
    red_mask = np.zeros(n_dense, bool)
    if red:
        red_mask = (np.abs(pos[:, 0]) < 0.35) & (np.abs(pos[:, 1]) < 0.35)
        colors[red_mask] = np.array([3.0, -1.7, -1.7], np.float32) + rng.normal(0, 0.1, (red_mask.sum(), 3))
    # floaters: far off the slab, faint, isolated
    fpos = rng.uniform(-1, 1, size=(n_floaters, 3)).astype(np.float32)
    fpos[:, 2] = rng.uniform(1.5, 3.0, n_floaters)
    fop = np.full(n_floaters, -4.0, np.float32)
    fsc = np.log(np.full((n_floaters, 3), 0.2, np.float32))
    fcol = rng.normal(0.0, 0.8, size=(n_floaters, 3)).astype(np.float32)

    data = {
        'positions': torch.tensor(np.concatenate([pos, fpos])),
        'opacities': torch.tensor(np.concatenate([opac, fop])).unsqueeze(1),
        'scales': torch.tensor(np.concatenate([scales, fsc])),
        'colors': torch.tensor(np.concatenate([colors, fcol])),
    }
    floater_mask = np.concatenate([np.zeros(n_dense, bool), np.ones(n_floaters, bool)])
    red_mask = np.concatenate([red_mask, np.zeros(n_floaters, bool)])
    return data, floater_mask, red_mask


def _run(data, cfg=None):
    cfg = cfg or _cfg()
    states = StateExtractor(cfg).extract_features(data)
    agent = ContextAwareAgent(cfg)
    return states, agent, agent.predict(states, data).cpu().numpy().astype(bool)


def test_log_scales_read_in_linear_units():
    data, _, _ = _scene()
    n = data['positions'].shape[0]
    tiny, needle, giant = 0, 1, 2
    data['scales'][tiny] = torch.tensor([-9.0, -9.0, -9.0])   # |log| huge, actually tiny
    data['scales'][needle] = torch.tensor([-1.0, -5.0, -5.0])  # max/mid = e^4
    data['scales'][giant] = torch.tensor([0.5, 0.4, 0.3])      # ~1.6 units vs ~0.05 spacing
    cfg = _cfg()
    states = StateExtractor(cfg).extract_features(data)
    f, _ = ContextAwareAgent(cfg).feature_scores(states, data)
    assert f['giant'][tiny] == 0.0, "a tiny Gaussian must not be flagged as giant"
    assert f['needle'][tiny] < 0.1
    assert f['needle'][needle] == 1.0, "a genuinely elongated Gaussian is a needle"
    assert f['giant'][giant] == 1.0
    # flat discs (one short axis) are ordinary surface splats, not needles
    assert float(f['needle'][3:n - 20].median()) < 0.2


def test_saturated_colour_consistent_with_neighbours_not_pruned():
    data, _, red = _scene()
    cfg = _cfg()
    states = StateExtractor(cfg).extract_features(data)
    f, _ = ContextAwareAgent(cfg).feature_scores(states, data)
    red_t = torch.tensor(red)
    # colour score of the coherent red patch is as low as everyone else's
    assert float(f['color'][red_t].mean()) <= float(f['color'][~red_t].mean()) + 0.05
    _, _, pruned = _run(data, cfg)
    assert pruned[red].mean() <= max(pruned[~red].mean(), 1e-9)


def test_budget_respected_and_floaters_pruned():
    data, floaters, _ = _scene(n_floaters=40)
    _, _, pruned = _run(data)
    n = len(pruned)
    assert pruned.sum() <= int(0.03 * n)
    assert pruned[floaters].mean() >= 0.8, "far faint floaters should be removed"


def test_budget_configurable():
    data, _, _ = _scene(n_floaters=200)
    _, _, pruned = _run(data, _cfg(PRUNE_BUDGET_FRAC=0.01))
    assert pruned.sum() <= int(0.01 * len(pruned))


def test_opaque_or_dense_gaussians_never_pruned():
    data, _, _ = _scene(n_floaters=60)
    # make half the floaters opaque: they must survive even though they are isolated
    data['opacities'][-30:] = 5.0
    cfg = _cfg(PRUNE_BUDGET_FRAC=0.2)  # generous budget so only the gates decide
    states, _, pruned = _run(data, cfg)
    op = states[:, 0].numpy()
    iso = states[:, 7].numpy()
    assert pruned.any()
    assert not pruned[-30:].any()
    assert (op[pruned] <= np.percentile(op, 50) + 1e-6).all()
    assert (iso[pruned] >= np.percentile(iso, 75) - 1e-6).all()


def test_heuristic_thresholds_are_scale_invariant():
    data, _, _ = _scene()
    cfg = _cfg()
    s1 = StateExtractor(cfg).extract_features(data)
    a = HeuristicAgent(cfg).predict(s1, data).numpy()
    big = dict(data)
    big['positions'] = data['positions'] * 100.0  # same scene in other units
    s2 = StateExtractor(cfg).extract_features(big)
    b = HeuristicAgent(cfg).predict(s2, big).numpy()
    assert (a == b).all()
    assert 0 < a.sum() < 0.2 * len(a)


def test_heuristic_absolute_fallback():
    data, _, _ = _scene()
    cfg = _cfg(HEURISTIC_OPACITY_PCT=None, HEURISTIC_ISOLATION_PCT=None,
               HEURISTIC_GHOST_ISOLATION_PCT=None)
    op_thr, iso_thr, ghost = HeuristicAgent(cfg).thresholds(torch.zeros(3), torch.zeros(3))
    assert (op_thr, iso_thr, ghost) == (-2.0, 0.5, 0.25)


# ---------------------------------------------------------------------------- governor
def _write_val(result_dir, step, psnr, num_gs):
    stats = result_dir / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (stats / f"val_step{step}.json").write_text(json.dumps({"psnr": psnr, "num_GS": num_gs}))


def test_governor_numeric_sort_and_real_confidence(tmp_path):
    out = tmp_path / "run"
    result = out / "results" / "acezero_3dgs"
    _write_val(result, 2999, 20.0, 150000)
    _write_val(result, 10099, 22.0, 290000)  # text sort would pick step 2999 as latest
    ace = out / "acezero_output"
    ace.mkdir(parents=True)
    lines = [f"images/f{i:03d}.jpg 1 0 0 0 0 0 0 500 {c}" for i, c in
             enumerate([6000] * 18 + [1200, 1100])]
    (ace / "poses_final.txt").write_text("\n".join(lines))

    gov = governor.PipelineGovernor(result, profile=None)
    gov.profile = None  # file fallback path
    state = gov.observe_state()
    assert state["gs_psnr"] == 22.0 and state["gs_density"] == 290000
    assert state["psnr_ref"] == 20.0 and state["psnr_prev"] == 20.0
    assert state["density_ref"] == 290000
    assert 0.0 < state["ace_confidence"] < 1.0  # weak tail near the threshold

    # same run via a profile (numeric-sorted train.val, explicit cap decision)
    prof = SceneProfile(data={
        "ace": {"conf": {f"f{i}": 6000.0 for i in range(20)}},
        "frames": {"count": 20},
        "points": {"count": 100000},
        "decisions": {"CAP_MAX": {"value": 300000}, "MIN_REGISTRATION_CONFIDENCE": {"value": 1000}},
    })
    prof.set("train.val", [
        {"step": 2999, "psnr": 20.0, "num_GS": 150000, "result_dir": str(result)},
        {"step": 10099, "psnr": 22.0, "num_GS": 290000, "result_dir": str(result)},
    ])
    state = governor.PipelineGovernor(result, profile=prof).observe_state()
    assert state["gs_psnr"] == 22.0
    assert state["ace_confidence"] == pytest.approx(1.0)
    assert state["density_ref"] == 300000
    vec = governor.director_vector(state)
    assert vec[9].item() == pytest.approx(0.5 * 22.0 / 20.0)
    assert vec[12].item() == pytest.approx(290000 / 300000)


def test_ace_confidence_threshold_relative():
    good = {str(i): 6000.0 for i in range(10)}
    weak = dict(good, **{"a": 1050.0, "b": 1100.0})
    assert governor.ace_confidence_from_conf(good, 1000) == pytest.approx(1.0)
    assert governor.ace_confidence_from_conf(weak, 1000) < 0.5
    # a lower registration threshold makes the same tail relatively safer
    assert governor.ace_confidence_from_conf(weak, 100) > governor.ace_confidence_from_conf(weak, 1000)
    # dropped frames count against it
    assert governor.ace_confidence_from_conf(good, 1000, n_frames=20) == pytest.approx(0.5)
    assert governor.ace_confidence_from_conf({}, 1000) is None


def test_director_vector_relative_vs_legacy():
    legacy = governor.director_vector({"gs_psnr": 24.0, "gs_density": 5000.0})
    assert legacy[9].item() == pytest.approx(24.0 / 40.0)
    assert legacy[12].item() == pytest.approx(5000 / 200000)
    assert legacy[15].item() == 1.0 and legacy[16].item() == 0.0
    assert legacy[8].item() == 1.0

    # Relative: 5000 of a 50000 cap is 10% -> not starving; 1000 is.
    rel = governor.director_vector({"gs_psnr": 24.0, "gs_density": 5000.0, "density_ref": 50000})
    assert rel[15].item() == 0.0 and rel[12].item() == pytest.approx(0.1)
    rel = governor.director_vector({"gs_psnr": 24.0, "gs_density": 1000.0, "density_ref": 50000})
    assert rel[15].item() == 1.0

    # Converged trigger: plateau at best, above the start (no 30 dB absolute bar)
    conv = {"gs_psnr": 24.0, "psnr_prev": 23.95, "psnr_best": 24.0, "psnr_ref": 22.0,
            "gs_density": 1.0}
    assert governor.director_vector(conv)[16].item() == 1.0
    climbing = dict(conv, psnr_prev=23.0)
    assert governor.director_vector(climbing)[16].item() == 0.0

    # progress slot used by train_director
    assert governor.director_vector({}, progress=0.25)[0].item() == pytest.approx(0.25)


@pytest.mark.parametrize("name", ["drone", "whatsapp"])
def test_governor_replay_profiles(name, drone_profile, whatsapp_profile, tmp_path):
    profile = drone_profile if name == "drone" else whatsapp_profile
    result = tmp_path / "out" / "results" / "acezero_3dgs"
    result.mkdir(parents=True)
    state = governor.PipelineGovernor(result, profile=profile).observe_state()
    vec = governor.director_vector(state)
    conf = state["ace_confidence"]
    assert 0.0 < conf <= 1.0
    if name == "drone":
        # stable flyover: every frame registered with a tight confidence spread
        assert conf > 0.95
        assert state["gs_psnr"] == pytest.approx(23.532)
        assert state["density_ref"] == 300000  # the run's own cap, not 200000
        assert vec[12].item() == pytest.approx(1.0)
        assert vec[9].item() == pytest.approx(0.5)  # first measurement
        assert vec[15].item() == 0.0 and vec[16].item() == 0.0
    else:
        # handheld night clip: weaker tail -> clearly lower confidence than the drone
        assert 0.5 < conf < 0.9
        assert state["gs_psnr"] == 0.0  # no training measurement recorded
        assert state["density_ref"] == 1153899  # points.count
        assert vec[9].item() == 0.0


def test_governor_decides_with_pretrained_director(tmp_path, drone_profile):
    result = tmp_path / "out" / "results" / "acezero_3dgs"
    result.mkdir(parents=True)
    gov = governor.PipelineGovernor(result, profile=drone_profile)
    gov.decide_action(gov.observe_state())
    gov.write_overrides()
    assert gov.current_action in ("DETERMINISTIC_FILL", "RANDOM_JITTER", "PURGE_WORST_VIEW",
                                  "WAIT_FINE_TUNE")
    assert (result / "cfg_governor.yml").exists()


def test_governor_sees_stats_written_after_construction(tmp_path):
    result = tmp_path / "out" / "results" / "acezero_3dgs"
    gov = governor.PipelineGovernor(result, profile=None)
    gov.profile = None
    assert gov.observe_state()["gs_psnr"] == 0.0
    _write_val(result, 999, 18.5, 1000)
    assert gov.observe_state()["gs_psnr"] == 18.5
