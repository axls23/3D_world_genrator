"""Device pose table: apply_deltas kernel vs its numpy reference and the session's transform."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from hypersplat.stream.cuda import (ANCHORED, APPLIED, NEW, OUTLIER, PROVISIONAL, STABLE, STALE,
                                    UNREGISTERED, DevicePoseTable, ReferenceState,
                                    apply_deltas_reference)
from hypersplat.stream.ring import DELTA_DTYPE, KIND_RELOC

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from datasets.normalize import transform_cameras  # noqa: E402


def _similarity(seed=0):
    rng = np.random.default_rng(seed)
    M = np.eye(4)
    M[:3, :3] = 0.37 * Rotation.from_rotvec(rng.standard_normal(3)).as_matrix()
    M[:3, 3] = rng.standard_normal(3)
    return M


def _w2c(center, rotvec):
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    c2w[:3, 3] = center
    return np.linalg.inv(c2w)


def _rec(kid, w2c, version=1, conf=1000.0, anchored=0):
    r = np.zeros(1, DELTA_DTYPE)
    r["kid"], r["map_version"], r["kind"], r["conf"] = kid, version, KIND_RELOC, conf
    r["anchored"] = anchored
    r["w2c"] = np.asarray(w2c, np.float64).reshape(16) if w2c is not None else np.nan
    return r


def _run(table, recs, **kw):
    table.apply(recs, **kw)
    return table.poll(wait=True)


def test_alpha_one_matches_session_transform():
    rng = np.random.default_rng(1)
    M = _similarity()
    t = DevicePoseTable(64)
    t.set_transform(M)
    w2cs = [_w2c(rng.standard_normal(3) * 5, rng.standard_normal(3)) for _ in range(20)]
    recs = np.concatenate([_rec(k, w) for k, w in enumerate(w2cs)])
    res = _run(t, recs, conf_threshold=500, extent=10.0)
    assert (res["result"] == APPLIED).all() and (res["tier"] == PROVISIONAL).all()
    expect = transform_cameras(M, np.stack([np.linalg.inv(w) for w in w2cs]))
    got = t.c2w_trainer.cpu().numpy().reshape(-1, 4, 4)[[t.slot_of(k) for k in range(20)]]
    np.testing.assert_allclose(got, expect, atol=2e-5)


def test_kernel_matches_reference_sequence():
    rng = np.random.default_rng(2)
    M = _similarity(3)
    kw = dict(conf_threshold=500.0, extent=4.0, alpha=0.3, promote_thr=0.012, outlier_thr=0.05,
              stable_after=2)
    t = DevicePoseTable(32)
    t.set_transform(M)
    ref = ReferenceState(32)
    base = {k: (rng.standard_normal(3) * 2, rng.standard_normal(3) * 0.5) for k in range(8)}
    batches = []
    for step in range(40):
        recs = []
        for _ in range(rng.integers(1, 4)):        # repeats of one kid in a batch happen
            k = int(rng.integers(0, 8))
            c, rv = base[k]
            jitter = rng.choice([0.002, 0.01, 0.2], p=[0.6, 0.3, 0.1]) * 4.0
            version = step // 10 + (0 if rng.random() > 0.05 else -1)   # a few stale rows
            conf = 1000.0 if rng.random() > 0.08 else 10.0               # a few failed relocs
            recs.append(_rec(k, _w2c(c + rng.standard_normal(3) * jitter / 1.7,
                                     rv + rng.standard_normal(3) * 0.003), version, conf))
        if step == 35:                              # the sidecar anchors kid 0
            c, rv = base[0]
            recs.append(_rec(0, _w2c(c, rv), step // 10, 1000.0, anchored=1))
        batches.append(np.concatenate(recs))
    got, want = [], []
    for recs in batches:
        res = _run(t, recs, **kw)
        slots = np.array([t.slot_of(int(k)) for k in recs["kid"]])
        want.extend(apply_deltas_reference(ref, recs, slots, M, **kw))
        got.extend(zip(res["move"], res["result"], res["tier"]))
    g, w = np.array(got, dtype=float), np.array(want, dtype=float)
    np.testing.assert_array_equal(g[:, 1:], w[:, 1:])                    # results, tiers
    np.testing.assert_allclose(g[:, 0], w[:, 0], rtol=1e-5, equal_nan=True)
    np.testing.assert_allclose(t.c2w_trainer.cpu().numpy().reshape(-1, 4, 4), ref.c2w_trainer, atol=2e-5)
    assert {APPLIED, STALE, UNREGISTERED, OUTLIER} <= set(g[:, 1].astype(int))
    assert STABLE in set(g[:, 2].astype(int)) and ANCHORED in set(g[:, 2].astype(int))


def test_filter_averages_jitter_and_passes_corrections():
    rng = np.random.default_rng(4)
    t = DevicePoseTable(4)
    true_c = np.array([1.0, 2.0, 3.0])
    kw = dict(conf_threshold=500.0, extent=1.0, alpha=0.25, promote_thr=0.02, outlier_thr=0.2)
    errs_raw, errs_f = [], []
    for i in range(200):
        c = true_c + rng.standard_normal(3) * 0.01
        _run(t, _rec(0, _w2c(c, [0.1, 0.2, 0.3])), **kw)
        if i > 20:
            f = t.c2w_f.cpu().numpy()[t.slot_of(0)].reshape(4, 4)[:3, 3]
            errs_raw.append(np.linalg.norm(c - true_c))
            errs_f.append(np.linalg.norm(f - true_c))
    assert np.mean(errs_f) < 0.5 * np.mean(errs_raw)
    # a 0.5 jump is a correction: taken whole, at once
    res = _run(t, _rec(0, _w2c(true_c + [0.5, 0, 0], [0.1, 0.2, 0.3])), **kw)
    assert res["result"][0] == OUTLIER
    f = t.c2w_f.cpu().numpy()[t.slot_of(0)].reshape(4, 4)[:3, 3]
    np.testing.assert_allclose(f, true_c + [0.5, 0, 0], atol=1e-9)


def test_unregistered_and_slot_reuse():
    t = DevicePoseTable(2)
    kw = dict(conf_threshold=500.0, extent=1.0)
    assert _run(t, _rec(5, _w2c([0, 0, 0], [0, 0, 0])), **kw)["tier"][0] == PROVISIONAL
    res = _run(t, _rec(5, None, conf=0.0), **kw)
    assert res["result"][0] == UNREGISTERED and res["tier"][0] == NEW
    assert int(t.has_pose[t.slot_of(5)]) == 0
    _run(t, _rec(6, _w2c([1, 0, 0], [0, 0, 0])), **kw)
    t.release(5)
    s7 = t.slot(7)
    assert s7 >= 0 and int(t.ver[s7]) == 0 and int(t.tier[s7]) == NEW
    assert t.slot(8) == -1            # full: 6 and 7
