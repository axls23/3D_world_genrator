"""Streaming pipeline (hypersplat/stream): CPU-only unit tests.

The ACE sidecar and Difix are replaced by fakes here; the GPU path is exercised by running
examples/simple_trainer.py with --stream_source (see docs in hypersplat/stream/session.py).
"""

import os
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pytest
import torch
from scipy.spatial.transform import Rotation

from hypersplat.stream import native
from hypersplat.stream.keyframes import KeyframeGate
from hypersplat.stream.schema import (COMMAND_SCHEMA, EVENT_SCHEMA, IpcWriter, command_batch,
                                      event_batch, ipc_batches, parse_meta, select_rows)
from hypersplat.stream.source import FrameSource, QueueFrameSource, encode_jpeg, parse_source
from hypersplat.stream.store import KeyframeStore, PseudoView
from hypersplat.stream.training import WithPseudo, inject_gaussians


def _cam(yaw_deg, x):
    c2w = np.eye(4)
    c2w[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    c2w[:3, 3] = [x, 0.0, 0.0]
    return c2w


def _texture(seed, h=120, w=160):
    rng = np.random.default_rng(seed)
    img = cv2.resize((rng.random((h // 8, w // 8, 3)) * 255).astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_NEAREST)
    return img


# ============================================================================ native
@pytest.mark.parametrize("n", [2, 7, 60])
def test_pose_nn_distance_native_matches_numpy_and_brute_force(n, monkeypatch):
    rng = np.random.default_rng(n)
    c2w = np.stack([_cam(rng.uniform(-40, 40), rng.uniform(-2, 2)) for _ in range(n)])
    query = (rng.random(n) > 0.3).astype(np.uint8)
    query[0] = 1
    fast = native.pose_nn_distance(c2w, query, 0.5, 15.0)

    brute = np.full(n, np.nan)
    for i in range(n):
        if query[i]:
            ds = []
            for j in range(n):
                if j != i:
                    d = np.linalg.norm(c2w[i, :3, 3] - c2w[j, :3, 3]) / 0.5
                    cos = (np.trace(c2w[i, :3, :3].T @ c2w[j, :3, :3]) - 1) / 2
                    ds.append(d + np.degrees(np.arccos(np.clip(cos, -1, 1))) / 15.0)
            brute[i] = min(ds)
    np.testing.assert_allclose(fast, brute, rtol=1e-9, equal_nan=True)

    monkeypatch.setattr(native, "_lib", None)
    monkeypatch.setattr(native, "_tried", True)  # force the numpy fallback
    np.testing.assert_allclose(native.pose_nn_distance(c2w, query, 0.5, 15.0), brute, rtol=1e-9,
                               equal_nan=True)


def test_native_library_builds():
    assert native.lib() is not None, "C toolchain present in this env: the native path must load"


def test_weighted_pick_distribution_and_edges():
    w = np.array([0.0, 1.0, 3.0, 0.0])
    assert native.weighted_pick(np.zeros(3), 0.5) == -1
    assert native.weighted_pick(w, 0.0) == 1
    assert native.weighted_pick(w, 0.2499) == 1
    assert native.weighted_pick(w, 0.2501) == 2
    assert native.weighted_pick(w, 0.999999) == 2
    rng = np.random.default_rng(0)
    counts = np.bincount([native.weighted_pick(w, rng.random()) for _ in range(4000)], minlength=4)
    assert counts[0] == counts[3] == 0
    assert abs(counts[2] / counts[1] - 3.0) < 0.4


# ============================================================================ store
def _filled_store(n=10, **kw):
    kw.setdefault("conf_threshold", 100)
    st = KeyframeStore(**kw)
    for i in range(n):
        kf = st.add(b"x" * 1000, 160, 120, float(i))
        st.update_pose(kf.kid, np.linalg.inv(_cam(i * 10.0, i * 1.0)), 1000.0, 1)
    return st


def test_store_split_and_registration():
    st = _filled_store(10, test_every=4)
    assert [kf.kid for kf in st.frames() if kf.split == "val"] == [4, 8]  # kid 0 is never held out
    kf = st.add(b"y", 160, 120, 11.0)
    assert kf in st.pending()
    st.update_pose(kf.kid, np.eye(4), 5.0, 1)  # below threshold
    assert kf in st.unregistered() and kf not in st.registered()


def test_store_evicts_most_redundant_registered_frame():
    st = _filled_store(10, max_keyframes=10, protect_recent=2, test_every=0)
    dup = st.add(b"x" * 1000, 160, 120, 20.0)
    st.update_pose(dup.kid, np.linalg.inv(_cam(30.0, 3.0)), 900.0, 1)  # same pose as kid 3
    for _ in range(3):  # keep the duplicate out of the protected newest frames
        k = st.add(b"x" * 1000, 160, 120, 30.0)
        st.update_pose(k.kid, np.linalg.inv(_cam(200.0 + k.kid * 20, 50.0 + k.kid * 5)), 1000.0, 1)
    evicted = st.evict_if_needed()
    assert len(st) == 10
    # the two copies of pose 3 are each other's nearest neighbour; the lower-confidence one goes
    assert dup.kid in [k.kid for k in evicted]


def test_store_eviction_order_and_pinning():
    st = KeyframeStore(max_keyframes=3, protect_recent=0, pending_ttl=0.0, conf_threshold=100)
    a = st.add(b"a", 1, 1, 0)
    a.pinned = True
    b = st.add(b"b", 1, 1, 0)  # pending
    c = st.add(b"c", 1, 1, 0)
    st.update_pose(c.kid, np.eye(4), 1.0, 1)  # failed registration
    d = st.add(b"d", 1, 1, 0)
    # No map yet: pending frames are not stale; the failed one goes first, pinned never
    assert [k.kid for k in st.evict_if_needed()] == [c.kid]
    st.map_ready_at = time.time() - 1
    st.add(b"e", 1, 1, 0)
    assert [k.kid for k in st.evict_if_needed()] == [b.kid]
    assert a.kid in [k.kid for k in st.frames()] and d.kid in [k.kid for k in st.frames()]


def test_store_keeps_map_anchors_through_a_keyframe_flood():
    # A flood of keyframes outrunning localization (pending, not yet stale) used to evict the
    # registered frames first; with 0 registered no refit could start and training never began
    st = KeyframeStore(max_keyframes=30, protect_recent=4, pending_ttl=1e9, conf_threshold=100,
                       min_registered=8)
    for i in range(8):   # the bootstrap map's anchors
        kf = st.add(b"r", 1, 1, float(i))
        st.update_pose(kf.kid, np.linalg.inv(_cam(i * 10.0, i * 1.0)), 1000.0, 1)
    st.map_ready_at = time.time()
    for i in range(100):  # pending frames waiting for a pose
        st.add(b"p", 1, 1, 100.0 + i)
        st.evict_if_needed()
    assert len(st.registered()) == 8 and len(st) == 30
    assert all(kf.w2c is None for kf in st.frames()[8:])          # the oldest pending went instead
    # Registered beyond the floor are evictable again (most redundant first)
    for i in range(30):
        kf = st.add(b"r", 1, 1, 300.0 + i)
        st.update_pose(kf.kid, np.linalg.inv(_cam(500.0 + i * 10.0, i * 1.0)), 1000.0, 1)
        st.evict_if_needed()
    assert len(st) == 30 and len(st.registered()) >= 8


def test_store_budget_bytes():
    st = KeyframeStore(budget_mb=3000 / 2 ** 20, max_keyframes=100, protect_recent=0, conf_threshold=0,
                       min_registered=0)
    for i in range(5):
        kf = st.add(b"z" * 1000, 1, 1, i)
        st.update_pose(kf.kid, np.linalg.inv(_cam(i * 5.0, i * 0.1)), 1.0, 1)
        st.evict_if_needed()
    assert st.nbytes <= 3000 and len(st) == 3


def test_views_table_is_zero_copy_and_filterable():
    st = _filled_store(9, test_every=4)
    st.set_pseudo([PseudoView("pseudo_0", pa.py_buffer(b"p" * 50), np.eye(4), np.eye(3), 160, 120,
                              0.3, 0.0, 2)])
    _, tbl = st.views(lambda c: c, lambda w, h: np.eye(3), lambda kf: 1.0)
    assert tbl.schema.names == [f.name for f in pa.schema(tbl.schema)]
    assert tbl.num_rows == 10
    assert select_rows(tbl, kind="pseudo").column("src_kid").to_pylist() == [2]
    assert select_rows(tbl, split="val", kind="real").num_rows == 2
    # jpeg column references the store's buffers (no copy)
    kf0 = st.frames()[0]
    chunk = tbl.column("jpeg").chunk(0)
    assert chunk.buffers()[2].address == kf0.jpeg.address


# ============================================================================ gate
def test_keyframe_gate_novelty_and_blur():
    g = KeyframeGate(min_motion=8.0, max_interval=None)
    a, b = _texture(1), _texture(2)
    assert g.offer(a, 0.0).accept
    assert g.offer(a, 0.1).reason == "redundant"
    assert g.offer(b, 0.2).accept
    for i in range(6):
        g.offer(_texture(10 + i), 0.3 + i)
    blurred = cv2.GaussianBlur(_texture(99), (31, 31), 12)
    assert g.offer(blurred, 9.0).reason == "blurry"
    assert g.avg_motion is not None and g.avg_motion > 8.0


def test_keyframe_gate_ignores_exposure_and_local_motion():
    base = _texture(5)
    # Auto exposure: the whole frame brighter (some highlights clip) or darker, same view
    exposed = np.clip(base.astype(np.float32) * 1.25 + 10, 0, 255).astype(np.uint8)
    darker = (base.astype(np.float32) * 0.75).astype(np.uint8)
    # A person walking through: one region changes, the camera does not move
    person = base.copy()
    h, w = base.shape[:2]
    person[h // 3:, : w // 4] = _texture(77)[h // 3:, : w // 4]
    # Camera motion: the whole view shifts
    moved = np.roll(base, w // 6, axis=1)

    legacy = KeyframeGate(min_motion=8.0, max_interval=None, exposure_invariant=False, cells=None)
    legacy.offer(base, 0.0)
    assert legacy.offer(exposed, 0.1).accept                 # the noise the new gate removes

    g = KeyframeGate(min_motion=8.0, max_interval=None)
    assert g.offer(base, 0.0).accept
    assert g.offer(exposed, 0.1).reason == "redundant"
    assert g.offer(darker, 0.15).reason == "redundant"
    assert g.offer(person, 0.2).reason == "redundant"
    assert g.offer(moved, 0.3).accept


def test_keyframe_gate_interval():
    g = KeyframeGate(min_motion=200.0, max_interval=5.0)
    img = _texture(3)
    assert g.offer(img, 0.0).accept
    assert not g.offer(img, 4.0).accept
    assert g.offer(img, 5.5).reason == "interval"


# ============================================================================ sources
@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for i in range(60):
        vw.write(cv2.cvtColor(np.roll(_texture(i % 4, 240, 320), i, axis=1), cv2.COLOR_RGB2BGR))
    vw.release()
    return path


def test_parse_source(tiny_video):
    assert parse_source("0") == ("camera", 0)
    assert parse_source("/dev/video2") == ("camera", "/dev/video2")
    assert parse_source("rtsp://cam/stream")[0] == "url"
    assert parse_source("flight://0.0.0.0:8815")[0] == "flight"
    assert parse_source(str(tiny_video))[0] == "file"
    with pytest.raises(FileNotFoundError):
        parse_source("/nope/missing.mp4")


def test_file_source_fast_reads_everything_resized(tiny_video):
    src = FrameSource(str(tiny_video), max_side=160, pace="auto")
    frames = list(src)
    assert src.pace == "fast" and len(frames) == 60 and src.frames_dropped == 0
    assert frames[0].image.shape == (120, 160, 3)
    assert frames[-1].t == pytest.approx(59 / 30)


def test_file_source_realtime_skips_when_consumer_is_slow(tiny_video):
    src = FrameSource(str(tiny_video), max_side=160, pace="realtime")
    seen = []
    for f in src:
        seen.append(f.index)
        time.sleep(0.1)  # 3 source frames per consumed frame
    assert src.frames_dropped > 20 and len(seen) < 40
    assert seen == sorted(seen)


def test_queue_source_keeps_only_newest():
    q = QueueFrameSource("flight://x", max_side=64)
    for i in range(5):
        q.push(i, _texture(i))
    q.end()
    frames = list(q)
    assert [f.t for f in frames] == [4.0] and q.frames_dropped == 4
    assert max(frames[0].image.shape[:2]) == 64


# ============================================================================ Arrow IPC
def test_ipc_roundtrip_over_pipe():
    r, w = os.pipe()
    with os.fdopen(w, "wb") as wf, os.fdopen(r, "rb") as rf:
        writer = IpcWriter(wf, EVENT_SCHEMA)
        writer.write(event_batch([{"kind": "pose", "version": 3, "frame": "/a.jpg",
                                   "w2c": np.eye(4).reshape(-1).tolist(), "conf": 900.0,
                                   "points": [1.0, 2.0, 3.0], "pix": [0.5, 0.5], "meta": {"x": 1}}]))
        writer.close()
        wf.close()
        batches = list(ipc_batches(rf))
    row = batches[0].to_pylist()[0]
    assert row["kind"] == "pose" and row["version"] == 3 and row["w2c"][0] == 1.0
    assert parse_meta(row["meta"]) == {"x": 1}
    cmd = command_batch("refit", ["/a", "/b"], ["--x", 1], iterations=5).to_pylist()[0]
    assert cmd["frames"] == ["/a", "/b"] and cmd["args"] == ["--x", "1"]
    assert parse_meta(cmd["params"]) == {"iterations": 5}
    assert command_batch("x").schema.equals(COMMAND_SCHEMA)


# ============================================================================ Gaussians
def _splats(n):
    params = {
        "means": torch.zeros((n, 3)), "scales": torch.zeros((n, 3)), "quats": torch.ones((n, 4)),
        "opacities": torch.zeros(n), "sh0": torch.zeros((n, 1, 3)), "shN": torch.zeros((n, 15, 3)),
    }
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in params.items()})
    opts = {k: torch.optim.Adam([{"params": splats[k], "lr": 1e-3, "name": k}]) for k in splats}
    loss = sum(p.sum() for p in splats.values())
    loss.backward()
    for o in opts.values():
        o.step()
    return splats, opts


def test_inject_gaussians_dedupes_by_voxel_and_extends_optimizer():
    splats, opts = _splats(4)  # all at the origin
    state = {"grad2d": torch.ones(4), "binoms": torch.ones((51, 51))}
    xyz = np.array([[0.0, 0, 0], [0.01, 0, 0], [1.0, 0, 0], [1.02, 0, 0], [2.5, 0, 0]], np.float32)
    rgb = np.full((5, 3), 0.5, np.float32)
    added, recycled = inject_gaussians(splats, opts, state, xyz, rgb, voxel=0.1)
    assert (added, recycled) == (2, 0)  # origin voxel occupied; 1.0/1.02 share a voxel
    assert splats["means"].shape == (6, 3)
    exp_avg = opts["means"].state[splats["means"]]["exp_avg"]
    assert exp_avg.shape == (6, 3) and torch.all(exp_avg[4:] == 0)
    assert state["grad2d"].shape == (6,) and state["binoms"].shape == (51, 51)
    assert torch.allclose(splats["sh0"][4:, 0], torch.zeros(3), atol=1e-6)  # rgb 0.5 -> SH 0


def test_inject_gaussians_recycles_dead_under_cap():
    splats, opts = _splats(4)
    with torch.no_grad():
        splats["opacities"][:] = torch.tensor([-10.0, 5.0, -10.0, 5.0])  # two dead
    xyz = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], np.float32)
    added, recycled = inject_gaussians(splats, opts, {}, xyz, np.zeros((3, 3), np.float32),
                                       voxel=0.1, cap_max=5)
    assert recycled == 2 and added == 1 and len(splats["means"]) == 5
    assert torch.allclose(splats["means"][[0, 2], 0], torch.tensor([1.0, 2.0]))


# ============================================================================ datasets
def test_with_pseudo_appends_views():
    class Base(torch.utils.data.Dataset):
        indices = [0, 1]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {"image": torch.zeros(4, 4, 3), "image_id": i}

    img = np.full((4, 4, 3), 200, np.uint8)
    view = PseudoView("pseudo_0", pa.py_buffer(encode_jpeg(img)), np.eye(4), np.eye(3), 4, 4, 0.4, 0, 1)
    ds = WithPseudo(Base(), [view], [1])
    assert len(ds) == 3 and ds.indices == [0, 1]
    assert ds[0]["is_pseudo"] is False
    item = ds[2]
    assert item["is_pseudo"] and item["image_id"] == 1 and item["loss_weight"] == pytest.approx(0.4)
    assert abs(float(item["image"].mean()) - 200) < 3


# ============================================================================ session orchestration
class FakeSidecar:
    def __init__(self):
        self.calls = []
        self.alive = True

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))


def _map_rows(paths, conf, reason="bootstrap", version=1):
    rows = []
    rng = np.random.default_rng(0)
    for i, p in enumerate(paths):
        pts = rng.normal(size=(50, 3)).astype(np.float32) + [0, 0, 5]
        rows.append({"kind": "map", "version": version, "frame": p,
                     "w2c": np.linalg.inv(_cam(i * 3.0, i * 0.2)), "conf": conf, "focal": 400.0,
                     "points": pts, "pix": rng.random((50, 2)).astype(np.float32),
                     "meta": {"reason": reason, "seconds": 1.0}})
    return rows


@pytest.fixture
def session(tmp_path, tiny_video):
    from hypersplat.stream.session import StreamConfig, StreamSession

    s = StreamSession(StreamConfig(source=str(tiny_video), work_dir=str(tmp_path / "w"),
                                   refit_min_unregistered=2, test_every=0))
    s.frames_dir.mkdir(parents=True)
    s.sidecar = FakeSidecar()
    s.source = type("Src", (), {"pace": "fast", "frames_read": 0, "frames_dropped": 0})()
    s.store.conf_threshold = 500
    s.ace_values = {"num_head_blocks": 1, "num_data_workers": 2, "aug_rotation": 5,
                    "repro_loss_soft_clamp": 40}
    return s


def _add(session, i):
    img = np.roll(_texture(i, 120, 160), i * 7, axis=1)
    session._add_keyframe(img, float(i), 1.0)
    return session.store.frames()[-1]


def test_session_bootstrap_map_track_and_refit(session):
    kfs = [_add(session, i) for i in range(4)]
    assert all(k.pinned for k in kfs) and not session.map_ready.is_set()
    session._on_event("map", _map_rows([k.ace_path for k in kfs[:3]], 900.0))
    assert session.map_ready.is_set() and session.transform is not None
    assert not any(k.pinned for k in session.store.frames())
    # The keyframe that arrived during the bootstrap is localized next
    assert [c for c in session.sidecar.calls if c[0] == "localize"][-1][1][0] == kfs[3].ace_path
    xyz, rgb = session.take_seeds()
    assert xyz.shape == (150, 3) and rgb.shape == (150, 3) and 0 <= rgb.min() <= rgb.max() <= 1
    version, table = session.views()
    assert table.num_rows == 3 and version == session.store.version
    K = np.array(table.column("K")[0].as_py()).reshape(3, 3)
    assert K[0, 0] == pytest.approx(400.0 * 160 / session.file_size[0]) and K[0, 2] == 80

    # Two frames fail to register -> refit of every localized frame
    for i in (10, 11):
        kf = _add(session, i)
        session._on_event("pose", [{"kind": "pose", "version": 1, "frame": kf.ace_path,
                                    "w2c": np.eye(4), "conf": 10.0, "focal": 400.0,
                                    "points": np.zeros((0, 3), np.float32),
                                    "pix": np.zeros((0, 2), np.float32), "meta": {}}])
    refits = [c for c in session.sidecar.calls if c[0] == "refit"]
    assert len(refits) == 1 and session._job == "refit"
    assert "--refine_calibration" in refits[0][1][3]
    session._on_event("map", _map_rows([k.ace_path for k in session.store.frames()], 900.0, "refit", 2))
    assert session._job is None and len(session.store.registered()) == len(session.store)
    assert session.stats()["refits"] == 1


def test_session_bootstrap_error_retries_with_more_frames(session):
    session._job = "bootstrap"
    session._on_event("error", [{"meta": {"cmd": "bootstrap", "msg": "boom"}}])
    assert session._job is None and session._bootstrap_target == 24 + 8 and session.failed is None


# ============================================================================ Flight
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_flight_push_and_get_roundtrip():
    import pyarrow.flight as flight
    from hypersplat.stream.flight import FrameFlightServer, get
    from hypersplat.stream.schema import FRAME_SCHEMA

    port = _free_port()
    frames = QueueFrameSource(f"flight://127.0.0.1:{port}", max_side=80)
    server = FrameFlightServer(f"grpc://127.0.0.1:{port}", frames=frames,
                               views=lambda: pa.table({"name": ["kf_000001"], "jpeg": [b"x"]}),
                               stats=lambda: {"keyframes": 1, "job": None})
    server.serve_in_background()
    try:
        client = flight.connect(f"grpc://127.0.0.1:{port}")
        writer, _ = client.do_put(flight.FlightDescriptor.for_path("frames"), FRAME_SCHEMA)
        writer.write_batch(pa.RecordBatch.from_pylist(
            [{"t": 0.5, "jpeg": encode_jpeg(_texture(7))}], schema=FRAME_SCHEMA))
        writer.close()
        client.close()
        got = []
        t = threading.Thread(target=lambda: got.append(next(iter(frames))), daemon=True)
        t.start()
        t.join(5)
        assert got and got[0].t == 0.5 and max(got[0].image.shape[:2]) == 80
        views = get(f"grpc://127.0.0.1:{port}", "views")
        assert views.column_names == ["name"]  # pixels never leave over do_get
        assert get(f"grpc://127.0.0.1:{port}", "stats").to_pylist()[0]["keyframes"] == 1
    finally:
        server.shutdown()


# ============================================================================ online Difix
def test_online_difix_round_weights_stats_and_stop(monkeypatch):
    from hypersplat.pipeline.params import SceneProfile
    from hypersplat.stream import difix_online
    from hypersplat.stream.difix_online import DifixSources, OnlineDifix

    class FakeFixer:
        def __init__(self, noise):
            self.noise = noise

        def to(self, device):
            return self

        def __call__(self, image, ref=None, scale=1.0, seed=None):
            rng = np.random.default_rng(0)
            return np.clip(image + rng.normal(0, self.noise, image.shape), 0, 255).astype(np.uint8)

    n = 6
    c2ws = np.stack([_cam(i * 4.0, i * 0.3) for i in range(n)])
    src = DifixSources(c2ws=c2ws, K=np.array([[80.0, 0, 40], [0, 80, 30], [0, 0, 1]]), width=80, height=60,
                       weights=np.ones(n), points=np.random.default_rng(0).normal(size=(500, 3)) + [0, 0, 6],
                       get_image=lambda i: _texture(i, 60, 80), ids=list(range(100, 100 + n)))
    render = lambda c2w, K, w, h: _texture(0, h, w)
    profile = SceneProfile()
    od = OnlineDifix(profile=profile)
    od.fixer = FakeFixer(noise=3.0)  # faithful: tiny changes
    views, sources, stats = od.run_round(render, src, step=100)
    assert stats["n_views"] == 8 and stats["n_kept"] == len(views) > 0
    assert all(0 < v.weight <= 0.5 for v in views) and {v.src_kid for v in views} <= set(src.ids)
    assert profile.get("difix.rounds")[0]["step"] == 100 and od.stop_reason is None
    # Heavy rewriting -> hallucination -> online rounds stop
    od.fixer = FakeFixer(noise=90.0)
    od.run_round(render, src, step=200)
    assert od.stop_reason is not None


def test_online_difix_steps_down_scale_on_oom():
    from hypersplat.stream.difix_online import SCALES, DifixSources, OnlineDifix

    class TightFixer:
        def __init__(self):
            self.scales = []

        def to(self, device):
            return self

        def __call__(self, image, ref=None, scale=1.0, seed=None):
            self.scales.append(scale)
            if scale > 0.75:
                raise torch.cuda.OutOfMemoryError("fake OOM")
            return image

    n = 4
    src = DifixSources(c2ws=np.stack([_cam(i * 4.0, i * 0.3) for i in range(n)]),
                       K=np.array([[80.0, 0, 40], [0, 80, 30], [0, 0, 1]]), width=80, height=60,
                       weights=np.ones(n), points=np.random.default_rng(0).normal(size=(200, 3)) + [0, 0, 6],
                       get_image=lambda i: _texture(i, 60, 80), ids=list(range(n)))
    od = OnlineDifix()
    od.fixer = TightFixer()
    views, _, stats = od.run_round(lambda c2w, K, w, h: _texture(0, h, w), src, step=1)
    assert SCALES[od.scale_idx] == 0.75 and stats["difix_scale"] == 0.75
    # 1.0 and 0.85 fail once, then every view runs at the remembered 0.75
    assert od.fixer.scales[:3] == [1.0, 0.85, 0.75] and set(od.fixer.scales[3:]) == {0.75}
    assert len(views) == stats["n_kept"]
    # views stay at the resolution Difix worked at (no blurry upsampled targets), K scaled
    assert all((v.width, v.height) == (56, 40) for v in views)
    assert np.isclose(views[0].K[0, 0], 80.0 * 56 / 80) and np.isclose(views[0].K[1, 2], 30.0 * 40 / 60)
    from hypersplat.stream.source import decode_jpeg
    assert decode_jpeg(views[0].jpeg.to_pybytes()).shape == (40, 56, 3)


def test_online_difix_chunked_round_matches_blocking_round():
    import time as _time

    from hypersplat.pipeline.params import SceneProfile
    from hypersplat.stream.difix_online import DifixSources, OnlineDifix

    class SlowFixer:
        def __init__(self):
            self.device_moves = []

        def to(self, device):
            self.device_moves.append(device)
            return self

        def __call__(self, image, ref=None, scale=1.0, seed=None):
            _time.sleep(0.02)
            return np.clip(image.astype(np.int16) + 2, 0, 255).astype(np.uint8)

    n = 6
    src = DifixSources(c2ws=np.stack([_cam(i * 4.0, i * 0.3) for i in range(n)]),
                       K=np.array([[80.0, 0, 40], [0, 80, 30], [0, 0, 1]]), width=80, height=60,
                       weights=np.ones(n), points=np.random.default_rng(0).normal(size=(500, 3)) + [0, 0, 6],
                       get_image=lambda i: _texture(i, 60, 80), ids=list(range(100, 100 + n)))
    render = lambda c2w, K, w, h: _texture(0, h, w)

    ref = OnlineDifix(profile=SceneProfile())
    ref.fixer = SlowFixer()
    views_ref, _, stats_ref = ref.run_round(render, src, step=100)

    od = OnlineDifix(profile=SceneProfile())
    od.fixer = SlowFixer()
    od.begin_round(src, step=100)
    assert od.round_pending
    chunks = 0
    while not od.step_round(render, budget_s=0.05):
        chunks += 1
        assert od.round_pending
    views, _, stats = od.finish_round(step=160)
    assert chunks >= 2 and stats["chunks"] == chunks + 1 and not od.round_pending
    assert [v.name for v in views] == [v.name for v in views_ref]
    assert np.allclose([v.weight for v in views], [v.weight for v in views_ref])
    assert stats["step_begin"] == 100 and stats["step"] == 160 and stats["difix_ms_per_view"] >= 15
    # parked on the CPU after every chunk
    assert od.fixer.device_moves.count("cpu") == stats["chunks"]

    capped = OnlineDifix(profile=SceneProfile())
    capped.fixer = SlowFixer()
    _, _, st = capped.run_round(render, src, step=1, max_views=3)
    assert st["n_views"] == 3


def test_sidecar_env_drops_trainer_allocator_config(monkeypatch):
    import subprocess
    from hypersplat.stream import sidecar as sc

    captured = {}

    class P:
        stdin = stdout = None

        def poll(self):
            return 0

    def fake_popen(cmd, **kw):
        captured.update(kw)
        return P()

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sc, "IpcWriter", lambda *a: None)
    monkeypatch.setattr(sc.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: None})())
    sc.AceSidecar(Path("/tmp/x"), lambda *a: None)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in captured["env"]


def test_session_bootstrap_retry_after_source_end_uses_all_frames(session):
    for i in range(5):
        _add(session, i)
    session._bootstrap_attempted = 3
    session._job = "bootstrap"
    session.source_ended = True
    session.file_size = session.file_size or (160, 120)
    session._on_event("error", [{"meta": {"cmd": "bootstrap", "msg": "boom"}}])
    boots = [c for c in session.sidecar.calls if c[0] == "bootstrap"]
    assert boots and len(boots[-1][1][0]) == 5 and session.failed is None
    session._on_event("error", [{"meta": {"cmd": "bootstrap", "msg": "boom again"}}])
    assert session.failed and "boom again" in session.failed


def _refit_map(session, n_registered, version):
    """Map event of a refit: the first n_registered keyframes register, the rest fail."""
    rows = _map_rows([k.ace_path for k in session.store.frames()], 900.0, "refit", version)
    for row in rows[n_registered:]:
        row["conf"] = 10.0
    session._on_event("map", rows)


def test_session_refits_keep_growing_after_source_end(session):
    kfs = [_add(session, i) for i in range(8)]
    session._on_event("map", _map_rows([k.ace_path for k in kfs[:3]], 900.0))
    session.source_ended = True
    refits = lambda: [c for c in session.sidecar.calls if c[0] == "refit"]
    session._maybe_refit(force=True)
    assert len(refits()) == 1
    # No live localizations after the source ended: only the map event's gain can chain refits
    _refit_map(session, 5, 2)
    assert len(refits()) == 2 and session._job == "refit"
    _refit_map(session, 6, 3)      # +1 < refit_min_gain: the chain stops
    assert len(refits()) == 2 and session._job is None
    assert len(session.store.unregistered()) == 2


def test_session_refit_growth_survives_gpu_lease_deferral(session):
    kfs = [_add(session, i) for i in range(8)]
    session._on_event("map", _map_rows([k.ace_path for k in kfs[:3]], 900.0))
    session._maybe_refit(force=True)
    assert session.gpu_lease.try_acquire("difix") is False  # the refit holds it
    session.gpu_lease.release("ace")
    assert session.gpu_lease.try_acquire("difix")
    _refit_map(session, 6, 2)       # gain while Difix holds the GPU: deferred, not dropped
    assert session._refit_deferred and len([c for c in session.sidecar.calls if c[0] == "refit"]) == 1
    session.release_gpu("difix")
    assert len([c for c in session.sidecar.calls if c[0] == "refit"]) == 2


@pytest.mark.parametrize("deterministic", [False, True])
def test_session_ace_deterministic_reaches_bootstrap_and_refit(session, deterministic):
    session.cfg.ace_deterministic = deterministic
    for i in range(3):
        _add(session, i)
    session.file_size = session.file_size or (160, 120)
    session._start_bootstrap()
    boot_args = [c for c in session.sidecar.calls if c[0] == "bootstrap"][-1][1][1]
    session._on_event("map", _map_rows([k.ace_path for k in session.store.frames()], 900.0))
    session._maybe_refit(force=True)
    refit_args = [c for c in session.sidecar.calls if c[0] == "refit"][-1][1][3]
    for args in (boot_args, refit_args):
        assert ("--deterministic" in args) == deterministic
        if deterministic:
            assert args[args.index("--deterministic") + 1] == "True"


# ============================================================================ dashboard / stop
def test_stop_file_ignores_stale_and_honours_new(tmp_path):
    from hypersplat.stream.training import StopFile

    stop = tmp_path / "STOP"
    stop.touch()
    os.utime(stop, (time.time() - 60, time.time() - 60))  # left over from an earlier run
    sf = StopFile(str(tmp_path))
    assert not sf.requested()
    stop.touch()  # written during this run (e.g. Ctrl+C in the demo script)
    assert sf.requested()


def test_wait_for_map_returns_on_stop_request(session):
    t0 = time.time()
    assert session.wait_for_map(timeout=10, should_stop=lambda: True) is False
    assert time.time() - t0 < 1


def test_session_phase_hints(session):
    assert session.phase()[0] == "waiting"
    _add(session, 0)
    session.gate.offered = 1
    phase, hint = session.phase()
    assert phase == "collecting" and "1/24" in hint
    session._job = "bootstrap"
    assert session.phase()[0] == "mapping"
    session._job = None
    session.map_ready.set()
    assert session.phase()[0] == "tracking"
    session.failed = "boom"
    assert session.phase() == ("failed", "boom")


def test_dashboard_serves_page_status_and_feed():
    import json as _json
    import urllib.request
    from hypersplat.stream.dashboard import Dashboard

    class FakeSession:
        def __init__(self):
            self.img = _texture(5, 90, 160)

        def stats(self):
            return {"frames_read": 3, "keyframes": 1}

        def phase(self):
            return "collecting", "Move the camera"

        def latest_frame(self):
            return (self.img, 0.1, True, 1)

        def recently_accepted(self):
            return True

    port = _free_port()
    dash = Dashboard(FakeSession(), port=port).start()
    try:
        base = f"http://127.0.0.1:{port}"
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "/frame.jpg" in page and "viewer" in page
        dash.viewer_port = 8080
        dash.update_training(step=10, max_steps=100, loss=None)
        st = _json.loads(urllib.request.urlopen(base + "/status.json", timeout=5).read())
        assert st["phase"] == "collecting" and st["viewer_port"] == 8080
        assert st["training"] == {"step": 10, "max_steps": 100}  # None values are not recorded
        with urllib.request.urlopen(base + "/feed.mjpg", timeout=5) as r:
            assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            head = r.read(16384)
        assert head.startswith(b"--frame\r\nContent-Type: image/jpeg") and b"\xff\xd8" in head
        # the part is terminated by the next boundary immediately (browsers draw it then)
        assert b"\xff\xd9\r\n--frame\r\n" in head
        with urllib.request.urlopen(base + "/frame.jpg", timeout=5) as r:
            assert r.headers["Content-Type"] == "image/jpeg"
            frame = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
        assert frame.shape == (90, 160, 3)
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/nope", timeout=5)
    finally:
        dash.stop()


# ============================================================================ phone detection
def _fake_camera(root, n, name, driver, vendor=None, removable=None, index=0, loopback=False):
    """sysfs layout: video4linux/videoN/device -> .../<usbdev>/<iface> with driver link."""
    usb = root / "devices" / (f"platform/v4l2loopback.{n}" if loopback else f"usb1/1-{n}")
    iface = usb if loopback else usb / f"1-{n}:1.0"
    (root / "drivers" / driver).mkdir(parents=True, exist_ok=True)
    iface.mkdir(parents=True, exist_ok=True)
    (iface / "driver").symlink_to(root / "drivers" / driver)
    if vendor:
        (usb / "idVendor").write_text(vendor + "\n")
    if removable:
        (usb / "removable").write_text(removable + "\n")
    node = root / "class" / f"video{n}"
    node.mkdir(parents=True)
    (node / "name").write_text(name + "\n")
    (node / "index").write_text(f"{index}\n")
    (node / "device").symlink_to(iface)


def test_phone_detection_prefers_android_uvc_and_skips_builtin(tmp_path, monkeypatch):
    from hypersplat.stream import phone

    _fake_camera(tmp_path, 0, "ACER HD User Facing", "uvcvideo", "04f2", "fixed")
    _fake_camera(tmp_path, 1, "ACER HD User Facing", "uvcvideo", "04f2", "fixed", index=1)
    cams = phone.list_cameras(tmp_path / "class")
    assert [c["device"] for c in cams] == ["/dev/video0"]  # metadata node skipped
    assert phone.find_uvc_phone(cams) is None  # built-in (fixed port) is never "the phone"

    _fake_camera(tmp_path, 4, "Pixel 8 Webcam", "uvcvideo", "18d1", "removable")
    d = phone.detect(tmp_path / "class")  # webcam mode wins before adb is even asked
    assert d["mode"] == "uvc" and d["device"] == "/dev/video4" and "Google" in d["label"]


def test_phone_detection_external_unknown_vendor(tmp_path):
    from hypersplat.stream import phone

    _fake_camera(tmp_path, 0, "Built-in", "uvcvideo", "04f2", "fixed")
    _fake_camera(tmp_path, 2, "Some Phone", "uvcvideo", "abcd", "unknown")
    assert phone.find_uvc_phone(phone.list_cameras(tmp_path / "class"))["device"] == "/dev/video2"


def test_phone_detection_android_adb_route_and_problems(tmp_path, monkeypatch):
    from hypersplat.stream import android, phone

    _fake_camera(tmp_path, 0, "Built-in", "uvcvideo", "04f2", "fixed")
    monkeypatch.setattr(android, "adb_path", lambda: None)
    monkeypatch.setattr(android, "server_path", lambda: (None, None))
    d = phone.detect(tmp_path / "class")
    assert d["mode"] is None and "scrcpy" in d["problems"][0]

    monkeypatch.setattr(android, "adb_path", lambda: "/tools/adb")
    monkeypatch.setattr(android, "server_path", lambda: ("/tools/scrcpy-server", "4.1"))
    monkeypatch.setattr(android, "adb_devices", lambda adb=None: [])
    assert "USB debugging" in phone.detect(tmp_path / "class")["problems"][0]
    monkeypatch.setattr(android, "adb_devices", lambda adb=None: [{"serial": "10BD", "state": "unauthorized"}])
    assert "Allow USB debugging" in phone.detect(tmp_path / "class")["problems"][0]
    monkeypatch.setattr(android, "adb_devices", lambda adb=None: [{"serial": "10BD", "state": "device"}])
    d = phone.detect(tmp_path / "class")
    assert d["mode"] == "android" and d["device"] == "android://10BD" and d["problems"] == []


def test_android_spec_and_server_args():
    from hypersplat.stream.android import parse_android_spec, server_args
    from hypersplat.stream.source import parse_source

    assert parse_source("android://") == ("android", "android://")
    assert parse_android_spec("android://") == {"serial": None, "camera": "back", "max_size": 1920, "fps": 30,
                                                "bitrate": None}
    o = parse_android_spec("android://10BD?camera=front&max_size=1280&fps=24")
    assert o == {"serial": "10BD", "camera": "front", "max_size": 1280, "fps": 24, "bitrate": None}
    args = server_args("4.1", "0badc0de", "back", 1920, 30)
    assert args[0] == "4.1"  # the server refuses a client of another version
    for kv in ("scid=0badc0de", "tunnel_forward=true", "video_source=camera", "camera_facing=back",
               "raw_stream=true", "audio=false", "control=false", "video_codec=h264"):
        assert kv in args
    assert not any(a.startswith("video_bit_rate=") for a in args)


def test_android_over_wifi_spec():
    import pytest as _pytest

    from hypersplat.stream.android import (_bit_rate, is_network_serial, parse_android_spec,
                                           server_args)

    o = parse_android_spec("android://192.168.1.23:5555?camera=back&bitrate=6M")
    assert o["serial"] == "192.168.1.23:5555" and o["bitrate"] == "6M"
    assert is_network_serial(o["serial"]) and is_network_serial("phone.lan:37123")
    assert not is_network_serial("LRFYSS6T9LGYSSFE") and not is_network_serial(None)
    assert "video_bit_rate=6000000" in server_args("4.1", "ab", "back", 1920, 30, "6M")
    assert _bit_rate("800K") == 800_000 and _bit_rate("2500000") == 2_500_000
    with _pytest.raises(ValueError):
        _bit_rate("fast")


def test_gpu_lease_defers_refit_until_difix_releases(session):
    kfs = [_add(session, i) for i in range(4)]
    session._on_event("map", _map_rows([k.ace_path for k in kfs[:3]], 900.0))
    seen = session.views_registered
    assert seen == 3
    assert session.gpu_lease.try_acquire("difix")
    for i in (10, 11):  # two failures -> a refit is due, but Difix holds the GPU
        kf = _add(session, i)
        session._on_event("pose", [{"kind": "pose", "version": 1, "frame": kf.ace_path,
                                    "w2c": np.eye(4), "conf": 10.0, "focal": 400.0,
                                    "points": np.zeros((0, 3), np.float32),
                                    "pix": np.zeros((0, 2), np.float32), "meta": {}}])
    assert not [c for c in session.sidecar.calls if c[0] == "refit"] and session._refit_deferred
    session.source_ended = True
    session._inflight = 0
    assert not session.finished  # a deferred refit is still outstanding
    assert not session.gpu_lease.try_acquire("ace")
    session.release_gpu("difix")
    assert len([c for c in session.sidecar.calls if c[0] == "refit"]) == 1
    assert session.gpu_lease.holder == "ace" and session._job == "refit"
    session._on_event("error", [{"meta": {"cmd": "refit", "msg": "oom"}}])
    assert session.gpu_lease.holder is None


def test_sidecar_send_never_blocks_caller():
    import queue
    from hypersplat.stream.sidecar import AceSidecar

    sc = AceSidecar.__new__(AceSidecar)
    sc.proc = type("P", (), {"poll": lambda self: None})()
    sc._outbox = queue.Queue()
    for i in range(1000):  # far more than a 64 KB pipe holds; nothing drains it here
        sc.localize(f"/f/{i}.jpg")
    assert sc._outbox.qsize() == 1000


def test_map_cache_is_keyed_to_the_capture_file(tmp_path):
    import json as _json
    import os as _os
    from types import SimpleNamespace

    from hypersplat.stream.session import StreamConfig, StreamSession

    video = tmp_path / "cap.mp4"
    video.write_bytes(b"x" * 10)
    cache = tmp_path / "cache"
    fake = SimpleNamespace(cfg=StreamConfig(source=str(video), map_cache=str(cache)), kind="file")
    d, meta, usable = StreamSession._map_cache(fake)
    assert d == cache and not usable  # nothing cached yet
    cache.mkdir()
    (cache / "head.pt").write_bytes(b"h")
    (cache / "poses.txt").write_text("p")
    (cache / "meta.json").write_text(_json.dumps({**meta, "image_resolution": 480}))
    assert StreamSession._map_cache(fake)[2]
    _os.utime(video, (1, 1))  # the capture changed: the cache no longer applies
    assert not StreamSession._map_cache(fake)[2]
    assert StreamSession._map_cache(SimpleNamespace(cfg=fake.cfg, kind="camera")) is None


def test_map_cache_relative_path_is_made_absolute(tmp_path, monkeypatch):
    # The ACE sidecar runs in scripts/acezero: a relative cache dir handed to it as-is
    # missed the cache and saved a new one under scripts/acezero
    from types import SimpleNamespace

    from hypersplat.stream.session import StreamConfig, StreamSession

    monkeypatch.chdir(tmp_path)
    (tmp_path / "cap.mp4").write_bytes(b"x")
    fake = SimpleNamespace(cfg=StreamConfig(source="cap.mp4", map_cache="cache/cap"), kind="file")
    d, _, _ = StreamSession._map_cache(fake)
    assert d.is_absolute() and d == tmp_path / "cache" / "cap"


def test_hard_view_sampling_prefers_underfit_views():
    from hypersplat.stream.training import StreamTrainSet

    ts = StreamTrainSet(session=None, hard_mix=0.5)
    ts._names, ts._kids, ts._pseudo = ["a", "b", "c", "p"], [1, 2, 3, 2], [False, False, False, True]
    assert (ts.hardness() == 1).all()  # nothing scored yet: novelty-only
    ts.report_loss([1, 2, 3, -1], [0.02, 0.04, 0.16, 9.0])  # kid -1 (pseudo) is ignored
    h = ts.hardness()
    assert h[1] == 1.0  # the median view is neutral
    assert h[0] == pytest.approx(0.75) and h[2] == pytest.approx(0.5 + 0.5 * 4.0)  # clipped at 4x
    assert h[3] == 1.0  # pseudo views stay neutral
    ts.report_loss([3], [0.02])  # EMA, not last value
    assert 0.02 < ts.loss_ema[3] < 0.16



@pytest.mark.parametrize("prefetch", [0, 2])
def test_sampler_raises_when_session_fails_without_views(prefetch):
    # Training used to wait forever at step 0 once no keyframe could register again
    import threading as _threading

    from hypersplat.stream.schema import VIEW_SCHEMA
    from hypersplat.stream.training import StreamTrainSet

    class NoViews:
        failed = None

        def views(self):
            return 1, VIEW_SCHEMA.empty_table()

        def stats(self):
            return {"registered": 0}

    session = NoViews()
    _threading.Timer(0.2, lambda: setattr(session, "failed", "mapping stalled")).start()
    ts = StreamTrainSet(session, prefetch=prefetch)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="mapping stalled"):
        next(iter(ts))
    assert time.time() - t0 < 5



def test_blocked_mapping_fails_session_once_nothing_can_register():
    import threading as _threading
    import types

    from hypersplat.stream.session import StreamSession

    st = KeyframeStore(conf_threshold=100)
    pending = st.add(b"p", 1, 1, 0.0)
    s = types.SimpleNamespace(store=st, source_ended=False, failed=None, _blocked_warned=False,
                              _cond=_threading.Condition())
    with s._cond:
        StreamSession._mapping_blocked(s)          # live source: warn only
        assert s._blocked_warned and s.failed is None
        s.source_ended = True
        StreamSession._mapping_blocked(s)          # a frame still waits for a pose
        assert s.failed is None
        st.update_pose(pending.kid, np.eye(4), 1.0, 1)   # it failed to register
        StreamSession._mapping_blocked(s)
        assert s.failed and "mapping stalled" in s.failed
