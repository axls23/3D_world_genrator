"""Record / replay of the ACE event stream (hypersplat/stream/record.py)."""
import time

import numpy as np

from hypersplat.stream.record import EventRecorder, ReplaySidecar, load_recording


def _row(kind, frame, w2c, conf, n=0, meta=None, version=1):
    return {"kind": kind, "version": version, "frame": frame, "w2c": w2c, "conf": conf, "focal": 500.0,
            "points": np.ones((n, 3), np.float32), "pix": np.ones((n, 2), np.float32), "meta": meta or {}}


def _record(path):
    w = np.eye(4)
    w[0, 3] = 1.5
    rec = EventRecorder(path)
    rec.add("map", [_row("map", "/x/kf_000000.jpg", w, 900.0, 3, {"reason": "load"}),
                    _row("map", "/x/kf_000001.jpg", None, 0.0, 0, {"reason": "load"})], -1)
    rec.add("pose", [_row("pose", "/x/kf_000002.jpg", w, 700.0, 2)], 3)
    rec.add("reloc", [_row("reloc", "/x/kf_000000.jpg", w, 800.0, 1, {"anchored": True}, 2)], 5)
    rec.close()
    return w


def _until(pred, timeout=2.0):
    t = time.time()
    while not pred() and time.time() - t < timeout:
        time.sleep(0.01)
    return pred()


def test_round_trip(tmp_path):
    w = _record(tmp_path / "e.arrows")
    ev = load_recording(tmp_path / "e.arrows")
    assert [(s, k, len(r)) for s, k, r in ev] == [(-1, "map", 2), (3, "pose", 1), (5, "reloc", 1)]
    np.testing.assert_array_equal(ev[0][2][0]["w2c"], w)
    assert ev[0][2][1]["w2c"] is None
    assert ev[0][2][0]["points"].shape == (3, 3) and ev[1][2][0]["pix"].shape == (2, 2)
    assert ev[2][2][0]["meta"] == {"anchored": True} and ev[2][2][0]["version"] == 2


def test_replay_gates_on_map_request_frame_request_and_step(tmp_path):
    _record(tmp_path / "e.arrows")
    got, step = [], [-1]
    sc = ReplaySidecar(str(tmp_path / "e.arrows"), lambda k, r: got.append((step[0], k)), lambda: step[0])
    try:
        time.sleep(0.15)
        assert got == []                                  # the map waits for load / bootstrap
        sc.load("head", "poses", [])
        assert _until(lambda: got == [(-1, "map")])
        step[0] = 4
        time.sleep(0.15)
        assert got == [(-1, "map")]                       # pose not sent for localization yet
        sc.localize("/x/kf_000002.jpg")
        assert _until(lambda: got[-1] == (4, "pose"))
        time.sleep(0.15)
        assert len(got) == 2                              # reloc waits for step 5
        step[0] = 5
        assert _until(lambda: got[-1] == (5, "reloc"))
    finally:
        sc.close()
