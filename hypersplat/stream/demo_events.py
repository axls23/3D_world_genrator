"""Serve the stream dashboard (base page at /, telemetry view at /streaming) from a demo event source.

    python -m hypersplat.stream.demo_events [--video demo/input/drone_demo.mp4] [--port 8090]

No camera, ACE or GPU is needed: frames come from a video file (or a generated texture), camera poses
walk a spiral with a few bad poses injected, and training loss/Gaussian count are synthetic and written
to a TensorBoard directory the way the trainer does. Events are paced at a fixed rate so the page
visibly updates; it is a stand-in for the live run, not a measurement.
"""

import argparse
import math
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np

from hypersplat.stream.dashboard import Dashboard

BAD_POSES = {17, 18, 33}   # injected outliers drawn as red markers


class _Store:
    def __init__(self):
        self.kfs: List[SimpleNamespace] = []

    def registered(self):
        return list(self.kfs)

    def __len__(self):
        return len(self.kfs)


class DemoSession:
    """The slice of StreamSession the dashboard reads."""

    def __init__(self, video: Optional[str], eps: float, total_steps: int):
        self.cap = cv2.VideoCapture(video) if video else None
        self.eps, self.total_steps = eps, total_steps
        self.store, self.map_version = _Store(), 1
        self.frames, self.accepted, self._frame, self._last_kf = 0, 0, None, 0.0
        self.ended = False

    def stats(self) -> Dict:
        n = len(self.store)
        return {"source": "demo events", "kind": "demo", "frames_read": self.frames, "frames_dropped": 0,
                "keyframes_offered": self.frames // 2, "keyframes_accepted": self.accepted, "keyframes": n,
                "registered": n, "pending": 0, "unregistered": 0, "pseudo": 0, "evicted": 0, "store_mb": n * 0.4,
                "budget_mb": 256, "map_version": self.map_version, "job": None, "source_ended": self.ended,
                "recording_stopped": None, "live": True, "reconnects": 0, "failed": None,
                "depth_maps": None, "depth_rejected": None, "depth_rel_err": None, "depth_ms": None}

    def phase(self):
        return ("ended", "Demo source ended") if self.ended else ("tracking", "Replaying demo events")

    def latest_frame(self):
        return None if self._frame is None else (self._frame, 0.1, True, self.frames)

    def recently_accepted(self) -> bool:
        return time.time() - self._last_kf < 0.6

    def to_trainer(self, c2w):
        return c2w

    def next_frame(self):
        img = None
        if self.cap is not None:
            ok, img = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self.cap.read()
            if ok:
                img = cv2.resize(img, (640, 360))
        if img is None:
            x = np.linspace(0, 255, 640, dtype=np.uint8)
            img = np.tile(np.stack([x, x[::-1], np.full(640, 90, np.uint8)], -1), (360, 1, 1))
        self._frame, self.frames = img, self.frames + 1

    def add_keyframe(self, i: int):
        a = 0.28 * i
        r = 1.0 + 0.05 * i
        x, y, z = r * math.cos(a), r * math.sin(a), 0.02 * i
        if i in BAD_POSES:
            x, y = x + 4.0, y - 3.5       # a camera pose the estimator got wrong
        c2w = np.eye(4)
        c2w[:3, 3] = (x, y, z)
        self.store.kfs.append(SimpleNamespace(kid=i, name=f"frame_{i:05d}.jpg", w2c=np.linalg.inv(c2w),
                                              conf=0.35 if i in BAD_POSES else 0.9))
        self.accepted += 1
        self._last_kf = time.time()


class DemoDashboard(Dashboard):
    def pose_path(self) -> Dict:
        out = super().pose_path()
        for p in out["poses"]:
            p["flagged"] = p["kid"] in BAD_POSES
        return out

    def status(self) -> Dict:
        s = super().status()
        s["anomaly_flags"] = [{"kid": p["kid"], "reason": f"Pose {p['name']} jumps away from the camera path"}
                              for p in self.pose_path()["poses"] if p["flagged"]]
        return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", default="demo/input/drone_demo.mp4")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--eps", type=float, default=5.0, help="events per second (the PDF's replay rate)")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()

    from torch.utils.tensorboard import SummaryWriter
    tb = tempfile.mkdtemp(prefix="demo_events_tb_")
    writer = SummaryWriter(tb, flush_secs=1)
    s = DemoSession(a.video, a.eps, a.steps)
    dash = DemoDashboard(s, port=a.port, host=a.host, tb_dir=tb).start()
    print(f"[demo-events] base UI {dash.url}   telemetry view {dash.url}/streaming   (Ctrl+C to stop)", flush=True)
    rng, i, step = np.random.default_rng(0), 0, 0
    try:
        while step < a.steps:
            s.next_frame()
            if s.frames % 2 == 0:
                s.add_keyframe(i)
                i += 1
            step += max(1, a.steps // 300)
            loss = 0.25 * math.exp(-step / 700) + 0.02 + 0.004 * rng.standard_normal()
            if 1500 <= step < 1530:
                loss += 0.08                                  # a loss spike
            writer.add_scalar("train/loss", loss, step)
            writer.add_scalar("train/num_GS", 40000 + 90 * step, step)
            writer.add_scalar("train/mem", 3.0 + step / 2000, step)
            dash.update_training(step=step, max_steps=a.steps, num_gs=40000 + 90 * step, loss=loss)
            time.sleep(1.0 / a.eps)
        s.ended = True
        dash.update_training(done=True)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        dash.stop()


if __name__ == "__main__":
    main()
