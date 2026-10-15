"""Live frame buffer and capture-fps request (hypersplat/stream/source.py); no camera needed."""

import threading
import time

import cv2
import numpy as np
import pytest

from hypersplat.stream.session import StreamSession
from hypersplat.stream.source import Frame, FrameSource, _FrameQueue


@pytest.fixture(autouse=True)
def _no_real_devices(monkeypatch):
    """A reconnect path must never open the machine's real camera (it turns the webcam on).
    Recorded as well as raised: ``_reopen`` swallows errors and retries."""
    opened = []

    def refuse(*args, **kwargs):
        opened.append(args)
        raise AssertionError(f"test opened a real capture device {args!r}")

    monkeypatch.setattr(cv2, "VideoCapture", refuse)
    yield
    assert not opened, f"test tried to open real capture devices: {opened}"


def _frame(i):
    return Frame(i, float(i), np.zeros((2, 2, 3), np.uint8))


def test_queue_keeps_every_frame_in_order_within_capacity():
    q = _FrameQueue(5)
    for i in range(5):
        q.put(_frame(i))
    assert [q.get(0.1).index for _ in range(5)] == [0, 1, 2, 3, 4]
    assert q.dropped == 0


def test_queue_drops_oldest_only_past_capacity():
    q = _FrameQueue(3)
    for i in range(5):
        q.put(_frame(i))
    assert q.dropped == 2
    assert [q.get(0.1).index for _ in range(3)] == [2, 3, 4]
    assert q.get(0.05) is None  # empty, not closed: times out


def test_queue_close_unblocks_reader():
    q = _FrameQueue(2)
    q.close()
    assert q.get(1.0) is None


MJPG = cv2.VideoWriter_fourcc(*"MJPG")
YUYV = cv2.VideoWriter_fourcc(*"YUYV")


class _FakeCap:
    """cv2.VideoCapture stand-in: ``frames`` reads succeed (forever when None); ``mjpg_ok=False``
    is a camera that stops delivering frames once switched to MJPG; ``max_fps`` caps the
    negotiated rate; ``fourcc=0`` is a backend that does not report its format."""

    def __init__(self, frames=None, mjpg_ok=True, size=(4, 6), max_fps=240.0, fourcc=YUYV):
        self.props = {cv2.CAP_PROP_FOURCC: float(fourcc), cv2.CAP_PROP_FPS: 30.0}
        self.frames, self.mjpg_ok, self.size, self.max_fps = frames, mjpg_ok, size, max_fps
        self.reads = 0
        self.released = False

    def set(self, prop, value):
        self.props[prop] = min(value, self.max_fps) if prop == cv2.CAP_PROP_FPS else value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def isOpened(self):
        return not self.released

    def read(self):
        if self.released or (not self.mjpg_ok and self.props[cv2.CAP_PROP_FOURCC] == MJPG):
            return False, None
        if self.frames is not None and self.reads >= self.frames:
            return False, None
        self.reads += 1
        return True, np.full((*self.size, 3), self.reads % 256, np.uint8)

    def grab(self):
        return False

    def release(self):
        self.released = True


def _live_source(kind="camera", capture_fps=60, cap=None, buffer_seconds=1.0, fps=30.0):
    """FrameSource without opening a device (the fields __init__ would set)."""
    src = FrameSource.__new__(FrameSource)
    src.kind, src.spec, src.target, src.live = kind, "0", 0, True
    src.capture_fps, src.buffer_seconds, src.fps, src.max_side = capture_fps, buffer_seconds, fps, 960
    src.reconnect_timeout, src.reconnects, src.status = 5.0, 0, None
    src.newest, src._capture_rate, src.android = None, None, None
    src.frames_read = src.frames_skipped = 0
    src._stop, src._slot, src._grabber = threading.Event(), None, None
    src.cap = cap if cap is not None else _FakeCap()
    return src


def test_capture_fps_requested_from_local_cameras_only():
    src = _live_source()
    for kind, asked in (("camera", True), ("url", False), ("android", False)):
        src.kind, cap = kind, _FakeCap()
        rate = src._request_fps(cap)
        assert (cap.props.get(cv2.CAP_PROP_FPS) == 60.0) is asked
        assert rate == (60.0 if asked else None)
        if asked:
            assert cap.props[cv2.CAP_PROP_FOURCC] == MJPG


def test_mjpg_kept_when_frames_keep_flowing():
    src, cap = _live_source(), _FakeCap()
    assert src._request_fps(cap) == 60.0
    assert cap.props[cv2.CAP_PROP_FOURCC] == MJPG and cap.props[cv2.CAP_PROP_FPS] == 60.0
    assert cap.reads == 1  # the one test read


def test_mjpg_falls_back_when_camera_stops_delivering():
    src, cap = _live_source(), _FakeCap(mjpg_ok=False)
    assert src._request_fps(cap) is None
    assert cap.props[cv2.CAP_PROP_FOURCC] == YUYV and cap.props[cv2.CAP_PROP_FPS] == 30.0
    assert cap.read()[0]  # frames flow again in the camera's own format


def test_unknown_format_is_not_switched():
    # A backend that reports FOURCC 0 could not be switched back: ask only for the rate
    src, cap = _live_source(), _FakeCap(mjpg_ok=False, fourcc=0)
    assert src._request_fps(cap) == 60.0
    assert cap.props[cv2.CAP_PROP_FOURCC] == 0 and cap.read()[0]


def test_reopen_validates_after_requesting_fps(monkeypatch):
    # A camera that stops in MJPG: reopen falls back before its read check instead of
    # accepting a capture that delivers nothing and looping through reconnects
    caps = []

    def open_cap(target, *args):
        caps.append(_FakeCap(mjpg_ok=False))
        return caps[-1]

    monkeypatch.setattr(cv2, "VideoCapture", open_cap)
    src = _live_source()
    src._capture_rate = 60.0  # MJPG worked before the camera dropped off
    assert src._reopen()
    assert len(caps) == 1 and src.cap is caps[0] and src.reconnects == 1
    assert caps[0].props[cv2.CAP_PROP_FOURCC] == YUYV and caps[0].read()[0]
    assert src._capture_rate is None and src._live_rate() == 30.0


def test_reopen_rejects_a_camera_that_never_delivers(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda target, *args: _FakeCap(frames=0))
    src = _live_source()
    src.reconnect_timeout = 0.3
    t0 = time.time()
    assert not src._reopen() and src.reconnects == 0
    assert time.time() - t0 < 1.0  # the retry wait is capped by the timeout


class _FakeAndroid:
    opts = {"fps": 60}


def test_queue_sized_from_requested_fps():
    # OpenCV reports 25-30 fps for both; the buffer must hold buffer_seconds at the requested 60
    cam = _live_source(fps=30.0)
    cam._capture_rate = cam._request_fps(cam.cap)
    assert cam._live_rate() == 60.0
    phone = _live_source(kind="android", capture_fps=None, fps=25.0)
    phone.android = _FakeAndroid()
    assert phone._live_rate() == 60.0
    slow = _live_source(fps=30.0, cap=_FakeCap(max_fps=30.0))  # MJPG, but 30 fps at most
    slow._capture_rate = slow._request_fps(slow.cap)
    assert slow._live_rate() == 30.0
    fallback = _live_source(fps=30.0, cap=_FakeCap(mjpg_ok=False))
    fallback._capture_rate = fallback._request_fps(fallback.cap)  # runs at its reported rate
    assert fallback._live_rate() == 30.0
    assert _live_source(kind="url", fps=25.0)._live_rate() == 25.0


def test_queue_resize_keeps_the_newest():
    q = _FrameQueue(5)
    for i in range(5):
        q.put(_frame(i))
    q.resize(2)
    assert q.dropped == 3 and [q.get(0.1).index for _ in range(2)] == [3, 4]
    q.resize(4)
    for i in range(5, 10):
        q.put(_frame(i))
    assert [q.get(0.1).index for _ in range(4)] == [6, 7, 8, 9]


def _drain(src, every=0.0):
    """Run the live iterator until the fake capture runs out, sleeping per consumed frame."""
    got = []
    for frame in src._iter_live():
        got.append(frame.index)
        time.sleep(every)
    src._grabber.join(timeout=5)
    return got


def test_live_fifo_and_newest_with_a_slow_consumer():
    src = _live_source(cap=_FakeCap(frames=300))
    src._capture_rate = src._request_fps(src.cap)
    src.cap.reads = 0  # the test read is not a frame
    src._reopen = lambda: False  # the end of the fake capture: no real camera to wait for
    got = _drain(src, every=0.002)
    assert src._slot._frames.maxlen == 60
    assert got == sorted(got) and len(set(got)) == len(got)  # in order, no repeats
    assert got[-60:] == list(range(240, 300))  # the tail was buffered; only older frames dropped
    assert len(got) + src._slot.dropped == 300
    assert src.newest.index == 299 and src.frames_read == 300


def test_fifo_resized_when_a_reconnect_changes_the_rate():
    src = _live_source(cap=_FakeCap(frames=5))
    src._capture_rate = 60.0
    reopens = []

    def reopen():  # back once, without MJPG this time (30 fps), then gone
        reopens.append(1)
        if len(reopens) > 1:
            return False
        src.cap, src._capture_rate = _FakeCap(frames=5), None
        return True

    src._reopen = reopen
    got = _drain(src)
    assert src._slot._frames.maxlen == 30 and len(reopens) == 2
    assert src.frames_read == 10 and got == sorted(got) and src.newest.index == 9


class _FakeSource:
    def __init__(self, newest=None):
        self.newest = newest


def _session(latest, newest):
    sess = StreamSession.__new__(StreamSession)
    sess._latest, sess.source = latest, _FakeSource(newest)
    return sess


def test_latest_frame_prefers_a_newer_grabbed_frame():
    img_a, img_b = np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)
    # Ingest is at frame 3 (a keyframe), the grabber at frame 9: preview frame 9, not judged yet
    sess = _session((img_a, 3.0, True, 4), Frame(9, 9.0, img_b))
    image, t, accepted, seq = sess.latest_frame()
    assert image is img_b and t == 9.0 and accepted is False and seq == 10
    # Ingest caught up with that frame: its keyframe flag, the same sequence number
    sess._latest = (img_b, 9.0, True, 5)
    image, t, accepted, seq2 = sess.latest_frame()
    assert image is img_b and accepted is True and seq2 == seq
    # The next grabbed frame: a new, larger sequence number
    sess.source.newest = Frame(10, 10.0, img_a)
    assert sess.latest_frame()[3] == 11
    # Before the first ingested frame, and sources without ``newest`` (files, flight)
    assert _session(None, Frame(0, 0.0, img_a)).latest_frame()[2:] == (False, 1)
    assert _session((img_a, 1.0, True, 2), None).latest_frame() == (img_a, 1.0, True, 2)
