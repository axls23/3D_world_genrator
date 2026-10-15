"""Live frame buffer and capture-fps request (hypersplat/stream/source.py); no camera needed."""

import cv2
import numpy as np

from hypersplat.stream.source import Frame, FrameSource, _FrameQueue


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


class _FakeCap:
    def __init__(self):
        self.props = {}

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)


def test_capture_fps_requested_from_local_cameras_only():
    src = FrameSource.__new__(FrameSource)
    src.capture_fps, src.spec = 60, "0"
    for kind, asked in (("camera", True), ("url", False), ("android", False)):
        src.kind, cap = kind, _FakeCap()
        src._request_fps(cap)
        assert (cap.props.get(cv2.CAP_PROP_FPS) == 60.0) is asked
        if asked:
            assert cap.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
