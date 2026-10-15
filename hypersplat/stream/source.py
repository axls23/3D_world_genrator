"""Frame sources: cameras, network streams, video files and Arrow Flight producers.

Every source yields ``Frame``s (RGB, at most ``max_side`` px on the long side), so the rest of
the pipeline is the same for just-in-time reconstruction of a recorded video and for a live
camera. Spec strings:

    path/to/video.mp4            file; pace "fast" (every frame, as fast as it is consumed)
    0, 1, /dev/video0            local camera (live)
    rtsp://, http(s)://, udp://  network camera / stream (live)
    flight://host:port           external camera pushing JPEG frames over Arrow Flight
                                 (python -m hypersplat.stream.flight push <camera> <url>)
    android://[SERIAL]?camera=back
                                 Android phone camera over the USB cable via adb (USB debugging;
                                 no app, no root - see android.py)

A local camera or phone that drops off USB mid-run is waited for (``reconnect_timeout``) and
reopened; ``status`` says what the source is waiting for meanwhile.

Live sources buffer up to ``buffer_seconds`` of frames in a FIFO filled by a grabber thread, so a
busy moment in the pipeline does not lose frames (and their keyframes); only a backlog longer than
the buffer drops the oldest frames. Cameras are asked for ``capture_fps`` (default 60) when given;
``newest`` is the latest grabbed frame, for previews that should not trail the buffer. A file can
be paced "realtime" to rehearse the live path on a recording (frames are skipped exactly like a
slow camera).
"""

import logging
import os
from collections import deque
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LIVE_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


@dataclass
class Frame:
    index: int          # frame number in the source
    t: float            # seconds since the source started (media time for files)
    image: np.ndarray   # RGB uint8 (H, W, 3)


def parse_source(spec: Union[str, int]) -> Tuple[str, Union[str, int]]:
    """(kind, target): kind is file | camera | url | flight | android."""
    s = str(spec).strip()
    if s.startswith("android://"):
        return "android", s
    if s.isdigit():
        return "camera", int(s)
    if s.startswith("/dev/video"):
        return "camera", s
    if s.startswith("flight://") or s.startswith("grpc://"):
        return "flight", s
    if s.startswith(LIVE_SCHEMES):
        return "url", s
    if Path(s).is_file():
        return "file", str(Path(s).resolve())
    raise FileNotFoundError(f"Stream source {s!r} is not a file, camera index, /dev/video* or URL")


def fit_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def decode_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("undecodable JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


class _LatestSlot:
    """Single-frame mailbox: put() overwrites, get() waits for a frame newer than the last."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._closed = False
        self.dropped = 0

    def put(self, frame: Frame):
        with self._cond:
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def get(self, timeout: Optional[float] = None) -> Optional[Frame]:
        with self._cond:
            end = None if timeout is None else time.time() + timeout
            while self._frame is None and not self._closed:
                remaining = None if end is None else end - time.time()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            frame, self._frame = self._frame, None
            return frame


class _FrameQueue:
    """Bounded FIFO mailbox: put() appends (dropping the oldest when full), get() takes the oldest."""

    def __init__(self, maxlen: int):
        self._cond = threading.Condition()
        self._frames = deque(maxlen=max(1, int(maxlen)))
        self._closed = False
        self.dropped = 0

    def put(self, frame: Frame):
        with self._cond:
            if len(self._frames) == self._frames.maxlen:
                self.dropped += 1
            self._frames.append(frame)
            self._cond.notify_all()

    def resize(self, maxlen: int):
        """New capacity (the source rate changed on a reconnect); shrinking drops the oldest."""
        with self._cond:
            maxlen = max(1, int(maxlen))
            self.dropped += max(0, len(self._frames) - maxlen)
            self._frames = deque(self._frames, maxlen=maxlen)

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def get(self, timeout: Optional[float] = None) -> Optional[Frame]:
        with self._cond:
            end = None if timeout is None else time.time() + timeout
            while not self._frames and not self._closed:
                remaining = None if end is None else end - time.time()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._frames.popleft() if self._frames else None


class FrameSource:
    """Iterate frames from a file, camera or network stream (see module docstring)."""

    def __init__(self, spec: Union[str, int], max_side: int = 960, pace: str = "auto",
                 reconnect_timeout: float = 300.0, capture_fps: Optional[float] = None,
                 buffer_seconds: float = 1.0):
        self.kind, self.target = parse_source(spec)
        if self.kind == "flight":
            raise ValueError("flight:// sources are served by hypersplat.stream.flight.FrameFlightServer")
        self.spec = str(spec)
        self.live = self.kind != "file"
        if pace == "auto":
            pace = "realtime" if self.live else "fast"
        if pace not in ("realtime", "fast"):
            raise ValueError(f"pace must be auto, realtime or fast, not {pace!r}")
        if self.live and pace == "fast":
            pace = "realtime"  # a camera cannot be read faster than it produces frames
        self.pace = pace
        self.max_side = max_side
        self.capture_fps = capture_fps
        self.buffer_seconds = buffer_seconds
        self.reconnect_timeout = reconnect_timeout
        self.reconnects = 0
        self.status: Optional[str] = None   # e.g. "phone disconnected: ..." while reconnecting
        self.newest: Optional[Frame] = None  # latest grabbed frame (live), ahead of the FIFO
        self._capture_rate: Optional[float] = None  # camera rate negotiated by _request_fps
        self.android = None
        if self.kind == "android":
            from .android import AndroidCamera
            self.android = AndroidCamera(self.spec)
            # Raw H.264 over the adb tunnel: open with low-latency demuxing
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay")
            self.cap = cv2.VideoCapture(self.android.start(), cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.target)
        if not self.cap.isOpened():
            if self.android is not None:
                self.android.close()
            raise RuntimeError(f"Could not open stream source {self.spec}")
        if self.live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # no stale driver-side queue
            self._capture_rate = self._request_fps(self.cap)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(self.fps) or self.fps <= 0 or self.fps > 240:
            self.fps = 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not self.live else 0
        self.frames_read = 0
        self.frames_skipped = 0
        self._stop = threading.Event()
        self._slot: Optional[_LatestSlot] = None
        self._grabber: Optional[threading.Thread] = None

    @property
    def frames_dropped(self) -> int:
        return self.frames_skipped + (self._slot.dropped if self._slot else 0)

    def _prepare(self, bgr: np.ndarray) -> np.ndarray:
        return fit_max_side(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.max_side)

    # ---------------------------------------------------------------- live
    def _reopen(self) -> bool:
        """The camera / phone stopped delivering frames (USB reset, cable, phone locked):
        wait until it is back and reopen it. False when it did not come back in time."""
        self.cap.release()
        end = time.time() + self.reconnect_timeout
        what = "phone" if self.kind == "android" else "camera"
        logger.warning(f"[Source] {what} {self.spec} stopped sending frames; waiting up to "
                       f"{self.reconnect_timeout:.0f} s for it to come back")
        while not self._stop.is_set() and time.time() < end:
            try:
                if self.kind == "android":
                    state = self.android.device_state()
                    if state != "device":
                        self.status = (f"phone disconnected ({state}): check the USB cable"
                                       + ("; unlock the phone and tap Allow on 'Allow USB debugging?'"
                                          if state == "unauthorized" else ""))
                        self._stop.wait(1.0)
                        continue
                    self.status = "phone back; restarting its camera"
                    cap = cv2.VideoCapture(self.android.restart(), cv2.CAP_FFMPEG)
                else:
                    self.status = f"{what} disconnected: check the USB cable"
                    cap = cv2.VideoCapture(self.target)
                rate = None
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    rate = self._request_fps(cap)  # before the check, so a broken format switch is caught
                if cap.isOpened() and cap.read()[0]:
                    self.cap, self._capture_rate = cap, rate
                    self.reconnects += 1
                    self.status = None
                    logger.info(f"[Source] {what} {self.spec} reconnected")
                    return True
                cap.release()
            except Exception as e:  # phone vanished again mid-restart, camera busy, ...
                self.status = f"{what} reconnect failed ({e}); retrying"
            self._stop.wait(min(2.0, max(0.0, end - time.time())))
        self.status = f"{what} did not come back"
        return False

    def _request_fps(self, cap) -> Optional[float]:
        """Ask a local camera for ``capture_fps``. UVC cameras mostly reach 60 fps only in MJPG.
        Network streams and the phone (android://...&fps=) set their rate at the sender.
        Returns the rate the camera runs at, or None when the request was not kept."""
        if not self.capture_fps or self.kind != "camera":
            return None
        fourcc, fps = cap.get(cv2.CAP_PROP_FOURCC), cap.get(cv2.CAP_PROP_FPS)
        # Switch format only when the current one is known, so a failed switch can be undone
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, float(self.capture_fps))
        if cap.read()[0]:
            got = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"[Source] camera {self.spec}: asked for {self.capture_fps:g} fps, got {got:g}")
            # V4L2 reports the negotiated rate, which can be lower than asked
            return float(min(got, self.capture_fps)) if 0 < got <= 240 else float(self.capture_fps)
        # Not every camera speaks MJPG (some UVC, virtual, v4l2loopback): one that stops
        # delivering frames after the switch gets its own format and rate back
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, int(fourcc))
        if fps:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        logger.warning(f"[Source] camera {self.spec}: no frames at {self.capture_fps:g} fps"
                       f"{' in MJPG' if fourcc else ''}; restored its own format ({fps:g} fps)")
        return None

    def _live_rate(self) -> float:
        """Frames per second the FIFO is sized for: the requested rate when it is known (OpenCV
        reports ~25-30 for the phone's raw H.264 whatever the sender runs at), else the reported."""
        if self.kind == "camera" and self._capture_rate:
            return self._capture_rate
        if self.kind == "android" and self.android is not None:
            return float(self.android.opts["fps"])
        return self.fps

    def _grab_loop(self):
        t0 = time.time()
        index = 0
        while not self._stop.is_set():
            ok, bgr = self.cap.read()
            if not ok:
                if self.kind in ("url", "android"):  # transient hiccup: retry once
                    time.sleep(0.2)
                    if self.cap.grab():
                        continue
                if self.kind in ("camera", "android") and not self._stop.is_set() and self._reopen():
                    # The reopened camera may run at another rate (MJPG kept or not this time)
                    self._slot.resize(round(self._live_rate() * self.buffer_seconds))
                    continue
                break
            frame = Frame(index, time.time() - t0, self._prepare(bgr))
            self.newest = frame  # set before the FIFO, so it is never older than an ingested frame
            self._slot.put(frame)
            self.frames_read += 1
            index += 1
        self._slot.close()

    def _iter_live(self) -> Iterator[Frame]:
        self._slot = _FrameQueue(round(self._live_rate() * self.buffer_seconds))
        self._grabber = threading.Thread(target=self._grab_loop, name="frame-grabber", daemon=True)
        self._grabber.start()
        while not self._stop.is_set():
            frame = self._slot.get()
            if frame is None:
                return
            yield frame

    # ---------------------------------------------------------------- file
    def _iter_file(self) -> Iterator[Frame]:
        index = 0
        t_start = time.time()
        while not self._stop.is_set():
            if self.pace == "realtime":
                # Skip (grab without decoding) the frames a camera would have dropped
                due = int((time.time() - t_start) * self.fps)
                while index < due:
                    if not self.cap.grab():
                        return
                    index += 1
                    self.frames_skipped += 1
                wait = t_start + index / self.fps - time.time()
                if wait > 0:
                    time.sleep(wait)
            ok, bgr = self.cap.read()
            if not ok:
                return
            self.frames_read += 1
            yield Frame(index, index / self.fps, self._prepare(bgr))
            index += 1

    def __iter__(self) -> Iterator[Frame]:
        try:
            yield from (self._iter_live() if self.live else self._iter_file())
        finally:
            self.close()

    def close(self):
        self._stop.set()
        if self._grabber is not None and self._grabber is not threading.current_thread():
            self._grabber.join(timeout=2.0)
        self.cap.release()
        if self.android is not None:
            self.android.close()
            self.android = None


class QueueFrameSource:
    """Frames pushed by another thread (e.g. an Arrow Flight do_put handler). Live semantics:
    only the newest frame is kept when the consumer is busy."""

    live = True
    pace = "realtime"
    kind = "flight"

    def __init__(self, spec: str, max_side: int = 960):
        self.spec = spec
        self.max_side = max_side
        self.fps = 0.0
        self.frame_count = 0
        self.frames_read = 0
        self._slot = _LatestSlot()
        self._index = 0

    @property
    def frames_dropped(self) -> int:
        return self._slot.dropped

    def push(self, t: float, image_rgb: np.ndarray):
        self._slot.put(Frame(self._index, float(t), fit_max_side(image_rgb, self.max_side)))
        self._index += 1
        self.frames_read += 1

    def end(self):
        self._slot.close()

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self._slot.get()
            if frame is None:
                return
            yield frame

    def close(self):
        self._slot.close()
