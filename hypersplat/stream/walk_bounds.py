"""Ground-level walk bounds for the live viewer.

A splat only renders well near where it was photographed: far from the training cameras the
viewer flies into floaters, needles and empty space. Walk mode keeps the viewer camera inside
the footprint of the registered cameras, at the height they were captured from.

    compute_walk_bounds  camera poses (trainer frame, +Z up) -> XY grid of walkable cells +
                         per-cell eye height (Gaussian-weighted camera z)
    WalkBoundsPublisher  sends the grid to every viser client (SetWalkBoundsMessage); the
                         patched viser client (patches/viser_walk_client.patch) moves the camera
                         on the ground plane and clamps it to the grid every frame

The stock viser client does not know SetWalkBoundsMessage, so the publisher only sends it when
the walk build is installed (scripts/viewer/build_walk_client.sh writes WALK_MARKER).
"""

import dataclasses
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

WALK_MARKER = ".hypersplat_walk"
MAX_CELLS = 256


@dataclasses.dataclass
class WalkConfig:
    radius_mult: float = 3.0      # walkable radius around each camera, in camera spacings
    speed_mult: float = 4.0       # walk speed, camera spacings per second
    pitch_deg: float = 35.0       # max look up/down from horizontal
    near_mult: float = 0.3        # viewer near plane in walk mode, in camera spacings
    bridge_mult: float = 4.0      # join consecutive cameras closer than this many spacings


@dataclasses.dataclass
class WalkBounds:
    origin: Tuple[float, float]   # world XY of cell (0, 0) centre
    cell: float
    mask: np.ndarray              # [H, W] bool, row = y, col = x
    heights: np.ndarray           # [H, W] float32 eye height (world z)
    spacing: float                # median nearest-neighbour camera distance
    speed: float
    pitch_deg: float
    near: float
    start_position: Tuple[float, float, float]
    start_look_at: Tuple[float, float, float]
    num_views: int

    def contains(self, xy: np.ndarray) -> np.ndarray:
        """Walkable test for world points [..., 2+]."""
        xy = np.asarray(xy, np.float64)
        ij = np.round((xy[..., :2] - np.asarray(self.origin)) / self.cell).astype(np.int64)
        h, w = self.mask.shape
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < w) & (ij[..., 1] >= 0) & (ij[..., 1] < h)
        out = np.zeros(ok.shape, bool)
        out[ok] = self.mask[ij[ok][:, 1], ij[ok][:, 0]]
        return out

    def save_npz(self, path) -> None:
        d = dataclasses.asdict(self)
        np.savez(path, **{k: np.asarray(v) for k, v in d.items()})

    @classmethod
    def load_npz(cls, path) -> "WalkBounds":
        z = np.load(path)
        kw = {}
        for f in dataclasses.fields(cls):
            v = z[f.name]
            if f.name in ("mask", "heights"):
                kw[f.name] = v
            elif f.name == "num_views":
                kw[f.name] = int(v)
            elif v.ndim:
                kw[f.name] = tuple(float(x) for x in v)
            else:
                kw[f.name] = float(v)
        return cls(**kw)


def _spacing(centers: np.ndarray, scene_scale: float) -> float:
    if len(centers) < 3:
        return 0.05 * scene_scale
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    nn = nn[np.isfinite(nn) & (nn > 1e-6)]
    if not len(nn):
        return 0.05 * scene_scale
    # Keyframes can be very close (slow walking); never let the corridor collapse
    return float(max(np.median(nn), 0.01 * scene_scale))


def _segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p [M, 2] to segments a->b [S, 2]: [M, S]."""
    ab = b - a
    t = np.einsum("msk,sk->ms", p[:, None, :] - a[None], ab) / np.maximum((ab * ab).sum(-1), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - q, axis=-1)


def compute_walk_bounds(camtoworlds: np.ndarray, scene_scale: float,
                        cfg: Optional[WalkConfig] = None) -> Optional[WalkBounds]:
    """Walkable XY grid + eye heights from camera poses in the +Z-up trainer frame.

    Cameras are taken in registration order: consecutive cameras closer than
    `bridge_mult` spacings are joined so the corridor stays continuous between keyframes,
    while a relocalization jump is not bridged across unseen space.
    """
    cfg = cfg or WalkConfig()
    c2w = np.asarray(camtoworlds, np.float64).reshape(-1, 4, 4)
    if not len(c2w):
        return None
    centers = c2w[:, :3, 3]
    s = _spacing(centers, scene_scale)
    r = cfg.radius_mult * s

    lo = centers[:, :2].min(0) - r
    hi = centers[:, :2].max(0) + r
    cell = max(s / 2.0, float((hi - lo).max()) / (MAX_CELLS - 1))
    w = int(np.ceil((hi[0] - lo[0]) / cell)) + 1
    h = int(np.ceil((hi[1] - lo[1]) / cell)) + 1
    gx = lo[0] + cell * np.arange(w)
    gy = lo[1] + cell * np.arange(h)
    pts = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)  # row-major: [y, x]

    xy = centers[:, :2]
    d_cam = np.linalg.norm(pts[:, None, :] - xy[None], axis=-1)  # [M, N]
    dist = d_cam.min(1)
    if len(xy) > 1:
        step = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
        keep = step < cfg.bridge_mult * s
        if keep.any():
            dist = np.minimum(dist, _segment_dist(pts, xy[:-1][keep], xy[1:][keep]).min(1))
    mask = (dist <= r).reshape(h, w)

    # Eye height: Gaussian-weighted camera z, shifted by the nearest distance so far cells
    # don't underflow to 0/0
    sigma = max(r, 1e-6)
    d2 = d_cam ** 2
    wts = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2.0 * sigma ** 2))
    heights = ((wts * centers[None, :, 2]).sum(1) / wts.sum(1)).reshape(h, w).astype(np.float32)

    # Start at the first camera, looking where it looked but levelled to the horizon
    fwd = c2w[0, :3, 2].copy()  # OpenCV: +Z is the viewing direction
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    p0 = centers[0]
    look = p0 + fwd * max(s, 1e-3)

    return WalkBounds(
        origin=(float(lo[0]), float(lo[1])), cell=float(cell), mask=mask, heights=heights,
        spacing=s, speed=cfg.speed_mult * s, pitch_deg=cfg.pitch_deg, near=cfg.near_mult * s,
        start_position=tuple(float(v) for v in p0), start_look_at=tuple(float(v) for v in look),
        num_views=len(c2w),
    )


def walk_client_installed() -> bool:
    try:
        import viser
    except ImportError:
        return False
    return (Path(viser.__file__).parent / "client" / "build" / WALK_MARKER).exists()


def _message_class():
    """SetWalkBoundsMessage, defined lazily so importing this module never needs viser."""
    global _MSG
    if _MSG is not None:
        return _MSG
    from viser._messages import Message

    @dataclasses.dataclass
    class SetWalkBoundsMessage(Message, include_in_scene_serialization=False):
        """Server -> client: walk-mode bounds (patched client only)."""
        enabled: bool
        version: int
        origin: Tuple[float, float]
        cell: float
        width: int
        height: int
        mask: npt.NDArray[np.uint8]
        heights: npt.NDArray[np.float32]
        speed: float
        pitch_deg: float

    _MSG = SetWalkBoundsMessage
    return _MSG


_MSG = None


class WalkBoundsPublisher:
    """Keeps every viser client's walk bounds in sync with the registered cameras."""

    def __init__(self, server, cfg: Optional[WalkConfig] = None, min_interval: float = 2.0):
        self.server = server
        self.cfg = cfg or WalkConfig()
        self.min_interval = min_interval
        self.bounds: Optional[WalkBounds] = None
        self.enabled = True
        self.version = 0
        self._key = None
        self._last = 0.0
        self._lock = threading.Lock()
        self._placed = False
        self.client_ok = walk_client_installed()
        if not self.client_ok:
            logger.warning("Walk mode: patched viser client not installed "
                           "(scripts/viewer/build_walk_client.sh); free-fly navigation only")
        with server.gui.add_folder("Walk"):
            self._toggle = server.gui.add_checkbox(
                "Walk mode", initial_value=self.client_ok, disabled=not self.client_ok,
                hint="Keep the camera at capture height inside the region the cameras covered.")
            self._info = server.gui.add_markdown("_waiting for cameras_")

        @self._toggle.on_update
        def _(_) -> None:
            self.enabled = bool(self._toggle.value)
            self._broadcast()

        server.on_client_connect(self._on_connect)

    @property
    def active(self) -> bool:
        return self.client_ok and self.enabled and self.bounds is not None

    def near_plane(self, default: float) -> float:
        return max(default, self.bounds.near) if self.active else default

    def update(self, camtoworlds: np.ndarray, scene_scale: float, key=None, force: bool = False) -> bool:
        """Recompute when `key` (e.g. views registered, map version) changed; throttled."""
        now = time.time()
        if not force and (key == self._key or now - self._last < self.min_interval):
            return False
        b = compute_walk_bounds(camtoworlds, scene_scale, self.cfg)
        if b is None:
            return False
        self._key, self._last = key, now
        self.publish(b)
        return True

    def publish(self, b: WalkBounds) -> None:
        """Send precomputed bounds (e.g. WalkBounds.load_npz of a finished run)."""
        with self._lock:
            self.bounds = b
            self.version += 1
        self._info.content = (f"WASD walk · drag to look · Shift run  \n"
                              f"{b.num_views} cameras · speed {b.speed:.2f}/s")
        if not self._placed:
            self._placed = True
            ic = self.server.initial_camera
            ic.up = (0.0, 0.0, 1.0)
            ic.position = b.start_position
            ic.look_at = b.start_look_at
        self._broadcast()

    def _message(self):
        b = self.bounds
        h, w = b.mask.shape
        return _message_class()(
            enabled=self.enabled, version=self.version, origin=b.origin, cell=b.cell, width=w,
            height=h, mask=np.ascontiguousarray(b.mask, np.uint8).reshape(-1),
            heights=np.ascontiguousarray(b.heights, np.float32).reshape(-1),
            speed=b.speed, pitch_deg=b.pitch_deg)

    def _broadcast(self):
        if not self.client_ok or self.bounds is None:
            return
        msg = self._message()
        for client in self.server.get_clients().values():
            client._websock_connection.queue_message(msg)

    def _on_connect(self, client) -> None:
        if not self.client_ok or self.bounds is None:
            return
        client._websock_connection.queue_message(self._message())
