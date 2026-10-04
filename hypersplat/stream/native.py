"""ctypes bindings for csrc/streamops.c, built on first use.

The library is compiled with the system C compiler into csrc/_build/streamops-<hash>.so (the
hash is of the source, so an edit rebuilds it). Without a compiler every function falls back
to a vectorized numpy implementation with identical results; HYPERSPLAT_NO_NATIVE=1 forces it.
"""

import ctypes
import hashlib
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CSRC = Path(__file__).resolve().parent / "csrc"
SOURCE = CSRC / "streamops.c"
CFLAGS = ["-O3", "-std=c99", "-fPIC", "-shared", "-fno-math-errno"]

_lib = None
_lock = threading.Lock()
_tried = False


def _build() -> Optional[Path]:
    digest = hashlib.sha1(SOURCE.read_bytes() + " ".join(CFLAGS).encode()).hexdigest()[:12]
    out = CSRC / "_build" / f"streamops-{digest}.so"
    if out.exists():
        return out
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    try:
        subprocess.run([cc, *CFLAGS, str(SOURCE), "-o", str(tmp), "-lm"], check=True,
                       capture_output=True, text=True)
        os.replace(tmp, out)  # atomic: concurrent builders race harmlessly
        for old in out.parent.glob("streamops-*.so"):
            if old != out:
                old.unlink(missing_ok=True)
        return out
    except (subprocess.CalledProcessError, OSError) as e:
        logger.warning(f"[native] build failed, using numpy fallbacks: {getattr(e, 'stderr', e)}")
        tmp.unlink(missing_ok=True)
        return None


def lib():
    """The loaded library, or None (numpy fallbacks)."""
    global _lib, _tried
    if _tried:
        return _lib
    with _lock:
        if _tried:
            return _lib
        _tried = True
        if os.environ.get("HYPERSPLAT_NO_NATIVE"):
            return None
        path = _build()
        if path is None:
            return None
        L = ctypes.CDLL(str(path))
        dp, i64, u8p = ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint8)
        L.pose_nn_distance.argtypes = [dp, i64, u8p, ctypes.c_double, ctypes.c_double, dp]
        L.pose_nn_distance.restype = None
        L.weighted_pick.argtypes = [dp, i64, ctypes.c_double]
        L.weighted_pick.restype = i64
        _lib = L
        return _lib


def _dptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def pose_nn_distance(c2w: np.ndarray, query: np.ndarray, spacing: float,
                     angle_unit_deg: float) -> np.ndarray:
    """(N,) nearest-neighbour pose distance for query[i], NaN elsewhere (see streamops.c)."""
    c2w = np.ascontiguousarray(c2w, dtype=np.float64).reshape(-1, 16)
    query = np.ascontiguousarray(query, dtype=np.uint8)
    n = len(c2w)
    out = np.empty(n, dtype=np.float64)
    L = lib()
    if L is not None:
        L.pose_nn_distance(_dptr(c2w), n, query.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                           float(spacing), float(angle_unit_deg), _dptr(out))
        return out
    m = c2w.reshape(-1, 4, 4)
    centers, R = m[:, :3, 3], m[:, :3, :3]
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1) / max(spacing, 1e-12)
    tr = np.einsum("irk,jrk->ij", R, R)
    d = d + np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))) / max(angle_unit_deg, 1e-12)
    np.fill_diagonal(d, np.inf)
    out[:] = np.nan
    q = query.astype(bool)
    if n > 1:
        out[q] = d[q].min(axis=1)
    else:
        out[q] = np.inf
    return out


def weighted_pick(weights: np.ndarray, u: float) -> int:
    """Index drawn with probability proportional to weights (u uniform in [0, 1))."""
    w = np.ascontiguousarray(weights, dtype=np.float64)
    L = lib()
    if L is not None:
        return int(L.weighted_pick(_dptr(w), len(w), float(u)))
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        return -1
    return int(min(np.searchsorted(np.cumsum(w), u * total, side="right"), len(w) - 1))
