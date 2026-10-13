"""Device-resident stream state, updated by CUDA kernels inside the trainer's step stream.

DevicePoseTable holds every keyframe's pose state on the GPU (slot per live keyframe):
filtered ACE-frame pose, trainer-frame c2w, stability tier, map version, confidence. ACE
DELTA records (hypersplat/stream/ring.py) are applied by ``apply_deltas`` (csrc/
stream_kernels.cu): stale guard, registration, camera move, SO(3) x R^3 pose filter, tier FSM
and the trainer-frame transform in one launch, no host round trip. Per-record results (move,
outcome, tier) come back through pinned memory and ``poll()`` never blocks: the host's
jitter statistics, slot resets and logs run one step behind the GPU.

apply_deltas_reference is the same computation in numpy (tests, and the definition of what
the kernel must do).
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..ring import DELTA_DTYPE

logger = logging.getLogger(__name__)

CSRC = Path(__file__).resolve().parent / "csrc"
NEW, PROVISIONAL, STABLE, ANCHORED = 0, 1, 2, 3
APPLIED, STALE, UNREGISTERED, FROZEN, OUTLIER, BAD_SLOT = 0, 1, 2, 3, 4, 5

_ext = None
_ext_lock = threading.Lock()


TOOLCHAIN = Path(__file__).resolve().parents[3].parent / ".conda_envs" / "cuda124-toolchain"


def ext():
    """The compiled extension (JIT-built on first use, like gsplat's).

    Built with the repo's CUDA 12.4 toolchain env when present (HYPERSPLAT_CUDA_HOME
    overrides): torch 2.6 is a cu124 build, and its headers do not get through CUDA 13's
    front end (template errors in ATen/core/List_inl.h whatever the host compiler), nor through
    GCC >= 16 as the host compiler; the env's nvcc 12.4 + GCC 12 build them."""
    global _ext
    if _ext is None:
        with _ext_lock:
            if _ext is None:
                import torch.utils.cpp_extension as cpp

                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "%d.%d" % torch.cuda.get_device_capability())
                home = Path(os.environ.get("HYPERSPLAT_CUDA_HOME") or TOOLCHAIN)
                cuda_flags, ld_flags = ["-O3"], []
                saved = (cpp.CUDA_HOME, os.environ.get("PYTORCH_NVCC"))
                if (home / "bin" / "nvcc").exists():
                    cpp.CUDA_HOME = str(home)
                    os.environ["PYTORCH_NVCC"] = str(home / "bin" / "nvcc")
                    gxx = home / "bin" / "x86_64-conda-linux-gnu-g++"
                    if gxx.exists():
                        cuda_flags += ["-ccbin", str(gxx)]
                    ld_flags = ["-L%s" % (home / "lib")]
                try:
                    _ext = cpp.load(name="hypersplat_stream_cuda", sources=[str(CSRC / "stream_kernels.cu")],
                                    extra_cuda_cflags=cuda_flags, extra_cflags=["-O3"],
                                    extra_ldflags=ld_flags, verbose=False)
                finally:
                    cpp.CUDA_HOME = saved[0]
                    if saved[1] is None:
                        os.environ.pop("PYTORCH_NVCC", None)
                    else:
                        os.environ["PYTORCH_NVCC"] = saved[1]
    return _ext


class DevicePoseTable:
    def __init__(self, capacity: int = 512, device="cuda"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        z = dict(device=self.device)
        self.has_pose = torch.zeros(capacity, dtype=torch.uint8, **z)
        self.tier = torch.zeros(capacity, dtype=torch.uint8, **z)
        self.stable_cnt = torch.zeros(capacity, dtype=torch.uint8, **z)
        self.ver = torch.zeros(capacity, dtype=torch.int32, **z)
        self.conf = torch.zeros(capacity, dtype=torch.float32, **z)
        self.c2w_f = torch.zeros(capacity, 16, dtype=torch.float64, **z)       # filtered, ACE frame
        self.c2w_trainer = torch.zeros(capacity, 16, dtype=torch.float32, **z)  # normalized frame
        self._slot: Dict[int, int] = {}
        self._free: List[int] = list(range(capacity - 1, -1, -1))
        self._stage = torch.empty(0, dtype=torch.uint8).pin_memory()
        self._pending: List[Tuple[torch.cuda.Event, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.M = torch.eye(4, dtype=torch.float64, device=self.device).reshape(16)

    # ------------------------------------------------------------------ slots (host)
    def slot(self, kid: int) -> int:
        """Slot of a keyframe, allocated on first sight; -1 when the table is full."""
        s = self._slot.get(kid)
        if s is None:
            if not self._free:
                return -1
            s = self._free.pop()
            self._slot[kid] = s
        return s

    def slot_of(self, kid: int) -> int:
        return self._slot.get(kid, -1)

    def release(self, kid: int):
        """Keyframe evicted: its slot is cleared (on the stream) and reused."""
        s = self._slot.pop(kid, None)
        if s is None:
            return
        for t in (self.has_pose, self.tier, self.stable_cnt, self.ver):
            t[s] = 0
        self._free.append(s)

    def set_transform(self, M: np.ndarray):
        """ACE frame -> trainer frame similarity (the session's frozen normalization)."""
        self.M.copy_(torch.as_tensor(np.asarray(M, np.float64).reshape(16)), non_blocking=False)

    # ------------------------------------------------------------------ apply (stream)
    def apply(self, records: np.ndarray, *, conf_threshold: float, extent: float, alpha: float = 1.0,
              promote_thr: float = 0.01, outlier_thr: float = float("inf"), stable_after: int = 2) -> int:
        """Queue DELTA records on the current stream. Returns how many were launched."""
        recs = np.ascontiguousarray(records, dtype=DELTA_DTYPE)
        k = len(recs)
        if k == 0:
            return 0
        slots = np.array([self.slot(int(kid)) for kid in recs["kid"]], dtype=np.int32)
        # Launch rounds of distinct slots, in record order: a slot's records apply in sequence
        rounds, seen, cur = [], set(), []
        for i, s in enumerate(slots):
            if s in seen:
                rounds.append(cur)
                cur, seen = [], set()
            cur.append(i)
            seen.add(int(s))
        rounds.append(cur)
        order = np.concatenate([np.asarray(r, np.int64) for r in rounds])
        nbytes = k * DELTA_DTYPE.itemsize + k * 4
        if self._stage.numel() < nbytes:
            self._stage = torch.empty(max(nbytes, 2 * self._stage.numel()), dtype=torch.uint8).pin_memory()
        stage = self._stage[:nbytes].numpy()
        stage[: k * DELTA_DTYPE.itemsize] = recs[order].view(np.uint8)
        stage[k * DELTA_DTYPE.itemsize:] = slots[order].view(np.uint8)
        dev = self._stage[:nbytes].to(self.device, non_blocking=True)
        d_recs = dev[: k * DELTA_DTYPE.itemsize].view(k, DELTA_DTYPE.itemsize)
        d_slots = dev[k * DELTA_DTYPE.itemsize:].view(torch.int32)
        move = torch.empty(k, dtype=torch.float32, device=self.device)
        result = torch.empty(k, dtype=torch.int8, device=self.device)
        tier = torch.empty(k, dtype=torch.uint8, device=self.device)
        E, a = ext(), 0
        for r in rounds:
            b = a + len(r)
            E.apply_deltas(d_recs[a:b], d_slots[a:b], self.M, self.has_pose, self.tier, self.stable_cnt,
                           self.ver, self.conf, self.c2w_f, self.c2w_trainer, move[a:b], result[a:b],
                           tier[a:b], float(conf_threshold), float(extent), float(alpha),
                           float(promote_thr), float(outlier_thr), int(stable_after))
            a = b
        # The pinned stage is reused by the next apply(): wait for this copy before rewriting it
        self._stage_ready = torch.cuda.Event()
        self._stage_ready.record()
        h_move = torch.empty(k, dtype=torch.float32).pin_memory()
        h_result = torch.empty(k, dtype=torch.int8).pin_memory()
        h_tier = torch.empty(k, dtype=torch.uint8).pin_memory()
        h_move.copy_(move, non_blocking=True)
        h_result.copy_(result, non_blocking=True)
        h_tier.copy_(tier, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        self._pending.append((done, recs["kid"][order].copy(), h_move, h_result, h_tier))
        return k

    def poll(self, wait: bool = False) -> Optional[Dict[str, np.ndarray]]:
        """Results of finished applies (kid, move, result, tier), oldest first; None if none.
        Never blocks unless wait=True."""
        out = []
        while self._pending:
            done = self._pending[0][0]
            if not (wait or done.query()):
                break
            if wait:
                done.synchronize()
            _, kids, m, r, t = self._pending.pop(0)
            out.append((kids, m.numpy().copy(), r.numpy().copy(), t.numpy().copy()))
        if not out:
            return None
        return {"kid": np.concatenate([o[0] for o in out]), "move": np.concatenate([o[1] for o in out]),
                "result": np.concatenate([o[2] for o in out]), "tier": np.concatenate([o[3] for o in out])}

    def ensure_stage_free(self):
        ev = getattr(self, "_stage_ready", None)
        if ev is not None:
            ev.synchronize()


# ====================================================================== numpy reference
def _so3_log(R):
    c = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * v
    return th / (2.0 * np.sin(th)) * v


def _so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        a, b = 1.0, 0.5
    else:
        a, b = np.sin(th) / th, (1.0 - np.cos(th)) / th ** 2
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + a * K + b * K @ K


class ReferenceState:
    def __init__(self, capacity: int):
        self.has_pose = np.zeros(capacity, np.uint8)
        self.tier = np.zeros(capacity, np.uint8)
        self.stable_cnt = np.zeros(capacity, np.uint8)
        self.ver = np.zeros(capacity, np.int32)
        self.conf = np.zeros(capacity, np.float32)
        self.c2w_f = np.zeros((capacity, 4, 4))
        self.c2w_trainer = np.zeros((capacity, 4, 4), np.float32)


def apply_deltas_reference(st: ReferenceState, recs: np.ndarray, slots: np.ndarray, M: np.ndarray, *,
                           conf_threshold, extent, alpha=1.0, promote_thr=0.01, outlier_thr=np.inf,
                           stable_after=2):
    """What apply_deltas_kernel computes, record by record (sequential)."""
    out = []
    for r, s in zip(recs, slots):
        if r["map_version"] < st.ver[s]:
            out.append((np.nan, STALE, st.tier[s]))
            continue
        st.ver[s] = r["map_version"]
        w2c = r["w2c"].reshape(4, 4)
        if not np.isfinite(w2c).all() or r["conf"] < conf_threshold:
            st.conf[s], st.has_pose[s], st.tier[s], st.stable_cnt[s] = r["conf"], 0, NEW, 0
            out.append((np.nan, UNREGISTERED, NEW))
            continue
        if st.tier[s] == ANCHORED and st.has_pose[s]:
            out.append((np.nan, FROZEN, ANCHORED))
            continue
        Rm = w2c[:3, :3].T
        cm = -Rm @ w2c[:3, 3]
        result, move = APPLIED, np.nan
        if st.has_pose[s]:
            Rf, cf = st.c2w_f[s, :3, :3].copy(), st.c2w_f[s, :3, 3].copy()
            move = np.linalg.norm(cm - cf) / max(extent, 1e-12)
            if move > outlier_thr:
                result, Rf, cf = OUTLIER, Rm, cm
            else:
                Rf = Rf @ _so3_exp(alpha * _so3_log(Rf.T @ Rm))
                cf = cf + alpha * (cm - cf)
        else:
            Rf, cf = Rm, cm
        t, sc = int(st.tier[s]), int(st.stable_cnt[s])
        if r["anchored"]:
            t = ANCHORED
        elif not st.has_pose[s]:
            t, sc = PROVISIONAL, 0
        elif result == OUTLIER:
            sc = 0
            t = PROVISIONAL if t == STABLE else t
        elif move <= promote_thr:
            sc = min(sc + 1, 255)
            if sc >= stable_after and t < STABLE:
                t = STABLE
        else:
            sc = 0
        st.tier[s], st.stable_cnt[s], st.has_pose[s], st.conf[s] = t, sc, 1, r["conf"]
        st.c2w_f[s] = np.eye(4)
        st.c2w_f[s, :3, :3], st.c2w_f[s, :3, 3] = Rf, cf
        T = M @ st.c2w_f[s]
        T[:3, :3] /= np.linalg.norm(T[0, :3])
        st.c2w_trainer[s] = T.astype(np.float32)
        out.append((move, result, t))
    return out


class JitterFloor:
    """Running robust statistics of relocalization moves -> apply_deltas thresholds.

    Continuous ACE relocalizes registered frames over and over; their camera moves (fraction
    of the path extent) are mostly jitter (drone / Kenilworth: median 1.1-1.5 %, p90 2.3-3 %)
    with occasional real corrections. Fixed thresholds either never promote (1 %: below the
    median) or never flag corrections, so both come from the recent distribution:

        promote_thr = median                       (a move no larger than typical jitter)
        outlier_thr = median + k * 1.4826 * MAD    (robust z > k: a correction, not jitter)

    Until ``min_samples`` moves were seen the fixed fallbacks apply."""

    def __init__(self, window: int = 512, k: float = 4.0, min_samples: int = 32,
                 promote_fallback: float = 0.01, outlier_fallback: float = float("inf")):
        self.buf = np.full(window, np.nan, np.float64)
        self.n = 0
        self.k = k
        self.min_samples = min_samples
        self.fallback = (promote_fallback, outlier_fallback)

    def add(self, moves: np.ndarray):
        m = np.asarray(moves, np.float64)
        m = m[np.isfinite(m)]
        for v in m[-len(self.buf):]:
            self.buf[self.n % len(self.buf)] = v
            self.n += 1

    def thresholds(self) -> Tuple[float, float]:
        if self.n < self.min_samples:
            return self.fallback
        v = self.buf[np.isfinite(self.buf)]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        return med, med + self.k * 1.4826 * max(mad, 1e-6)
