"""Glue between a StreamSession / online Difix and examples/simple_trainer.py.

    StreamParser     the COLMAP Parser attributes the trainer reads, backed by the session
    StreamTrainSet   endless sampler over the session's current train views (real + pseudo)
    StreamValSet     held-out keyframes, snapshotted at each eval
    WithPseudo       batch datasets + the latest Difix pseudo views (online Difix, batch mode)
    inject_gaussians grow the model where new keyframes see unmodelled space
    StreamViewAdjust per-keyframe pose deltas + exposure (gain/bias) for a changing image set
"""

import math
import queue
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .native import weighted_pick
from .schema import select_rows
from .source import decode_jpeg
from .store import PseudoView


# ============================================================================ parser stand-in
class StreamParser:
    """Duck-types datasets.colmap.Parser for the trainer (normalized frame, one camera)."""

    def __init__(self, session, data_dir: str):
        self.session = session
        self.data_dir = str(data_dir)
        self.factor = 1
        self.normalize = True
        self.test_every = session.cfg.test_every
        self.transform = session.transform
        self.scene_scale = max(float(session.scene_scale), 1e-3)
        xyz, rgb = session.take_seeds()
        self.points = xyz.astype(np.float32)
        self.points_rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        self.points_err = np.zeros(len(xyz), np.float32)
        self.point_indices: Dict = {}
        self.params_dict = {0: np.empty(0, np.float32)}
        self.mask_dict = {0: None}
        self.tensor_cache: Dict = {}
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}
        self.bounds = np.array([0.01, 1.0])

    def _train_views(self):
        _, table = self.session.views()
        return select_rows(table, kind="real")

    @property
    def camtoworlds(self) -> np.ndarray:
        t = self._train_views()
        return np.array(t.column("c2w").to_pylist(), dtype=np.float64).reshape(-1, 4, 4)

    @property
    def image_names(self) -> List[str]:
        return self._train_views().column("name").to_pylist()

    @property
    def image_paths(self) -> List[str]:
        return ["STREAM_MEMORY"] * len(self.image_names)

    @property
    def camera_ids(self) -> List[int]:
        return [0] * len(self.image_names)

    @property
    def Ks_dict(self) -> Dict[int, np.ndarray]:
        w, h = self.session.train_size
        return {0: self.session.intrinsics(w, h)}

    @property
    def imsize_dict(self) -> Dict[int, Tuple[int, int]]:
        return {0: tuple(self.session.train_size)}

    def refresh(self) -> int:
        return 0


def _view_item(table, i: int, image_id: int, session=None) -> Dict:
    image = decode_jpeg(table.column("jpeg")[i].as_py())
    item = {
        "K": torch.tensor(table.column("K")[i].as_py(), dtype=torch.float32).reshape(3, 3),
        "camtoworld": torch.tensor(table.column("c2w")[i].as_py(), dtype=torch.float32).reshape(4, 4),
        "image": torch.from_numpy(image).float(),
        "image_id": image_id,
        "loss_weight": float(table.column("weight")[i].as_py()),
        "is_pseudo": table.column("kind")[i].as_py() == "pseudo",
        "kid": int(table.column("kid")[i].as_py()),
    }
    if session is not None and session.cfg.depth:
        # Always present (zeros = no depth yet / pseudo view) so batches collate uniformly
        h, w = image.shape[:2]
        d = session.depth_map(int(table.column("kid")[i].as_py()))
        item["depth"] = (torch.from_numpy(d.astype(np.float32)) if d is not None and d.shape == (h // 2, w // 2)
                         else torch.zeros(h // 2, w // 2))
        wmap = session.depth_weight(int(table.column("kid")[i].as_py()))
        item["depth_w"] = (torch.from_numpy(wmap.astype(np.float32)) if wmap is not None and wmap.shape == (h // 2, w // 2)
                           else torch.ones(h // 2, w // 2))
    return item


# ============================================================================ device poses
class DevicePoses:
    """Trainer side of the device pose table (hypersplat/stream/cuda).

    Every step: ``sync`` applies the session's queued pose rows on the current CUDA stream and
    reads back the previous applies' results without blocking (jitter statistics -> filter /
    tier thresholds, host tier map -> sampling weights); ``camtoworlds`` swaps each real view's
    camera for its row of the table (a device gather, no host sync). With alpha = 1 and tiers
    off the table reproduces the session's poses exactly."""

    TIER_WEIGHTS = {0: 0.0, 1: 0.3, 2: 1.0, 3: 1.0}

    def __init__(self, session, trainset=None, alpha: float = 1.0, tiers: bool = False,
                 provisional_weight: float = 0.3, capacity: Optional[int] = None):
        from .cuda import DevicePoseTable, JitterFloor

        self.session = session
        self.trainset = trainset
        self.alpha = float(alpha)
        self.tiers = bool(tiers)
        self.weights = {**self.TIER_WEIGHTS, 1: float(provisional_weight)}
        self.table = DevicePoseTable(capacity or session.cfg.max_keyframes + 64)
        self.jitter = JitterFloor()
        self.tier_of: Dict[int, int] = {}
        self.extent = 1.0
        self._transform_set = False
        self.counts = {"applied": 0, "stale": 0, "unregistered": 0, "outlier": 0, "frozen": 0}
        # Rows applied before the table existed (first map, early localizations): seed it with
        # the store's current poses, ahead of anything queued from now on
        with session._cond:
            session.device_deltas_on = True
            session._device_deltas.insert(0, self._store_records())

    def _store_records(self) -> np.ndarray:
        from .ring import DELTA_DTYPE, KIND_POSE

        frames = [kf for kf in self.session.store.frames() if kf.w2c is not None]
        recs = np.zeros(len(frames), DELTA_DTYPE)
        for r, kf in zip(recs, frames):
            r["kid"], r["map_version"], r["kind"] = kf.kid, kf.map_version, KIND_POSE
            r["conf"], r["step"] = kf.conf, -1
            r["inlier_frac"] = r["move"] = np.nan
            r["w2c"] = np.asarray(kf.w2c, np.float64).reshape(16)
        return recs

    def sync(self, step: int):
        from .cuda import APPLIED, FROZEN, OUTLIER, STALE, UNREGISTERED

        s = self.session
        if not self._transform_set:
            if s.transform is None:
                return                       # before the first map: rows stay queued
            self.table.set_transform(s.transform)
            self._transform_set = True
        if step % 50 == 0:
            self.extent = s.path_extent()
            live = {kf.kid for kf in s.store.frames()}
            for kid in [k for k in self.table._slot if k not in live]:
                self.table.release(kid)
                self.tier_of.pop(kid, None)
        recs = s.take_device_deltas()
        if len(recs):
            self.table.ensure_stage_free()
            promote, outlier = self.jitter.thresholds()
            self.table.apply(recs, conf_threshold=s.store.conf_threshold, extent=self.extent,
                             alpha=self.alpha, promote_thr=promote, outlier_thr=outlier)
        res = self.table.poll()
        if res is None:
            return
        ok = res["result"] == APPLIED
        self.jitter.add(res["move"][ok])
        for name, code in (("applied", APPLIED), ("stale", STALE), ("unregistered", UNREGISTERED),
                           ("outlier", OUTLIER), ("frozen", FROZEN)):
            self.counts[name] += int((res["result"] == code).sum())
        for kid, t in zip(res["kid"].tolist(), res["tier"].tolist()):
            self.tier_of[int(kid)] = int(t)
        if self.tiers and self.trainset is not None:
            self.trainset.tier_weight = {k: self.weights[t] for k, t in self.tier_of.items()}

    def camtoworlds(self, kids, c2w: torch.Tensor) -> torch.Tensor:
        """c2w (B, 4, 4) on the device with real views' rows replaced by the table's."""
        if not self._transform_set:
            return c2w
        kid_list = kids.tolist() if torch.is_tensor(kids) else list(np.atleast_1d(kids))
        slots = [self.table.slot_of(int(k)) if int(k) >= 0 else -1 for k in kid_list]
        if all(sl < 0 for sl in slots):
            return c2w
        idx = torch.tensor([max(sl, 0) for sl in slots], device=c2w.device)
        valid = torch.tensor([sl >= 0 for sl in slots], device=c2w.device)
        valid = valid & (self.table.has_pose.index_select(0, idx) > 0)
        rows = self.table.c2w_trainer.index_select(0, idx).reshape(-1, 4, 4).to(c2w.dtype)
        return torch.where(valid[:, None, None], rows, c2w)

    def stats(self) -> Dict:
        tiers = np.bincount(np.array(list(self.tier_of.values()) or [0]), minlength=4).tolist()
        promote, outlier = self.jitter.thresholds()
        return {**self.counts, "tiers": tiers if self.tier_of else [0, 0, 0, 0],
                "promote_thr": round(promote, 4), "outlier_thr": round(outlier, 4), "extent": round(self.extent, 3)}


# ============================================================================ datasets
class StreamTrainSet(torch.utils.data.IterableDataset):
    """Samples the session's current train views forever. Each draw re-reads the session's
    (cached) Arrow snapshot, so new keyframes, refined poses and pseudo views take effect on
    the next step without rebuilding the DataLoader. Views are drawn with probability
    ~ 1/sqrt(1 + times drawn): new keyframes get attention quickly, old ones are not starved.

    With ``hard_mix`` > 0 that weight is also scaled by how badly each real view is fitted:
    (1 - hard_mix) + hard_mix * hardness, hardness = (loss EMA / median EMA) ** hard_power
    clipped to [1/4, 4] (views not scored yet and pseudo views count as median). The trainer
    reports each step's photometric loss through ``report_loss``. A batch of one view per step
    makes the loss swing 33-63% step to step; drawing under-fitted views more often spends
    those steps where the error (and the floaters) are.

    Use with num_workers=0 (the snapshot lives in this process); decoding is prefetched on a
    thread instead."""

    def __init__(self, session, seed: int = 0, prefetch: int = 2, hard_mix: float = 0.0,
                 hard_power: float = 1.0, loss_beta: float = 0.7, pseudo_share: Optional[float] = None):
        super().__init__()
        self.session = session
        self.prefetch = prefetch
        self.rng = np.random.default_rng(seed)
        self.counts: Dict[str, int] = {}
        self.hard_mix = float(hard_mix)
        self.hard_power = float(hard_power)
        self.loss_beta = float(loss_beta)
        # Cap on the probability of drawing a pseudo (Difix) view. Every cycle round adds fresh,
        # never-drawn pseudo views, which the novelty weight favours: uncapped they took ~40% of
        # the steps of a 2k drone run. None = no cap
        self.pseudo_share = pseudo_share
        self.loss_ema: Dict[int, float] = {}   # kid -> photometric loss EMA (real views)
        # kid -> sampling factor from the device pose table's stability tier (DevicePoses);
        # views not listed (and pseudo views) count 1
        self.tier_weight: Dict[int, float] = {}
        self._version = None
        self._table = None

    def report_loss(self, kids, losses):
        """Per-view photometric loss of the step just trained (pseudo views: kid < 0 or skip)."""
        b = self.loss_beta
        for kid, loss in zip(kids, losses):
            kid, loss = int(kid), float(loss)
            if kid < 0 or not np.isfinite(loss):
                continue
            prev = self.loss_ema.get(kid)
            self.loss_ema[kid] = loss if prev is None else b * prev + (1.0 - b) * loss

    def _cap_pseudo(self, w: np.ndarray) -> np.ndarray:
        """Scale pseudo-view weights down so they hold at most ``pseudo_share`` of the draw
        probability (real views keep their relative weights)."""
        cap = self.pseudo_share
        if cap is None or not any(self._pseudo):
            return w
        p = np.asarray(self._pseudo, dtype=bool)
        tot_p, tot_r = float(w[p].sum()), float(w[~p].sum())
        if tot_r <= 0 or tot_p <= 0 or tot_p / (tot_p + tot_r) <= cap:
            return w
        w = w.copy()
        w[p] *= cap * tot_r / ((1.0 - cap) * tot_p)
        return w

    def hardness(self) -> np.ndarray:
        """Per current train view sampling factor from the loss EMAs (1 = neutral)."""
        n = len(self._names)
        if self.hard_mix <= 0 or not self.loss_ema:
            return np.ones(n)
        ema = np.array([self.loss_ema.get(k, np.nan) if not p else np.nan
                        for k, p in zip(self._kids, self._pseudo)], dtype=np.float64)
        scored = np.isfinite(ema)
        if scored.sum() < 2:
            return np.ones(n)
        rel = np.ones(n)
        rel[scored] = np.clip((ema[scored] / max(np.median(ema[scored]), 1e-8)) ** self.hard_power, 0.25, 4.0)
        return (1.0 - self.hard_mix) + self.hard_mix * rel

    def __len__(self) -> int:  # informational (CameraOptModule sizing etc.)
        # Read-only: _refresh() swaps the sampler state the prefetch thread is drawing from
        _, table = self.session.views()
        return select_rows(table, split="train").num_rows

    def _refresh(self):
        version, table = self.session.views()
        if version != self._version:
            self._table = select_rows(table, split="train")
            self._names = self._table.column("name").to_pylist()
            self._kids = self._table.column("kid").to_pylist()
            self._pseudo = [k == "pseudo" for k in self._table.column("kind").to_pylist()]
            self._drawn = np.array([self.counts.get(k, 0) for k in self._names], dtype=np.float64)
            self._version = version

    def sample(self) -> Dict:
        while True:
            self._refresh()
            if self._table is not None and self._table.num_rows:
                break
            time.sleep(0.05)
        w = self.hardness() / np.sqrt(1.0 + self._drawn)
        if self.tier_weight:
            tw = self.tier_weight
            w = w * np.array([1.0 if p else tw.get(k, 1.0) for k, p in zip(self._kids, self._pseudo)])
        i = weighted_pick(self._cap_pseudo(w), self.rng.random())
        self._drawn[i] += 1
        self.counts[self._names[i]] = int(self._drawn[i])
        return _view_item(self._table, i, 0, self.session)

    def __iter__(self):
        if self.prefetch <= 0:
            # Inline draws (deterministic runs): each view is drawn after the previous step's
            # loss report and against that step's snapshot, not ahead of it on a thread
            while True:
                yield self.sample()
        # JPEG decode (~5 ms at 960 px, libjpeg releases the GIL) runs on a prefetch thread,
        # overlapped with the GPU step instead of adding to it
        q: "queue.Queue[Dict]" = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def fill():
            while not stop.is_set():
                item = self.sample()
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        pass

        threading.Thread(target=fill, name="stream-prefetch", daemon=True).start()
        try:
            while True:
                yield q.get()
        finally:
            stop.set()


class StreamValSet(torch.utils.data.Dataset):
    """Held-out keyframes (every test_every-th); refresh() before each eval."""

    def __init__(self, session):
        self.session = session
        self._table = None
        self.refresh()

    def refresh(self):
        _, table = self.session.views()
        self._table = select_rows(table, split="val", kind="real")

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, i: int) -> Dict:
        item = _view_item(self._table, i, i)
        item.pop("is_pseudo")
        item.pop("loss_weight")
        return item


class WithPseudo(torch.utils.data.Dataset):
    """A batch dataset plus the current Difix pseudo views (appended at the end).

    Pseudo items carry is_pseudo=True (the trainer skips pose correction for them - they were
    rendered from the model itself) and the train index of their source view as image_id, so
    per-image appearance modules use the source frame's embedding."""

    def __init__(self, base, views: Sequence[PseudoView], src_index: Sequence[int]):
        self.base = base
        self.views = list(views)
        self.src_index = list(src_index)

    def __len__(self) -> int:
        return len(self.base) + len(self.views)

    def __getitem__(self, i: int) -> Dict:
        if i < len(self.base):
            item = self.base[i]
            item["is_pseudo"] = False
            return item
        k = i - len(self.base)
        p = self.views[k]
        return {
            "K": torch.from_numpy(np.asarray(p.K, np.float32)),
            "camtoworld": torch.from_numpy(np.asarray(p.c2w, np.float32)),
            "image": torch.from_numpy(decode_jpeg(p.jpeg.to_pybytes())).float(),
            "image_id": int(self.src_index[k]),
            "loss_weight": float(p.weight),
            "is_pseudo": True,
        }

    def __getattr__(self, name):  # indices, parser, image_weights, ... of the base dataset
        if name in ("base", "views", "src_index"):
            raise AttributeError(name)
        return getattr(self.base, name)


# ============================================================================ Gaussians
def _hash(q: torch.Tensor) -> torch.Tensor:
    """Non-negative voxel coordinates -> one int64 (21 bits per axis)."""
    q = q.clamp(0, (1 << 21) - 1)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


class SlotMap:
    """Keyframe id -> embedding slot for per-view modules over a changing image set.

    Keyframe ids grow forever while at most ``capacity`` keyframes are alive, so slots of
    evicted keyframes are reused (least recently used first). ``assign`` reports when a slot
    must be reset: freshly (re)assigned, or the keyframe's pose moved (an ACE refit) since the
    slot learned against it."""

    def __init__(self, capacity: int, move_tol: float = 0.01, angle_tol_deg: float = 0.5):
        self.capacity = capacity
        self.move_tol = move_tol
        self.cos_tol = math.cos(math.radians(angle_tol_deg))
        self.slot_of: Dict[int, int] = {}
        self.c2w: Dict[int, np.ndarray] = {}      # slot -> pose its deltas were learned against
        self.used: Dict[int, int] = {}            # slot -> last use tick (LRU)
        self.free = list(range(capacity - 1, -1, -1))
        self._tick = 0

    def assign(self, kid: int, c2w: np.ndarray) -> Tuple[int, bool, bool]:
        """(slot, fresh, moved)."""
        self._tick += 1
        slot = self.slot_of.get(kid)
        fresh = slot is None
        if fresh:
            if self.free:
                slot = self.free.pop()
            else:  # all taken: recycle the least recently used keyframe's slot
                slot = min(self.used, key=self.used.get)
                old = next(k for k, v in self.slot_of.items() if v == slot)
                del self.slot_of[old]
            self.slot_of[kid] = slot
            self.c2w[slot] = np.array(c2w, np.float64)
        moved = False
        if not fresh:
            ref = self.c2w[slot]
            dt = np.linalg.norm(ref[:3, 3] - c2w[:3, 3])
            cos = (np.trace(ref[:3, :3].T @ c2w[:3, :3]) - 1.0) / 2.0
            if dt > self.move_tol or cos < self.cos_tol:
                moved = True
                self.c2w[slot] = np.array(c2w, np.float64)
        self.used[slot] = self._tick
        return slot, fresh, moved

    def release_missing(self, live_kids) -> int:
        """Free the slots of keyframes no longer in the store; returns how many."""
        gone = [k for k in self.slot_of if k not in live_kids]
        for k in gone:
            slot = self.slot_of.pop(k)
            self.used.pop(slot, None)
            self.c2w.pop(slot, None)
            self.free.append(slot)
        return len(gone)

    @property
    def active(self) -> List[int]:
        return list(self.slot_of.values())

    def state_dict(self) -> Dict:
        return {"slot_of": dict(self.slot_of), "c2w": {k: v.tolist() for k, v in self.c2w.items()}}


def _zero_rows(param: torch.nn.Parameter, optimizers: Sequence, rows: torch.Tensor):
    """Zero parameter rows and their Adam moments."""
    with torch.no_grad():
        param[rows] = 0.0
    for opt in optimizers:
        for v in opt.state.get(param, {}).values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == param.shape[0]:
                v[rows] = 0.0


class StreamViewAdjust(torch.nn.Module):
    """Per-keyframe corrections for a stream: pose deltas (a CameraOptModule over slots) and
    exposure, a per-channel log-gain + bias on the rendered colours (camera auto-exposure /
    white balance). Pseudo views and held-out views get no slot (-1): no correction.

    Exposure is a gauge the SH colours could absorb: a common offset in every slot would move
    into the Gaussians' colours and give every uncorrected (held-out) render a colour cast.
    Corrections are therefore applied relative to the mean over live slots, so the average
    correction is exactly none (a soft penalty on the mean was too weak: -2.9 dB)."""

    def __init__(self, capacity: int, pose_module: Optional[torch.nn.Module] = None,
                 exposure: bool = True, move_tol: float = 0.01):
        super().__init__()
        self.slots = SlotMap(capacity, move_tol=move_tol)
        self.pose = pose_module
        self.exposure = torch.nn.Embedding(capacity, 6) if exposure else None
        if self.exposure is not None:
            torch.nn.init.zeros_(self.exposure.weight)
        self.pose_optimizers: List = []
        self.exposure_optimizers: List = []
        self.resets = 0

    def slot_ids(self, kids: torch.Tensor, c2ws: torch.Tensor, is_pseudo: Optional[torch.Tensor]) -> torch.Tensor:
        out = []
        pose_reset, full_reset = [], []
        for i, kid in enumerate(kids.tolist()):
            if kid < 0 or (is_pseudo is not None and bool(is_pseudo[i])):
                out.append(-1)
                continue
            slot, fresh, moved = self.slots.assign(int(kid), c2ws[i].detach().cpu().numpy())
            if fresh:
                full_reset.append(slot)
            elif moved:
                pose_reset.append(slot)
            out.append(slot)
        device = c2ws.device
        if full_reset or pose_reset:
            self.resets += len(full_reset) + len(pose_reset)
            if self.pose is not None:
                _zero_rows(self.pose.embeds.weight, self.pose_optimizers,
                           torch.tensor(full_reset + pose_reset, device=device))
            if self.exposure is not None and full_reset:
                _zero_rows(self.exposure.weight, self.exposure_optimizers, torch.tensor(full_reset, device=device))
        return torch.tensor(out, device=device, dtype=torch.long)

    def adjust_pose(self, c2ws: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        m = slots >= 0
        if self.pose is None or not m.any():
            return c2ws
        if m.all():
            return self.pose(c2ws, slots)
        out = c2ws.clone()
        out[m] = self.pose(c2ws[m], slots[m])
        return out

    def adjust_colors(self, colors: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """colors (B, H, W, 3)."""
        m = slots >= 0
        if self.exposure is None or not m.any():
            return colors
        active = torch.tensor(self.slots.active, device=slots.device)
        mean = self.exposure.weight[active].mean(0) if len(active) else 0.0
        e = (self.exposure(slots.clamp_min(0)) - mean) * m[:, None].float()   # (B, 6), zero where no slot
        gain, bias = e[:, None, None, :3], e[:, None, None, 3:]
        return colors * torch.exp(gain) + bias

    def exposure_reg(self) -> torch.Tensor:
        if self.exposure is None or not self.slots.active:
            return torch.zeros((), device=self.exposure.weight.device if self.exposure is not None else "cpu")
        idx = torch.tensor(self.slots.active, device=self.exposure.weight.device)
        return self.exposure.weight[idx].mean(0).pow(2).sum()

    def extra_state(self) -> Dict:
        return {"slots": self.slots.state_dict(), "resets": self.resets}


@torch.no_grad()
def inject_gaussians(splats: torch.nn.ParameterDict, optimizers: Dict, strategy_state: Dict,
                     xyz: np.ndarray, rgb: np.ndarray, voxel: float, init_opacity: float = 0.1,
                     init_scale: float = 1.0, cap_max: Optional[int] = None) -> Tuple[int, int]:
    """Add Gaussians for seed points in voxels no Gaussian occupies yet.

    One Gaussian per empty voxel (``voxel`` in normalized world units), so seeds from many
    keyframes of the same area do not pile up. Under an MCMC cap the lowest-opacity (dead)
    Gaussians are recycled first; the rest is appended up to the cap. Optimizer moments and
    per-Gaussian strategy state are zero for the new entries. Returns (added, recycled).
    """
    from gsplat.strategy.ops import _update_param_with_optimizer

    if len(xyz) == 0:
        return 0, 0
    if "sh0" not in splats:
        raise ValueError("inject_gaussians supports SH colour models only (not app_opt features)")
    device = splats["means"].device
    pts = torch.from_numpy(np.asarray(xyz, np.float32)).to(device)
    cols = torch.from_numpy(np.asarray(rgb, np.float32)).to(device).clamp(0, 1)

    # Keep one seed per empty voxel
    origin = torch.minimum(pts.min(0).values, splats["means"].min(0).values)
    occupied = _hash(torch.floor((splats["means"] - origin) / voxel).long())
    cells = _hash(torch.floor((pts - origin) / voxel).long())
    fresh = ~torch.isin(cells, occupied)
    cells, first = torch.unique(cells[fresh], return_inverse=True)
    if len(cells) == 0:
        return 0, 0
    pick = torch.full((len(cells),), -1, dtype=torch.long, device=device)
    idx = torch.nonzero(fresh).squeeze(1)
    pick.scatter_(0, first, idx)  # any representative per cell
    pts, cols = pts[pick], cols[pick]
    m = len(pts)

    sh_rest = splats["shN"].shape[1]
    C0 = 0.28209479177387814
    new = {
        "means": pts,
        "scales": torch.full((m, 3), math.log(voxel * init_scale), device=device),
        "quats": torch.nn.functional.normalize(torch.rand((m, 4), device=device), dim=-1),
        "opacities": torch.full((m,), float(torch.logit(torch.tensor(init_opacity))), device=device),
        "sh0": ((cols - 0.5) / C0).unsqueeze(1),
        "shN": torch.zeros((m, sh_rest, 3), device=device),
    }

    n = len(splats["means"])
    recycled = 0
    if cap_max is not None and n + m > cap_max:
        # Recycle Gaussians that have faded out (MCMC relocates them anyway)
        opa = torch.sigmoid(splats["opacities"])
        dead = torch.nonzero(opa < 0.005).squeeze(1)
        dead = dead[torch.argsort(opa[dead])][: n + m - cap_max]
        recycled = len(dead)
        if recycled:
            for name, value in new.items():
                splats[name].data[dead] = value[:recycled]
            for name, opt in optimizers.items():
                state = opt.state.get(splats[name])
                for key, v in (state or {}).items():
                    if key != "step" and isinstance(v, torch.Tensor) and v.shape[:1] == (n,):
                        v[dead] = 0
            new = {k: v[recycled:] for k, v in new.items()}
        room = max(cap_max - n, 0)
        new = {k: v[:room] for k, v in new.items()}
    k = len(new["means"])
    if k:
        def param_fn(name, p):
            return torch.nn.Parameter(torch.cat([p, new[name].to(p.dtype)]), requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.cat([v, torch.zeros((k, *v.shape[1:]), device=v.device, dtype=v.dtype)])

        _update_param_with_optimizer(param_fn, optimizer_fn, splats, optimizers, names=list(new))
        for key, v in list(strategy_state.items()):
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == n:
                strategy_state[key] = torch.cat([v, torch.zeros((k, *v.shape[1:]), device=v.device, dtype=v.dtype)])
    return k, recycled


def seed_voxel_size(points: np.ndarray, scene_scale: float) -> float:
    """Voxel for seed de-duplication: the initial cloud's median nearest-neighbour spacing,
    clamped to [scene_scale / 1000, scene_scale / 50]."""
    lo, hi = scene_scale / 1000.0, scene_scale / 50.0
    if len(points) < 16:
        return hi
    sample = points[np.random.default_rng(0).choice(len(points), min(len(points), 4000), replace=False)]
    d = torch.cdist(torch.from_numpy(sample).float(), torch.from_numpy(sample).float())
    d.fill_diagonal_(float("inf"))
    return float(np.clip(d.min(1).values.median().item(), lo, hi))


class StopFile:
    """``<result_dir>/STOP`` ends a live run gracefully (final ckpt/ply/eval as usual).

    Only a file created after this run started counts: a stale one from an earlier run is
    ignored, while one written during the bootstrap (before training) is honoured."""

    def __init__(self, result_dir: str):
        from pathlib import Path
        self.path = Path(result_dir) / "STOP"
        self.since = time.time() - 1.0  # filesystem mtime granularity

    def requested(self) -> bool:
        try:
            return self.path.stat().st_mtime >= self.since
        except OSError:
            return False
