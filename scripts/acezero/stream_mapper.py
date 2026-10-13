"""Continuous ACE mapping for the streaming sidecar: ACE-Zero's mapping / relocalization loop
run a few iterations at a time, in step with the 3DGS trainer, instead of as whole refits.

ACE-Zero (Brachmann et al., ECCV 2024) alternates two stages: train the scene coordinate head
on the registered images with their current poses as targets, then relocalize every image
with the new head; images that relocalize well join the next mapping round. A stream refit ran
that as one ~2 min job (buffer rebuild, warmup, a few thousand iterations) whose result landed
on the trainer as a jump. Here the same loop runs incrementally:

    add_frame   encode a newly registered frame once (a few augmented passes) into its own
                slot of a fixed patch buffer: no buffer rebuild, ever
    step        one head iteration on a random batch over all slots; targets are the slots'
                current poses (the pose table), so a relocalized pose is the next target
    set_pose    adopt a relocalization result as the frame's target
    publish     an eval copy of the network for localization (the training copy keeps moving)

Learning rate is constant at ACE-Zero's cooldown level (5e-4): the head is always warm. The
reprojection loss uses a fixed point of the dyntanh schedule. No pose refiner: poses change
only through relocalization, and the trainer sees every change as a pose event.

Cold start (no bootstrap map): the mapper starts from a fresh head and ``train_seed`` runs
ACE-Zero's seed stage on one frame - identity pose, scene coordinate targets from a
monocular depth map (dsac* loss with depth: predictions far from their target are pulled to
it), annealing the reprojection clamp - until the frame fits. That frame fixes the gauge; the
other frames register against the head as they arrive.

Python 3.8 (ACE-Zero env).
"""

import copy
import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from ace_loss import AceLoss
from ace_network import Regressor
from ace_prior_loss import PriorLoss
from ace_repro_loss import ReproLoss
from ace_util import get_pixel_grid
from dataset import CamLocDataset

_logger = logging.getLogger("ace-mapper")


class ContinuousMapper:
    def __init__(self, encoder_state: dict, head_state: Optional[dict], device: torch.device, *,
                 max_frames: int = 400, passes: int = 2, samples_per_pass: int = 1024,
                 lr: float = 5e-4, batch_size: int = 5120, image_resolution: int = 480,
                 aug_rotation: float = 15.0, aug_scale: float = 1.5, use_half: bool = True,
                 soft_clamp: float = 50.0, soft_clamp_min: float = 1.0, progress: float = 0.75,
                 seed: int = 0, num_head_blocks: int = 1, mean=(0.0, 0.0, 0.0)):
        """``head_state`` None: a fresh head (cold start, see ``train_seed``) with
        ``num_head_blocks`` and scene coordinates relative to ``mean``."""
        assert batch_size % 512 == 0
        self.device = device
        self.batch_size = batch_size
        self.passes = passes
        self.samples_per_pass = samples_per_pass
        self.per_frame = passes * samples_per_pass
        self.image_resolution = image_resolution
        self.aug_rotation = aug_rotation
        self.aug_scale = aug_scale
        self.use_half = use_half
        self.max_frames = max_frames
        self.rng = torch.Generator().manual_seed(seed)

        self.lr = lr
        if head_state is None:
            torch.manual_seed(seed)  # head init
            self.net = Regressor.create_from_encoder(
                encoder_state, mean=torch.tensor(mean, dtype=torch.float32),
                num_head_blocks=num_head_blocks, use_homogeneous=True).to(device)
        else:
            self.net = Regressor.create_from_split_state_dict(encoder_state, head_state).to(device)
        for p in self.net.encoder.parameters():
            p.requires_grad_(False)
        self.net.train()
        self.opt = torch.optim.AdamW(self.net.heads.parameters(), lr=lr, betas=(0.9, 0.95))
        self.scaler = GradScaler(enabled=use_half)
        self.grid = get_pixel_grid(Regressor.OUTPUT_SUBSAMPLE).to(device)

        # Fixed point of ACE's dyntanh schedule: the head is warm, no annealing from scratch
        total = 1000
        self.loss_iteration, self.loss_total = int(progress * total), total
        repro = ReproLoss(total_iterations=total, soft_clamp=soft_clamp, soft_clamp_min=soft_clamp_min,
                          type="dyntanh", circle_schedule=True)
        prior = PriorLoss(prior_loss_type="none", prior_loss_bandwidth=0.1, prior_loss_location=1,
                          device=device)
        self.loss = AceLoss(loss_structure="dsac*", repro_loss=repro, prior_loss=prior,
                            prior_loss_weight=1, use_depth=False, depth_min=0.1, depth_max=1000,
                            repro_loss_hard_clamp=1000, learning_rate_cooldown_trigger_px_threshold=10,
                            depth_target=10)
        # Seed stage: the same loss with depth targets (rows without one get no proxy loss)
        self.seed_loss = AceLoss(loss_structure="dsac*", repro_loss=repro, prior_loss=prior,
                                 prior_loss_weight=1, use_depth=True, depth_min=0.1, depth_max=1000,
                                 repro_loss_hard_clamp=1000,
                                 learning_rate_cooldown_trigger_px_threshold=10, depth_target=10)
        self.use_targets = False

        # Patch buffer: frame slot k owns rows [k * per_frame, (k + 1) * per_frame). CPU, fp16
        # features (~1 KB per patch); everything else is small
        n, C = max_frames * self.per_frame, self.net.feature_dim
        dt = torch.float16 if use_half else torch.float32
        self.feat = torch.empty((n, C), dtype=dt)
        self.px = torch.empty((n, 2))
        self.aug_inv = torch.empty((n, 3, 4))
        self.K = torch.empty((n, 3, 3))
        self.Kinv = torch.empty((n, 3, 3))
        self.tgt = torch.zeros((n, 3))   # CAMERA-space points from depth (0 = none); lifted by the slot's pose in step()
        self.depth_frac = 1.0            # share of depth-target rows used per step (1 = all)
        # Soft depth prior (after the seed stage): the normal reprojection loss stays the main
        # loss and the head also pays depth_weight * min(|sc - depth-lifted target|, depth_clip) on
        # rows with a target. ACE's own use_depth loss instead drops every prediction >0.1 from
        # the target out of the reprojection loss, which let monocular depth (a few % off) override
        # the multi-view constraints: inliers fell to 43% (84% without) and no pose ever anchored.
        self.depth_prior = False
        self.depth_weight = 50.0         # loss units per scene unit (focal/depth ~ 340 px per unit)
        self.depth_clip = 0.5
        self.pending_depth: Dict[str, tuple] = {}   # frame -> (depth file, focal) that arrived before the frame joined
        self.poses = torch.eye(4).repeat(max_frames, 1, 1).to(device)  # w2c target per slot
        self.c2ws = torch.eye(4).repeat(max_frames, 1, 1).to(device)   # their inverses (depth targets)
        self.slot_of: Dict[str, int] = {}
        self.extra_slots: Dict[str, List[int]] = {}   # more augmentations of one frame (seed)
        self._occupied: Optional[torch.Tensor] = None
        self.free: List[int] = list(range(max_frames))[::-1]
        self.iterations = 0
        self.inlier_ema: Optional[float] = None

    # ------------------------------------------------------------------ frames
    @property
    def frames(self) -> List[str]:
        return list(self.slot_of)

    @torch.no_grad()
    def add_frame(self, frame: str, w2c: np.ndarray, focal: float, depth_file: Optional[str] = None,
                  slots: int = 1) -> bool:
        """Encode a registered frame into free slots (False when the buffer is full).
        ``depth_file``: a depth map (.npy, camera z, any resolution, 0 = invalid) whose scene
        coordinates under ``w2c`` become the frame's targets (used while ``use_targets``).
        ``slots``: encode this many slots' worth of augmentations (the seed frame)."""
        refresh = frame in self.slot_of
        if refresh and depth_file is None:
            self.set_pose(frame, w2c)
            return True
        if not refresh and len(self.free) < slots:
            return False
        use_depth = depth_file is not None
        ds = CamLocDataset(rgb_files=frame, augment=True, aug_rotation=self.aug_rotation,
                           aug_scale_max=self.aug_scale, aug_scale_min=1 / self.aug_scale,
                           image_short_size=self.image_resolution, use_half=self.use_half,
                           use_heuristic_focal_length=focal <= 0,
                           depth_files=depth_file, use_depth=use_depth)
        if focal > 0:
            ds.set_external_focal_length(focal)
        # (no pose file: the dataset's depth targets are CAMERA coordinates; step() lifts them by
        # the slot's current pose, so a relocalized pose moves its own targets)
        taken = ([self.slot_of[frame]] + self.extra_slots.get(frame, []) if refresh
                 else [self.free.pop() for _ in range(slots)])
        was_training = self.net.training
        self.net.eval()
        for slot in taken:
            base = slot * self.per_frame
            for p in range(self.passes):
                image, mask, _, aug_inv, K, Kinv, coords, _, _ = ds[0]  # new random augmentation
                with autocast(enabled=self.use_half):
                    f = self.net.get_features(image[None].to(self.device))  # 1, C, H, W
                _, C, H, W = f.shape
                m = torch.nn.functional.interpolate(mask[None].float(), size=(H, W), mode="nearest").view(-1)
                if use_depth:
                    tg = torch.zeros((3, H, W))
                    ch, cw = min(H, coords.shape[1]), min(W, coords.shape[2])
                    tg[:, :ch, :cw] = coords[:, :ch, :cw]
                    tg = tg.reshape(3, -1).T
                    has = tg.abs().sum(1) > 1e-5
                    if (m * has.float()).sum() > 0:
                        m = m * has.float()  # sample where the depth is valid
                if m.sum() == 0:
                    m = torch.ones_like(m)
                idx = torch.multinomial(m, self.samples_per_pass, replacement=True, generator=self.rng)
                rows = slice(base + p * self.samples_per_pass, base + (p + 1) * self.samples_per_pass)
                self.feat[rows] = f[0].reshape(C, -1).T[idx.to(self.device)].to(self.feat.dtype).cpu()
                self.px[rows] = self.grid[:, :H, :W].reshape(2, -1).T[idx.to(self.device)].cpu()
                self.aug_inv[rows] = aug_inv[:3].expand(self.samples_per_pass, 3, 4)
                self.K[rows] = K.expand(self.samples_per_pass, 3, 3)
                self.Kinv[rows] = Kinv.expand(self.samples_per_pass, 3, 3)
                self.tgt[rows] = tg[idx] if use_depth else 0.0
        if was_training:
            self.net.train()
        if not refresh:
            self.slot_of[frame] = taken[0]
            if len(taken) > 1:
                self.extra_slots[frame] = taken[1:]
        self._occupied = None
        self.set_pose(frame, w2c)
        late = self.pending_depth.pop(frame, None) if depth_file is None else None
        if late is not None:
            self.set_depth(frame, late[0], late[1])
        return True

    def set_depth(self, frame: str, depth_file: str, focal: float) -> bool:
        """Attach (or replace) the aligned depth map of a registered frame: its slots are
        re-encoded with depth targets under the frame's current pose, and from now on the
        mapper loss pulls invalid predictions toward the depth-lifted points."""
        slot = self.slot_of.get(frame)
        if slot is None:
            self.pending_depth[frame] = (depth_file, focal)   # applied when the frame joins
            return False
        w2c = self.poses[slot].detach().cpu().numpy()
        ok = self.add_frame(frame, w2c, focal, depth_file=depth_file)
        if ok:
            self.depth_prior = True
        return ok

    def set_pose(self, frame: str, w2c: np.ndarray):
        slot = self.slot_of.get(frame)
        if slot is not None:
            pose = torch.as_tensor(np.asarray(w2c), dtype=torch.float32, device=self.device)
            inv = torch.linalg.inv(pose)
            for s in [slot] + self.extra_slots.get(frame, []):
                self.poses[s] = pose
                self.c2ws[s] = inv

    def remove_frame(self, frame: str):
        slot = self.slot_of.pop(frame, None)
        self.pending_depth.pop(frame, None)
        if slot is not None:
            for s in [slot] + self.extra_slots.pop(frame, []):
                self.tgt[s * self.per_frame:(s + 1) * self.per_frame] = 0.0
                self.free.append(s)
            self._occupied = None

    def occupied(self) -> torch.Tensor:
        if self._occupied is None:
            self._occupied = torch.tensor(list(self.slot_of.values())
                                          + [s for v in self.extra_slots.values() for s in v])
        return self._occupied

    # ------------------------------------------------------------------ training
    def step(self) -> Optional[float]:
        """One head iteration; returns the batch inlier ratio (10 px), None without frames."""
        if not self.slot_of:
            return None
        slots = self.occupied()
        pick = slots[torch.randint(len(slots), (self.batch_size,), generator=self.rng)]
        rows = pick * self.per_frame + torch.randint(self.per_frame, (self.batch_size,), generator=self.rng)
        d = self.device
        feat = self.feat[rows].to(d, non_blocking=True)
        px = self.px[rows].to(d, non_blocking=True)
        aug_inv = self.aug_inv[rows].to(d, non_blocking=True)
        K = self.K[rows].to(d, non_blocking=True)
        Kinv = self.Kinv[rows].to(d, non_blocking=True)
        w2c = self.poses[pick.to(d)]

        C = feat.shape[1]
        f = feat[None, None, ...].view(-1, 16, 32, C).permute(0, 3, 1, 2)
        with autocast(enabled=self.use_half):
            sc = self.net.get_scene_coordinates(f)
        sc = sc.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()  # b, 3, 1
        sc_h = torch.cat([sc, torch.ones_like(sc[:, :1])], 1)
        cam = torch.bmm(torch.bmm(aug_inv, w2c), sc_h)  # b, 3, 1
        proj = torch.bmm(K, cam)
        proj[:, 2].clamp_(min=0.1)
        err = torch.norm(proj[:, :2, 0] / proj[:, 2:3, 0] - px, dim=1, keepdim=True, p=1)
        if self.use_targets:
            target, _ = self._depth_targets(pick, rows)
            loss_fn = self.seed_loss
        else:
            loss_fn, target = self.loss, torch.zeros_like(px[:, :1]).expand(-1, 3)
        inliers, loss = loss_fn.compute(
            pred_cam_coords_b31=cam, pred_scene_coords_b31=sc, reprojection_error_b1=err,
            target_crds_b3=target, target_px_b2=px,
            invKs_b33=Kinv, iteration=self.loss_iteration, max_iterations=self.loss_total)
        if self.depth_prior and not self.use_targets:
            target, keep = self._depth_targets(pick, rows)
            if keep.any():
                dist = (sc.squeeze(-1) - target).norm(dim=1).clamp(max=self.depth_clip)
                loss = loss + self.depth_weight * (dist * keep).sum() / self.batch_size
        if not torch.isfinite(loss):
            _logger.warning("non-finite mapper loss; step skipped")
            return None
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.iterations += 1
        inl = float(inliers)
        self.inlier_ema = inl if self.inlier_ema is None else 0.98 * self.inlier_ema + 0.02 * inl
        return inl

    def train_seed(self, min_iters: int = 300, max_iters: int = 2000, min_inliers: float = 0.7,
                   lr: float = 2e-3, keep_targets: bool = False, stop=None) -> int:
        """ACE-Zero's seed stage on the frames added so far (the seed frame, with depth
        targets): ``lr``, the reprojection clamp annealed to the mapper's fixed point over
        ``min_iters``, until the batch inlier EMA reaches ``min_inliers`` (after at least
        ``min_iters``) or ``max_iters``. Then the mapper's own LR and loss (depth targets
        dropped unless ``keep_targets``: as in ACE-Zero, depth only initializes the map).
        Returns the iterations run."""
        fixed = self.loss_iteration
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.use_targets = True
        self.inlier_ema = None
        it = 0
        try:
            while it < max_iters and not (stop is not None and stop.is_set()):
                self.loss_iteration = int(fixed * min(1.0, it / max(1, min_iters)))
                self.step()
                it += 1
                if it >= min_iters and (self.inlier_ema or 0.0) >= min_inliers:
                    break
        finally:
            self.loss_iteration = fixed
            for g in self.opt.param_groups:
                g["lr"] = self.lr
            self.use_targets = keep_targets
        return it

    def _depth_targets(self, pick: torch.Tensor, rows: torch.Tensor):
        """World-space depth targets of a batch (camera points lifted by each slot's current
        pose) and the mask of rows that have one."""
        d = self.device
        cam_t = self.tgt[rows].to(d, non_blocking=True)
        c2w = self.c2ws[pick.to(d)]
        target = torch.bmm(c2w[:, :3, :3], cam_t.unsqueeze(-1)).squeeze(-1) + c2w[:, :3, 3]
        keep = cam_t.abs().sum(1) > 1e-5
        if self.depth_frac < 1.0:
            keep = keep & (torch.rand(keep.shape, device=d) < self.depth_frac)
        return torch.where(keep.unsqueeze(1), target, torch.zeros_like(target)), keep

    def snapshot(self) -> Regressor:
        """Frozen eval copy for localization."""
        net = copy.deepcopy(self.net)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    def head_state(self) -> dict:
        return {k: v.detach().half().cpu() for k, v in self.net.heads.state_dict().items()}
