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
    def __init__(self, encoder_state: dict, head_state: dict, device: torch.device, *,
                 max_frames: int = 400, passes: int = 2, samples_per_pass: int = 1024,
                 lr: float = 5e-4, batch_size: int = 5120, image_resolution: int = 480,
                 aug_rotation: float = 15.0, aug_scale: float = 1.5, use_half: bool = True,
                 soft_clamp: float = 50.0, soft_clamp_min: float = 1.0, progress: float = 0.75,
                 seed: int = 0):
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

        # Patch buffer: frame slot k owns rows [k * per_frame, (k + 1) * per_frame). CPU, fp16
        # features (~1 KB per patch); everything else is small
        n, C = max_frames * self.per_frame, self.net.feature_dim
        dt = torch.float16 if use_half else torch.float32
        self.feat = torch.empty((n, C), dtype=dt)
        self.px = torch.empty((n, 2))
        self.aug_inv = torch.empty((n, 3, 4))
        self.K = torch.empty((n, 3, 3))
        self.Kinv = torch.empty((n, 3, 3))
        self.poses = torch.eye(4).repeat(max_frames, 1, 1).to(device)  # w2c target per slot
        self.slot_of: Dict[str, int] = {}
        self.free: List[int] = list(range(max_frames))[::-1]
        self.iterations = 0
        self.inlier_ema: Optional[float] = None

    # ------------------------------------------------------------------ frames
    @property
    def frames(self) -> List[str]:
        return list(self.slot_of)

    @torch.no_grad()
    def add_frame(self, frame: str, w2c: np.ndarray, focal: float) -> bool:
        """Encode a registered frame into a free slot (False when the buffer is full)."""
        if frame in self.slot_of:
            self.set_pose(frame, w2c)
            return True
        if not self.free:
            return False
        ds = CamLocDataset(rgb_files=frame, augment=True, aug_rotation=self.aug_rotation,
                           aug_scale_max=self.aug_scale, aug_scale_min=1 / self.aug_scale,
                           image_short_size=self.image_resolution, use_half=self.use_half,
                           use_heuristic_focal_length=focal <= 0)
        if focal > 0:
            ds.set_external_focal_length(focal)
        slot = self.free.pop()
        base = slot * self.per_frame
        was_training = self.net.training
        self.net.eval()
        for p in range(self.passes):
            image, mask, _, aug_inv, K, Kinv, _, _, _ = ds[0]  # a new random augmentation each call
            with autocast(enabled=self.use_half):
                f = self.net.get_features(image[None].to(self.device))  # 1, C, H, W
            _, C, H, W = f.shape
            m = torch.nn.functional.interpolate(mask[None].float(), size=(H, W), mode="nearest").view(-1)
            if m.sum() == 0:
                m = torch.ones_like(m)
            idx = torch.multinomial(m, self.samples_per_pass, replacement=True, generator=self.rng)
            rows = slice(base + p * self.samples_per_pass, base + (p + 1) * self.samples_per_pass)
            self.feat[rows] = f[0].reshape(C, -1).T[idx.to(self.device)].to(self.feat.dtype).cpu()
            self.px[rows] = self.grid[:, :H, :W].reshape(2, -1).T[idx.to(self.device)].cpu()
            self.aug_inv[rows] = aug_inv[:3].expand(self.samples_per_pass, 3, 4)
            self.K[rows] = K.expand(self.samples_per_pass, 3, 3)
            self.Kinv[rows] = Kinv.expand(self.samples_per_pass, 3, 3)
        if was_training:
            self.net.train()
        self.slot_of[frame] = slot
        self.poses[slot] = torch.as_tensor(np.asarray(w2c), dtype=torch.float32, device=self.device)
        return True

    def set_pose(self, frame: str, w2c: np.ndarray):
        slot = self.slot_of.get(frame)
        if slot is not None:
            self.poses[slot] = torch.as_tensor(np.asarray(w2c), dtype=torch.float32, device=self.device)

    def remove_frame(self, frame: str):
        slot = self.slot_of.pop(frame, None)
        if slot is not None:
            self.free.append(slot)

    # ------------------------------------------------------------------ training
    def step(self) -> Optional[float]:
        """One head iteration; returns the batch inlier ratio (10 px), None without frames."""
        if not self.slot_of:
            return None
        slots = torch.tensor(list(self.slot_of.values()))
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
        inliers, loss = self.loss.compute(
            pred_cam_coords_b31=cam, pred_scene_coords_b31=sc, reprojection_error_b1=err,
            target_crds_b3=torch.zeros_like(px[:, :1]).expand(-1, 3), target_px_b2=px,
            invKs_b33=Kinv, iteration=self.loss_iteration, max_iterations=self.loss_total)
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

    def snapshot(self) -> Regressor:
        """Frozen eval copy for localization."""
        net = copy.deepcopy(self.net)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    def head_state(self) -> dict:
        return {k: v.detach().half().cpu() for k, v in self.net.heads.state_dict().items()}
