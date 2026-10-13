"""Latent-loss helpers (hypersplat/stream/latent.py): CPU-only, no encoder weights needed."""

import pytest
import torch
import torch.nn.functional as F

from hypersplat.stream.latent import LatentEncoder, latent_cosine_loss, latent_lambda


def test_lambda_warmup_hold_and_anneal():
    assert latent_lambda(0, 0.1, warmup=100, anneal_end=1000) == 0.0
    assert latent_lambda(100, 0.1, warmup=100, anneal_end=1000) == pytest.approx(0.1)
    assert latent_lambda(550, 0.1, warmup=100, anneal_end=1000) == pytest.approx(0.05)
    assert latent_lambda(1000, 0.1, warmup=100, anneal_end=1000) == 0.0
    # anneal_end <= 0 holds lambda (arm B)
    assert latent_lambda(5000, 0.1, warmup=100, anneal_end=0) == pytest.approx(0.1)


def test_cosine_loss_zero_for_identical_and_stops_target_grad():
    pred = F.normalize(torch.randn(2, 16, 8), dim=-1).requires_grad_(True)
    target = pred.detach().clone().requires_grad_(True)
    loss = latent_cosine_loss(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    opposite = latent_cosine_loss(pred, -target)
    assert opposite.item() == pytest.approx(2.0, abs=1e-5)
    opposite.backward()
    assert pred.grad is not None and target.grad is None


def test_cosine_loss_weight_masks_patches():
    pred = F.normalize(torch.randn(1, 4, 8), dim=-1)
    target = pred.clone()
    target[:, 2:] = -target[:, 2:]
    w = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert latent_cosine_loss(pred, target, w).item() == pytest.approx(0.0, abs=1e-6)
    assert latent_cosine_loss(pred, target).item() == pytest.approx(1.0, abs=1e-5)


def test_prep_resizes_to_patch_multiple():
    enc = LatentEncoder(input_side=224, device="cpu")
    x = enc._prep(torch.rand(1, 3, 270, 480))
    assert x.shape[2] % 14 == 0 and x.shape[3] % 14 == 0
    assert max(x.shape[2:]) in (224, 224 + 7, 224 - 7)
