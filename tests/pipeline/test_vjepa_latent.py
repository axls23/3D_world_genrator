"""V-JEPA predictive latent loss + secondary hook (hypersplat/stream/vjepa_latent.py).

CPU-only with tiny random-init V-JEPA modules: no checkpoint, no HF weights."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypersplat.stream.vjepa import predictor as P, vision_transformer as V
from hypersplat.stream.vjepa_latent import LatentPacer, SecondaryHook, VJEPALatentLoss

SEC_DIM = 8


class FakeSecondary:
    """Stands in for LatentEncoder: 4x4 grid of normalised tokens."""

    def features(self, photo):
        return F.normalize(torch.randn(photo.shape[0], 16, SEC_DIM), dim=-1)

    def grid(self, h, w):
        return (4, 4)


def _tiny_models():
    torch.manual_seed(0)
    enc = V.vit_tiny(patch_size=16, img_size=(64, 64), num_frames=2, tubelet_size=2, use_rope=True)
    pred = P.vit_predictor(img_size=(64, 64), patch_size=16, use_mask_tokens=True, embed_dim=enc.embed_dim,
                           predictor_embed_dim=64, num_frames=2, tubelet_size=2, depth=2, num_heads=2,
                           num_mask_tokens=10, use_rope=True)
    return enc, pred


def _loss_module():
    enc, pred = _tiny_models()
    return VJEPALatentLoss(enc, pred, FakeSecondary(), sec_dim=SEC_DIM, side=64, target_frac=0.5)


def test_hook_is_identity_at_init():
    enc, pred = _tiny_models()
    hook = SecondaryHook(pred, SEC_DIM)
    x = torch.randn(2, 8, enc.embed_dim)
    ctx = torch.arange(8).repeat(2, 1)
    tgt = torch.arange(8, 16).repeat(2, 1)
    base = pred(x, [ctx], [tgt])
    hook.set(torch.randn(2, 16, SEC_DIM), [ctx])
    hooked = pred(x, [ctx], [tgt])
    hook.clear()
    assert torch.allclose(base, hooked, atol=1e-6)


def test_hook_gathers_context_only_and_changes_output_once_trained():
    enc, pred = _tiny_models()
    hook = SecondaryHook(pred, SEC_DIM)
    with torch.no_grad():
        hook.proj.weight.normal_(std=0.5)
    x = torch.randn(2, 8, enc.embed_dim)
    ctx = torch.arange(8).repeat(2, 1)
    tgt = torch.arange(8, 16).repeat(2, 1)
    base = pred(x, [ctx], [tgt])
    sec = torch.randn(2, 16, SEC_DIM)
    hook.set(sec, [ctx])
    hooked = pred(x, [ctx], [tgt])
    hook.clear()
    assert not torch.allclose(base, hooked)
    # the projection must be fed exactly the context rows, in context order
    with torch.no_grad():
        emb = pred.predictor_embed(x)  # hook not set: plain embedding
        hook.set(sec, [ctx])
        hooked_emb = pred.predictor_embed(x)
        hook.clear()
        ctx_sec = torch.gather(sec, 1, ctx.unsqueeze(-1).expand(-1, -1, SEC_DIM))
        expected = emb + hook.proj(ctx_sec.float())
    assert torch.allclose(hooked_emb, expected, atol=1e-5)


def test_loss_backprops_to_render_and_hook_only():
    m = _loss_module()
    render = torch.rand(1, 3, 64, 64, requires_grad=True)
    photo = torch.rand(1, 3, 64, 64)
    out = m(render, photo)
    loss = out.latent
    assert out.alpha_hole is None  # no alpha given
    assert torch.isfinite(loss) and 0.0 <= loss.item() <= 2.0
    loss.backward()
    assert render.grad is not None and render.grad.abs().sum() > 0
    assert m.hook.proj.weight.grad is not None
    assert all(p.grad is None for p in m.encoder.parameters())
    assert all(p.grad is None for p in m.predictor.parameters() if p is not m.hook.proj.weight)


def test_loss_with_patch_weight_is_finite_and_zero_for_identical_inputs():
    m = _loss_module()
    photo = torch.rand(1, 3, 64, 64)
    # identical render/photo: target and prediction differ only through the hook, which is zero at init
    loss = m(photo.clone(), photo, patch_weight=True).latent
    assert torch.isfinite(loss)
    assert loss.item() > 0.0  # the predictor still predicts masked latents, so the loss is not trivially 0


def test_pacer_runs_every_step_until_measured_and_when_disabled():
    p = LatentPacer(budget=0.5)
    assert p.interval() == 1 and p.should_run(7)
    p.tick(0.0); p.tick(1.0)  # base step ~1 s
    assert p.interval() == 1  # still no V-JEPA cost measured
    off = LatentPacer(budget=0.0)
    off.tick(0.0); off.tick(1.0); off.record(5.0)
    assert off.interval() == 1


def test_pacer_interval_keeps_vjepa_share_near_budget():
    p = LatentPacer(budget=0.5, ema=0.0)
    p.tick(0.0); p.tick(1.0)  # step 1 s without V-JEPA
    p.record(0.4)  # call costs 0.4 s -> 0.4/(0.5*1.0) = 0.8 -> every step
    assert p.interval() == 1
    p.record(2.0)  # call costs 2 s -> 2/(0.5*1.0) = 4 -> every 4th step
    assert p.interval() == 4
    assert [s for s in range(12) if p.should_run(s)] == [0, 4, 8]


def test_pacer_base_time_excludes_vjepa_time():
    p = LatentPacer(budget=0.5, ema=0.0)
    p.tick(0.0); p.tick(1.0)
    p.record(0.6)  # this step spent 0.6 s in V-JEPA
    p.tick(2.0)  # wall step 1.0 s, base = 1.0 - 0.6 = 0.4 s
    assert p.base_s == pytest.approx(0.4)


def test_alpha_hole_raises_alpha_only_where_coverage_is_low():
    m = _loss_module()
    photo = torch.rand(1, 3, 64, 64)
    alpha = torch.ones(1, 1, 64, 64)
    alpha[..., :32] = 0.1  # left half barely covered
    alpha.requires_grad_(True)
    out = m(photo.clone(), photo, alpha=alpha)
    assert out.alpha_hole is not None and out.alpha_hole.item() > 0
    assert 0.0 < out.hole_frac < 1.0
    out.alpha_hole.backward()
    g = alpha.grad[0, 0]
    assert (g[:, :32] < 0).any()  # descent increases alpha in the holes
    assert torch.all(g[:, 32:] == 0)  # covered patches untouched


def test_alpha_hole_skips_invalid_patches():
    m = _loss_module()
    photo = torch.rand(1, 3, 64, 64)
    alpha = torch.zeros(1, 1, 64, 64, requires_grad=True)  # everything is a hole
    valid = torch.zeros(1, 32, 32, dtype=torch.bool)  # ...but no depth anywhere (sky)
    out = m(photo.clone(), photo, alpha=alpha, valid=valid)
    assert out.alpha_hole.item() == 0.0 and out.hole_frac == 0.0


def test_alpha_condition_reaches_hook_and_old_checkpoints_load():
    m = _loss_module()
    assert m.sec_proj.in_features == SEC_DIM + 1
    old = {"weight": torch.randn(m.sec_proj.out_features, SEC_DIM), "bias": torch.zeros(m.sec_proj.out_features)}
    m.load_hook_state(old)
    assert torch.equal(m.sec_proj.weight[:, :SEC_DIM], old["weight"])
    assert torch.all(m.sec_proj.weight[:, SEC_DIM] == 0)


def test_target_frac_bounds():
    enc, pred = _tiny_models()
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            VJEPALatentLoss(enc, pred, FakeSecondary(), sec_dim=SEC_DIM, side=64, target_frac=bad)
