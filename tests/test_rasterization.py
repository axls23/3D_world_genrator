"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing import Optional, Tuple

import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("per_view_color", [True, False])
@pytest.mark.parametrize("sh_degree", [None, 3])
@pytest.mark.parametrize("render_mode", ["RGB", "RGB+D", "D"])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_rasterization(
    per_view_color: bool,
    sh_degree: Optional[int],
    render_mode: str,
    packed: bool,
    batch_dims: Tuple[int, ...],
):
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    C, N = 3, 10_000
    means = torch.rand(batch_dims + (N, 3), device=device)
    quats = torch.randn(batch_dims + (N, 4), device=device)
    scales = torch.rand(batch_dims + (N, 3), device=device)
    opacities = torch.rand(batch_dims + (N,), device=device)
    if per_view_color:
        if sh_degree is None:
            colors = torch.rand(batch_dims + (C, N, 3), device=device)
        else:
            colors = torch.rand(
                batch_dims + (C, N, (sh_degree + 1) ** 2, 3), device=device
            )
    else:
        if sh_degree is None:
            colors = torch.rand(batch_dims + (N, 3), device=device)
        else:
            colors = torch.rand(
                batch_dims + (N, (sh_degree + 1) ** 2, 3), device=device
            )

    width, height = 300, 200
    focal = 300.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(batch_dims + (C, -1, -1))
    viewmats = torch.eye(4, device=device).expand(batch_dims + (C, -1, -1))

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
    )

    if render_mode == "D":
        assert renders.shape == batch_dims + (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == batch_dims + (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == batch_dims + (C, height, width, 4)

    _renders, _alphas, _meta = _rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )
    torch.testing.assert_close(renders, _renders, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(alphas, _alphas, rtol=1e-4, atol=1e-4)


def _ut_scene(n: int = 2000):
    torch.manual_seed(0)
    means = torch.randn(n, 3, device=device) * 0.6 + torch.tensor([0.0, 0.0, 4.0], device=device)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
    scales = torch.full((n, 3), 0.03, device=device)
    opacities = torch.full((n,), 0.7, device=device)
    colors = torch.rand(n, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], device=device)[None]
    return means, quats, scales, opacities, colors, viewmats, Ks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("render_mode", ["RGB+ED", "RGB+D", "ED", "D"])
def test_rasterization_ut_depth_modes(render_mode: str):
    """UT + eval3d renders depth modes (the eval3d kernel is 3-channel only) and a loss on
    rendered depth reaches the means (the UT projection itself is not differentiable)."""
    from gsplat.rendering import rasterization

    means, quats, scales, opacities, colors, viewmats, Ks = _ut_scene()
    means.requires_grad_(True)
    kw = dict(width=320, height=240, packed=False, with_ut=True, with_eval3d=True)
    out, alphas, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, render_mode=render_mode, **kw
    )
    n_ch = 4 if render_mode.startswith("RGB") else 1
    assert out.shape == (1, 240, 320, n_ch)
    # One pass with all channels == channel-chunked passes
    chunked, _, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, render_mode=render_mode,
        channel_chunk=1, **kw
    )
    torch.testing.assert_close(out, chunked)
    if n_ch == 4:
        rgb, rgb_alphas, _ = rasterization(
            means, quats, scales, opacities, colors, viewmats, Ks, render_mode="RGB", **kw
        )
        torch.testing.assert_close(out[..., :3], rgb)
        torch.testing.assert_close(alphas, rgb_alphas)

    # Same depth as the (differentiable) standard rasterizer
    ref, ref_alphas, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, render_mode=render_mode,
        width=320, height=240, packed=False,
    )
    if render_mode.endswith("ED"):
        m = (alphas[..., 0] > 0.5) & (ref_alphas[..., 0] > 0.5)
        assert (out[..., -1][m] - ref[..., -1][m]).abs().median() < 0.05

    out[..., -1].mean().backward()
    assert means.grad is not None and means.grad[:, 2].abs().sum() > 0
