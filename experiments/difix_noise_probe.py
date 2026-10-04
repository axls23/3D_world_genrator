"""Difix output noise on one input: repeated calls with sampled vs posterior-mean latents.

Prints the PSNR between repeated outputs (inf = identical) and between the two modes, on a
degraded drone frame (downscale/upscale + noise, as a stand-in for a splat render) with the
clean neighbouring frame as reference.
"""
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from hypersplat.beta.difix.augment import DifixFixer, change_psnr  # noqa: E402

cap = cv2.VideoCapture("demo/input/drone_demo.mp4")
frames = []
for i in range(60):
    ok, f = cap.read()
    if ok and i in (40, 48):
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
h, w = frames[0].shape[:2]
s = 960 / max(h, w)
img, ref = [cv2.resize(f, (int(w * s) // 8 * 8, int(h * s) // 8 * 8), interpolation=cv2.INTER_AREA) for f in frames]
rng = np.random.default_rng(0)
small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
render = np.clip(cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                 + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)

fx = DifixFixer("nvidia/difix_ref", "cuda")
for scale in (0.5, 1.0):
    out = {}
    for mode in (False, True):
        fx.deterministic = mode
        torch.manual_seed(1)
        a = fx(render, ref, scale=scale)
        torch.manual_seed(2)
        b = fx(render, ref, scale=scale)
        c = fx(render, ref, scale=scale, seed=7)
        d = fx(render, ref, scale=scale, seed=7)
        out[mode] = a
        print(f"scale {scale} {'mean' if mode else 'sample'}: repeat PSNR {change_psnr(a, b):.2f} dB, "
              f"seeded repeat {change_psnr(c, d):.2f} dB, change vs input {change_psnr(render, a):.2f} dB, "
              f"vs clean frame {change_psnr(img, a):.2f} dB")
    print(f"scale {scale}: sample vs mean {change_psnr(out[False], out[True]):.2f} dB; "
          f"input vs clean {change_psnr(img, render):.2f} dB")
