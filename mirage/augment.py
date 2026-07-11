"""Stochastic augmentations for robust perturbations (Sec. 4.3).

Following prior work on robust adversarial examples, we apply a randomly-drawn augmentation
alpha ~ A to (x + delta) before passing it to each surrogate similarity phi_i, i.e. we
optimize phi_i(alpha(x + delta), T). This makes the perturbation survive real-world
transforms (a weak adversary applying blur/JPEG/resize, per Fig. 5).

A = {random crop, resize, gaussian blur, JPEG compression, horizontal flip}. One
augmentation is sampled per step. JPEG is non-differentiable, so we use a straight-through
estimator: the forward pass sees the compressed image, the backward pass sees identity.
"""

from __future__ import annotations

import io
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

AUGMENTATIONS = ("flip", "resize", "crop", "blur", "jpeg")


def apply_random_augmentation(x: torch.Tensor, prob: float = 0.9) -> torch.Tensor:
    """With probability `prob`, apply one random augmentation to x (1,3,H,W) in [0,1]."""
    if random.random() > prob:
        return x
    aug = random.choice(AUGMENTATIONS)
    if aug == "flip":
        return torch.flip(x, dims=[-1])
    if aug == "resize":
        return _random_resize(x)
    if aug == "crop":
        return _random_crop(x)
    if aug == "blur":
        return _gaussian_blur(x)
    if aug == "jpeg":
        return _straight_through(x, _jpeg)
    return x


def _random_resize(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    scale = random.uniform(0.5, 1.0)
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)


def _random_crop(x: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    scale = random.uniform(0.7, 1.0)
    ch, cw = max(8, int(h * scale)), max(8, int(w * scale))
    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)
    return x[:, :, top : top + ch, left : left + cw]


def _gaussian_blur(x: torch.Tensor, ksize: int = 5) -> torch.Tensor:
    sigma = random.uniform(0.1, 2.0)
    coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)  # (k,k)
    kernel = kernel_2d.expand(3, 1, ksize, ksize).contiguous()
    pad = ksize // 2
    xpad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(xpad, kernel, groups=3)


def _jpeg(x: torch.Tensor) -> torch.Tensor:
    """Actual (non-differentiable) JPEG round-trip via PIL; returns same-shape tensor."""
    quality = random.randint(40, 90)
    arr = (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    dec = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    out = torch.from_numpy(dec).permute(2, 0, 1).unsqueeze(0).to(x.device, x.dtype)
    return out


def _straight_through(x: torch.Tensor, fn) -> torch.Tensor:
    """Forward = fn(x); backward = identity (STE for non-differentiable fn)."""
    y = fn(x.detach())
    return x + (y - x.detach())
