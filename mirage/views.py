"""Global and Local views (Eq. 2).

Most CLIP encoders are trained at <=336px, but a practical immunizer must handle full-res
(up to 1024px) images. Naively downscaling the whole image gives spatially coarse gradients
and weak immunization (Fig. 8). The fix is two complementary views:

  * Global view  x_G = Resize(x, r): the whole image at the encoder's native resolution r,
    capturing coarse semantic structure. (This is exactly what Encoder.embed_image already
    does internally, so we don't need a separate helper for it.)

  * Local view: p patches of size r x r randomly cropped from x at its *native* resolution,
    preserving fine detail. We then keep the top-k most target-aligned patches (Eq. 2).

Patches are extracted by slicing (differentiable). If the image is smaller than r along a
dimension, we crop the largest region we can and let embed_image resize it up.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def extract_patches(x: torch.Tensor, r: int, p: int) -> torch.Tensor:
    """Randomly crop p patches of side r from a full-res image x (1,3,H,W).

    Returns a (p, 3, r, r) batch, differentiable w.r.t. x.
    """
    assert x.dim() == 4 and x.shape[0] == 1, "expected a single image (1,3,H,W)"
    _, c, h, w = x.shape
    ch, cw = min(r, h), min(r, w)  # crop size (native res when the image is big enough)
    patches = []
    for _ in range(p):
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        patch = x[:, :, top : top + ch, left : left + cw]
        if (ch, cw) != (r, r):
            patch = F.interpolate(patch, size=(r, r), mode="bicubic", align_corners=False)
        patches.append(patch)
    return torch.cat(patches, dim=0)  # (p, 3, r, r)
