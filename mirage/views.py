"""Global and Local views (Eq. 2).

Most CLIP encoders are trained at <=336px, but a practical immunizer must handle full-res
(up to 1024px) images. Naively downscaling the whole image gives spatially coarse gradients
and weak immunization (Fig. 8). The fix is two complementary views:

  * Global view  x_G = Resize(x, r): the whole image at each encoder's own native resolution
    r, capturing coarse semantic structure. (This is exactly what Encoder.embed_image already
    does internally, so we don't need a separate helper for it.)

  * Local view: p patches cropped from x at a **fixed** size (Table 5: "Patch size: 256"),
    sampled *once per step and shared across the entire ensemble* -- every surrogate scores
    the same 16 regions of the image, rather than each encoder sampling its own crops at its
    own resolution. Each encoder's own preprocessing then resizes the shared 256x256 patches
    to whatever input size it needs internally. We keep the top-k most target-aligned patches
    per encoder (Eq. 2).

Patches are extracted by slicing (differentiable). If the image is smaller than the patch
size along a dimension, we crop the largest region we can and resize it up.
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
