"""The MIRAGE objective (Eq. 1 & 2).

Every surrogate exposes a unified per-view score ``view_score(view) -> (B,)`` (ensemble.py):
for CLIP/DINO it is the mean cosine similarity to the target set; for a moderation model it
is the differentiable "unsafe" probability. Both are quantities we want to *maximize*.

With global + local views (Eq. 2), the per-surrogate score is

    S_i(x) = s_i(x_G) + lambda * mean_{j in top-k} s_i(x_L^{(j)})

where s_i is view_score for surrogate i, x_G is the whole image at native resolution, and
x_L are p native-resolution patches. The full objective (Eq. 1) sums over the ensemble:

    S(x) = sum_i S_i(x)      (maximized)
"""

from __future__ import annotations

from typing import List

import torch

from .config import MirageConfig
from .ensemble import Encoder
from .views import extract_patches


def encoder_score(enc: Encoder, x: torch.Tensor, cfg: MirageConfig) -> torch.Tensor:
    """Per-surrogate score S_i(x); differentiable scalar. x is (1,3,H,W) in [0,1]."""
    phi_g = enc.view_score(x).mean()  # global view (B=1)
    # Heavy VLM moderators run global-only (patch forwards through a 4B/11B model per step
    # would be prohibitive); enc.use_local gates this.
    if not cfg.use_global_local or not getattr(enc, "use_local", True):
        return phi_g
    patches = extract_patches(x, enc.resolution, cfg.num_patches)  # (p,3,r,r)
    phi_p = enc.view_score(patches)                                # (p,)
    k = min(cfg.top_k, phi_p.numel())
    topk_mean = torch.topk(phi_p, k).values.mean()
    return phi_g + cfg.lambda_local * topk_mean


def total_objective(encoders: List[Encoder], x: torch.Tensor, cfg: MirageConfig) -> torch.Tensor:
    """Full ensemble objective S(x) = sum_i S_i(x)."""
    return torch.stack([encoder_score(enc, x, cfg) for enc in encoders]).sum()
