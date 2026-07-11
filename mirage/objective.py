"""The MIRAGE objective (Eq. 1 & 2).

Every surrogate exposes a unified per-view score ``view_score(view) -> (B,)`` (ensemble.py):
for CLIP/SigLIP/DINOv2 it is the mean cosine similarity to the target set; for a moderation
model it is the differentiable "unsafe" probability. Both are quantities we want to *maximize*.

With global + local views (Eq. 2), the per-surrogate score is

    S_i(x) = s_i(x_G) + lambda * mean_{j in top-k} s_i(x_L^{(j)})

where s_i is view_score for surrogate i, x_G is the whole image at that surrogate's own
native resolution, and x_L are ``num_patches`` patches shared across the whole ensemble,
cropped once per step at a fixed size (Table 5: patch_size=256) and resized internally by
each encoder's own preprocessing. The full objective (Eq. 1) sums over the ensemble:

    S(x) = sum_i S_i(x)      (maximized)
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .config import MirageConfig
from .ensemble import Encoder
from .views import extract_patches


def sample_shared_patches(x: torch.Tensor, cfg: MirageConfig) -> Optional[torch.Tensor]:
    """Sample the ONE set of local-view patches used by every encoder this step (Table 5:
    a fixed 256x256 crop size, shared across the ensemble, not per-encoder resolution)."""
    if not cfg.use_global_local:
        return None
    return extract_patches(x, cfg.patch_size, cfg.num_patches)


def encoder_score(
    enc: Encoder, x: torch.Tensor, cfg: MirageConfig, patches: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Per-surrogate score S_i(x); differentiable scalar. x is (1,3,H,W) in [0,1].

    ``patches`` is the shared (num_patches,3,patch_size,patch_size) local-view batch (see
    ``sample_shared_patches``); each encoder resizes it to its own resolution internally.
    """
    phi_g = enc.view_score(x).mean()  # global view (B=1)
    # Heavy VLM moderators run global-only (patch forwards through a 4B/11B model per step
    # would be prohibitive); enc.use_local gates this.
    if patches is None or not getattr(enc, "use_local", True):
        return phi_g
    phi_p = enc.view_score(patches)                                # (p,)
    k = min(cfg.top_k, phi_p.numel())
    topk_mean = torch.topk(phi_p, k).values.mean()
    return phi_g + cfg.lambda_local * topk_mean


def total_objective(encoders: List[Encoder], x: torch.Tensor, cfg: MirageConfig) -> torch.Tensor:
    """Full ensemble objective S(x) = sum_i S_i(x)."""
    patches = sample_shared_patches(x, cfg)
    return torch.stack([encoder_score(enc, x, cfg, patches) for enc in encoders]).sum()
