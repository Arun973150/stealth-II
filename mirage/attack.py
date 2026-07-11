"""MIRAGE immunizer: PGD / iterated FGSM over the ensemble objective (Sec. 4).

We construct the immunized image x_hat = x + delta by maximizing the ensemble objective
S(x + delta) (objective.py) under an L-inf budget ||delta||_inf <= B, using:

  * iterated FGSM steps (sign gradient ascent),
  * random augmentation per step (augment.py),
  * global + local views (views.py / objective.py),
  * model dropout + secant gradient caching (secant.py),
  * optional checkpoint selection via a public moderation API (off by default; the paper
    finds omitting it performs comparably or better).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from .augment import apply_random_augmentation
from .config import MirageConfig
from .ensemble import Encoder, attach_targets, build_ensemble
from .objective import encoder_score, total_objective
from .secant import SecantGradientCache, sample_active
from .targets import build_target_set
from .utils import (
    linf,
    openai_moderation_score,
    psnr,
    resolve_device,
    seed_everything,
)


@dataclass
class ImmunizeResult:
    image: torch.Tensor          # immunized image (1,3,H,W) in [0,1]
    delta: torch.Tensor          # the applied perturbation
    objective: float             # final objective value S(x_hat)
    linf_255: float              # perturbation L-inf in 0-255 units
    psnr_db: float               # PSNR vs original
    history: List[float]         # objective per logged step


def _project(x0: torch.Tensor, delta: torch.Tensor, budget: float) -> torch.Tensor:
    """Project delta to the L-inf ball and keep x0+delta a valid image in [0,1]."""
    delta = torch.clamp(delta, -budget, budget)
    delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0
    return delta


def _ensemble_gradient(
    encoders: List[Encoder],
    keys: List[str],
    x0: torch.Tensor,
    delta: torch.Tensor,
    cfg: MirageConfig,
    cache: Optional[SecantGradientCache],
):
    """Summed gradient d S / d delta with model dropout + secant approximation."""
    delta = delta.detach().requires_grad_(True)
    x_adv = x0 + delta
    x_in = apply_random_augmentation(x_adv, cfg.aug_prob) if cfg.use_augmentations else x_adv

    active = sample_active(keys, cfg.active_models_per_step) if cfg.model_dropout else list(keys)
    active_set = set(active)

    total_grad = torch.zeros_like(delta)
    obj_val = 0.0
    for enc in encoders:
        if enc.key in active_set:
            score = encoder_score(enc, x_in, cfg)
            g = torch.autograd.grad(score, delta, retain_graph=True)[0]
            total_grad = total_grad + g
            obj_val += float(score.detach())
            if cache is not None:
                cache.update(enc.key, delta, g)
        elif cache is not None:
            g_approx = cache.approximate(enc.key, delta.detach())
            if g_approx is not None:
                total_grad = total_grad + g_approx
    return total_grad.detach(), obj_val


@torch.no_grad()
def _full_objective(encoders: List[Encoder], x0: torch.Tensor, delta: torch.Tensor, cfg) -> float:
    return float(total_objective(encoders, (x0 + delta).clamp(0, 1), cfg).item())


def immunize(
    image: torch.Tensor,
    cfg: Optional[MirageConfig] = None,
    encoders: Optional[List[Encoder]] = None,
    progress: Optional[Callable[[int, float], None]] = None,
) -> ImmunizeResult:
    """Immunize a single image.

    Args:
        image: (1,3,H,W) or (3,H,W) float tensor in [0,1].
        cfg: MirageConfig (defaults to the paper config).
        encoders: pre-built ensemble (with targets attached). If None, built from cfg.
        progress: optional callback(step, objective) for custom logging.
    """
    cfg = cfg or MirageConfig()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    if image.dim() == 3:
        image = image.unsqueeze(0)
    x0 = image.to(device).clamp(0, 1)

    # ---- build ensemble + attach targets (unless a prepared one is provided) ----
    if encoders is None:
        base = build_ensemble(cfg.ensemble, device)
        targets = build_target_set(
            cfg.target_categories, cfg.use_text_targets, cfg.image_targets_dir
        )
        encoders = attach_targets(base, targets)
    keys = [e.key for e in encoders]

    cache = SecantGradientCache(keys) if cfg.use_secant_cache else None

    # ---- initialize delta ----
    if cfg.random_init:
        delta = (torch.rand_like(x0) * 2 - 1) * cfg.budget
    else:
        delta = torch.zeros_like(x0)
    delta = _project(x0, delta, cfg.budget)

    step_size = cfg.resolved_step_size()
    history: List[float] = []
    # Public-API checkpoint selection state (Sec. 4.3); only used if enabled + key present.
    api_enabled = cfg.select_by_public_api and bool(cfg.openai_moderation_key)
    best_ckpt = {"delta": delta.clone(), "api_score": -1.0}

    for t in range(cfg.steps):
        grad, _ = _ensemble_gradient(encoders, keys, x0, delta, cfg, cache)
        delta = delta + step_size * grad.sign()
        delta = _project(x0, delta, cfg.budget)

        # Checkpoint every `checkpoint_every` steps: optionally score the checkpoint under
        # the public moderator C-tilde and keep the highest-scoring one (early stopping).
        if api_enabled and (t + 1) % cfg.checkpoint_every == 0:
            try:
                score = openai_moderation_score(
                    (x0 + delta).clamp(0, 1), cfg.openai_moderation_key
                )
                if score > best_ckpt["api_score"]:
                    best_ckpt = {"delta": delta.clone(), "api_score": score}
            except Exception as e:  # noqa: BLE001 - never let API hiccups kill the run
                print(f"[MIRAGE] public moderation API error at step {t + 1}: {e!r}")

        if (t + 1) % cfg.log_every == 0 or t == cfg.steps - 1:
            obj = _full_objective(encoders, x0, delta, cfg)
            history.append(obj)
            if progress is not None:
                progress(t + 1, obj)
            else:
                print(f"[MIRAGE] step {t + 1}/{cfg.steps}  objective={obj:.4f}")

    # Selection: by default return the final iterate (paper finds this comparable or better
    # than public-API checkpoint selection). Otherwise return the best C-tilde checkpoint.
    chosen = best_ckpt["delta"] if api_enabled else delta
    x_hat = (x0 + chosen).clamp(0, 1)
    return ImmunizeResult(
        image=x_hat.detach().cpu(),
        delta=chosen.detach().cpu(),
        objective=_full_objective(encoders, x0, chosen, cfg),
        linf_255=linf(x_hat, x0),
        psnr_db=psnr(x_hat, x0),
        history=history,
    )
