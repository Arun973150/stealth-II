"""Model dropout + secant gradient caching (Sec. 4.3).

Computing grad_delta phi_i for every surrogate at every PGD step is memory-intensive as the
ensemble grows. So at each step we compute *real* gradients for only a random subset of
models (model dropout) and *approximate* the gradients of the dropped models from cached
history using a one-secant quasi-Newton rule.

For a dropped model i we keep its two most recent gradients g^{t-1}, g^{t-2} and the
perturbations delta^{t-1}, delta^{t-2} at which they were computed. With the secant
direction s = delta^{t-1} - delta^{t-2}, the approximation is

    g~_i(delta^t) = g^{t-1} + (g^{t-1} - g^{t-2}) * <s, delta^t - delta^{t-1}> / ||s||^2

The second term is a rank-1 curvature correction: it rescales the most recent gradient
*change* by how far the current iterate has moved along the previous secant direction. This
is a lightweight quasi-Newton prediction requiring only two cached gradients and their
evaluation points (cf. classical secant / quasi-Newton and incremental-gradient methods).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional

import torch


class SecantGradientCache:
    """Per-model cache of the two most recent (delta, gradient) pairs."""

    def __init__(self, keys: List[str], eps: float = 1e-12):
        self._store: Dict[str, deque] = {k: deque(maxlen=2) for k in keys}
        self._eps = eps

    def update(self, key: str, delta: torch.Tensor, grad: torch.Tensor) -> None:
        self._store[key].append((delta.detach().clone(), grad.detach().clone()))

    def approximate(self, key: str, delta_now: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the secant-approximated gradient for `key`, or None if no history yet."""
        entries = self._store[key]
        if len(entries) == 0:
            return None
        if len(entries) == 1:
            # Only one past gradient: best we can do is reuse it (zeroth-order hold).
            return entries[-1][1]
        (d_prev2, g_prev2), (d_prev1, g_prev1) = entries[0], entries[1]
        s = d_prev1 - d_prev2
        denom = (s * s).sum()
        if denom < self._eps:
            return g_prev1
        coeff = torch.dot((delta_now - d_prev1).flatten(), s.flatten()) / denom
        return g_prev1 + (g_prev1 - g_prev2) * coeff


def sample_active(keys: List[str], k: int) -> List[str]:
    """Choose k models to receive a real gradient this step (model dropout)."""
    if k >= len(keys):
        return list(keys)
    return random.sample(keys, k)
