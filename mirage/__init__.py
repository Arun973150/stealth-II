"""MIRAGE — Moderation Induced Resistance Against Generative Editing.

A faithful, from-scratch reproduction of the MIRAGE image-immunization method.

The core idea: instead of attacking a black-box image *editor* (impossible without
weights), we craft an imperceptible perturbation that makes the editor's *safety
moderator* flag the image as policy-violating, triggering a prompt-agnostic refusal.

We do this by maximizing the cosine similarity between the (perturbed) source image
and a set of "unsafe" concept targets, across an ensemble of open-source encoders
(CLIP-style + self-supervised), using global + local image views, augmentations,
and PGD with model-dropout + secant gradient caching.

Public API:
    >>> from mirage import immunize, MirageConfig
    >>> cfg = MirageConfig(budget=16/255, steps=5000)
    >>> immunized = immunize(image, cfg)
"""

from .config import MirageConfig, DEMO_CONFIG
from .attack import immunize

__all__ = ["MirageConfig", "DEMO_CONFIG", "immunize"]
__version__ = "0.1.0"
