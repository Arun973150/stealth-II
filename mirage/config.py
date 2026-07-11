"""Configuration for MIRAGE.

Every hyperparameter mentioned in the paper is surfaced here. Two ready-made
configs are provided:

* ``MirageConfig()``            -- the paper defaults (needs a GPU to be practical)
* ``DEMO_CONFIG``               -- a tiny, CPU-runnable config to verify correctness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a single surrogate encoder in the ensemble."""

    kind: Literal["openclip", "dino", "moderation"]
    name: str
    pretrained: Optional[str] = None  # open_clip pretrained tag, or timm has weights baked in

    def key(self) -> str:
        return f"{self.kind}:{self.name}:{self.pretrained or ''}"


# --- The paper's ensemble: 8 CLIP-style transformers (diverse training data) + DINOv2 ---
# Chosen to span OpenAI / LAION / DataComp / MetaCLIP / EVA training distributions so the
# perturbation transfers across many representation spaces (Sec. 4.3, Fig. 7).
DEFAULT_ENSEMBLE: List[ModelSpec] = [
    ModelSpec("openclip", "ViT-B-32", "openai"),
    ModelSpec("openclip", "ViT-B-16", "laion2b_s34b_b88k"),
    ModelSpec("openclip", "ViT-L-14", "laion2b_s32b_b82k"),
    ModelSpec("openclip", "ViT-B-32", "datacomp_xl_s13b_b90k"),
    ModelSpec("openclip", "ViT-B-16", "datacomp_l_s1b_b8k"),
    ModelSpec("openclip", "ViT-B-32-quickgelu", "metaclip_400m"),
    ModelSpec("openclip", "EVA02-B-16", "merged2b_s8b_b131k"),
    ModelSpec("openclip", "ViT-L-14", "datacomp_xl_s13b_b90k"),
    # Self-supervised encoder for more robust perturbations (image targets only).
    ModelSpec("dino", "vit_small_patch14_dinov2.lvd142m"),
    # Open-source image safety classifier (stands in for ShieldGemma); differentiable
    # unsafe-probability that we maximize directly (Sec. 4.3).
    ModelSpec("moderation", "Falconsai/nsfw_image_detection"),
]

# A minimal ensemble that downloads fast and runs on CPU for smoke-testing.
DEMO_ENSEMBLE: List[ModelSpec] = [
    ModelSpec("openclip", "ViT-B-32", "openai"),
    ModelSpec("openclip", "ViT-B-32", "laion2b_s34b_b79k"),
]

# Opt-in ensemble that adds the large moderation VLMs from the paper (Sec. 4.3). These are
# LICENSE-GATED on Hugging Face (accept Google/Meta terms + authenticate) and heavy (4B/11B),
# so they are GPU-only and NOT part of the default ensemble. ShieldGemma-2 is confirmed in
# the paper (ref [68]); Llama Guard 3 Vision is a plausible candidate for the unnamed ref [15].
VLM_MODERATOR_ENSEMBLE: List[ModelSpec] = DEFAULT_ENSEMBLE + [
    ModelSpec("shieldgemma2", "google/shieldgemma-2-4b-it"),
    ModelSpec("llamaguard_vision", "meta-llama/Llama-Guard-3-11B-Vision"),
]


@dataclass
class MirageConfig:
    # ---- Perturbation / PGD (Sec. 4.3) ----
    budget: float = 16 / 255           # L-inf bound B on delta (paper default 16/255)
    steps: int = 5000                  # PGD iterations
    step_size: Optional[float] = None  # iterated FGSM step; defaults to 2.5*budget/steps-ish
    random_init: bool = True           # start delta from a random point in the L-inf ball

    # ---- Ensemble ----
    ensemble: List[ModelSpec] = field(default_factory=lambda: list(DEFAULT_ENSEMBLE))

    # ---- Global-Local views (Eq. 2) ----
    use_global_local: bool = True
    num_patches: int = 16              # p: random full-res patches for the local view
    top_k: int = 4                     # k: keep k most-aligned patches
    lambda_local: float = 1.0          # lambda: local vs global balance

    # ---- Target concept set T (Sec. 4.2) ----
    # "sexual" is the most strongly moderated target (Fig. 9); "violence" also supported.
    target_categories: List[str] = field(default_factory=lambda: ["sexual", "violence"])
    use_text_targets: bool = True      # use text captions as targets (safe; no explicit media)
    image_targets_dir: Optional[str] = None  # optional dir of user-supplied image targets

    # ---- Augmentations (Sec. 4.3) ----
    use_augmentations: bool = True
    aug_prob: float = 0.9              # prob. of applying an augmentation on a given step

    # ---- Model dropout + secant gradient caching (Sec. 4.3) ----
    model_dropout: bool = True
    active_models_per_step: int = 4    # how many models get a real gradient each step
    use_secant_cache: bool = True      # approximate dropped-model gradients via secant rule

    # ---- Validation / checkpointing (Sec. 4.3) ----
    checkpoint_every: int = 250
    select_by_public_api: bool = False  # paper finds omitting this performs comparably/better
    openai_moderation_key: Optional[str] = None

    # ---- Runtime ----
    device: str = "auto"               # "auto" | "cpu" | "cuda"
    seed: int = 0
    log_every: int = 50
    dtype: Literal["float32", "float16"] = "float32"

    def resolved_step_size(self) -> float:
        if self.step_size is not None:
            return self.step_size
        # A standard PGD heuristic: enough total travel to cross the ball a few times.
        return max(1.0 / 255.0, 2.5 * self.budget / max(1, min(self.steps, 200)))


# CPU-friendly config: tiny ensemble, few steps, small views. For correctness checks only.
DEMO_CONFIG = MirageConfig(
    budget=16 / 255,
    steps=40,
    ensemble=list(DEMO_ENSEMBLE),
    num_patches=4,
    top_k=2,
    active_models_per_step=2,
    model_dropout=False,
    use_secant_cache=False,
    checkpoint_every=20,
    log_every=5,
    target_categories=["sexual"],
)
