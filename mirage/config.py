"""Configuration for MIRAGE.

Every hyperparameter here is set to match **Appendix B, Table 5** of the paper verbatim
("Hyper-parameters for MIRAGE used in the main experiments"), which is the authoritative
source -- it overrides the looser prose description in Sec. 4.3 where the two disagree.

Two ready-made configs are provided:

* ``MirageConfig()``            -- the paper's exact main-experiment config (needs a GPU)
* ``DEMO_CONFIG``               -- a tiny, CPU-runnable config to verify correctness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a single surrogate encoder in the ensemble."""

    kind: Literal[
        "openclip", "dino", "hf_dinov2", "hf_siglip", "moderation",
        "shieldgemma2", "llamaguard_vision",
    ]
    name: str
    pretrained: Optional[str] = None  # open_clip pretrained tag; unused by HF-loaded kinds

    def key(self) -> str:
        return f"{self.kind}:{self.name}:{self.pretrained or ''}"


# --- Table 5's exact surrogate ensemble (8 models) ---
# "hf dinov2:facebook/dinov2-base; hf siglip:google/siglip-base-patch16-224;
#  open_clip:ViT-B-32/laion2b_s34b_b79k; open_clip:ViT-B-16/laion2b_s34b_b88k;
#  open_clip:ViT-L-14/datacomp_xl_s13b_b90k; open_clip:ViT-L-14/dfn2b_s39b;
#  open_clip:ViT-H-14/dfn5b; shieldgemma:google/shieldgemma-2-4b-it: sexual"
DEFAULT_ENSEMBLE: List[ModelSpec] = [
    ModelSpec("hf_dinov2", "facebook/dinov2-base"),
    ModelSpec("hf_siglip", "google/siglip-base-patch16-224"),
    ModelSpec("openclip", "ViT-B-32", "laion2b_s34b_b79k"),
    ModelSpec("openclip", "ViT-B-16", "laion2b_s34b_b88k"),
    ModelSpec("openclip", "ViT-L-14", "datacomp_xl_s13b_b90k"),
    ModelSpec("openclip", "ViT-L-14", "dfn2b_s39b"),      # registered non-quickgelu
    ModelSpec("openclip", "ViT-H-14-quickgelu", "dfn5b"),  # dfn5b needs quickgelu activation
    ModelSpec("shieldgemma2", "google/shieldgemma-2-4b-it"),
]

# A minimal ensemble that downloads fast and runs on CPU for smoke-testing (not from the
# paper; purely for verifying the plumbing without a GPU or gated models).
DEMO_ENSEMBLE: List[ModelSpec] = [
    ModelSpec("openclip", "ViT-B-32", "openai"),
    ModelSpec("openclip", "ViT-B-32", "laion2b_s34b_b79k"),
]

# Opt-in extra: Llama Guard 3 Vision is a REAL citation in the paper (ref [15], J. Chi et al.,
# arXiv:2411.10414) named alongside ShieldGemma [68] in the Sec. 4.3 prose as an available
# open moderation model -- but Table 5's actual main-experiment ensemble does NOT include it
# (only ShieldGemma-2 is listed there). This is offered as an additional opt-in surrogate on
# top of the exact Table-5 ensemble, not a paper-verified part of the headline results.
VLM_MODERATOR_ENSEMBLE: List[ModelSpec] = DEFAULT_ENSEMBLE + [
    ModelSpec("llamaguard_vision", "meta-llama/Llama-Guard-3-11B-Vision"),
]


@dataclass
class MirageConfig:
    # ---- Perturbation / PGD (Table 5) ----
    budget: float = 16 / 255           # ||delta||_inf <= B (main result; paper also sweeps 2/4/8)
    steps: int = 5000                  # PGD iterations
    step_size: float = 1.0 / 255.0     # peak step size (Table 5: "Step size: 1.0", in /255 units)
    step_schedule: Literal["cosine", "constant"] = "cosine"  # Table 5: "Schedule: Cosine"
    random_init: bool = True           # start delta from a random point in the L-inf ball
    eot_samples: int = 1               # Table 5: "EOT samples: 1" (one augmented view per step)

    # ---- Ensemble (Table 5's exact 8-model list) ----
    ensemble: List[ModelSpec] = field(default_factory=lambda: list(DEFAULT_ENSEMBLE))

    # ---- Global-Local views (Eq. 2, Table 5) ----
    use_global_local: bool = True
    patch_size: int = 256              # Table 5: "Patch size: 256" -- ONE fixed crop size,
                                        # shared across the whole ensemble (not per-encoder).
    num_patches: int = 16               # Table 5: "Number of local patches: 16"
    top_k: int = 8                      # Table 5: "Local pooling k: 8"
    lambda_local: float = 0.25          # Table 5: "Local view weight: 0.25"

    # ---- Target concept set T (Table 5: "Objective category: Sexual") ----
    target_categories: List[str] = field(default_factory=lambda: ["sexual"])
    use_text_targets: bool = True      # use text captions as targets (safe; no explicit media)
    image_targets_dir: Optional[str] = None  # optional dir of user-supplied image targets
    local_text_only: bool = False      # image targets drive the global view only; local
                                        # patches align to text only -> avoids the attack
                                        # "painting" explicit high-res target detail.

    # ---- Spatial mask (restrict WHERE delta is applied) ----
    mask: Optional[str] = None         # None | "background" | "border:<frac>" -- keep the
                                        # subject clean by perturbing only background/edges.
    achromatic: bool = False           # force delta to be greyscale (equal RGB channels) ->
                                        # no color is painted, imprint reads as grey texture.
    perceptual: bool = False           # JND / contrast masking: weight delta by local texture
                                        # so smooth skin stays clean, noise hides in busy areas.
    perceptual_floor: float = 0.25     # min budget multiplier in smooth regions (0..1).

    # ---- Augmentations (Sec. 4.3) ----
    use_augmentations: bool = True
    aug_prob: float = 0.9              # prob. of applying an augmentation on a given step

    # ---- Model dropout + secant gradient caching (Table 5) ----
    model_dropout: bool = True
    surrogate_drop_prob: float = 0.3    # Table 5: "Surrogate drop probability: 0.3"
    min_active_surrogates: int = 3      # Table 5: "Minimum selected surrogates: 3"
    use_secant_cache: bool = True       # approximate dropped-model gradients via secant rule

    # ---- Validation / checkpointing (Sec. 4.3) ----
    checkpoint_every: int = 250
    select_by_public_api: bool = False  # paper finds omitting this performs comparably/better
    openai_moderation_key: Optional[str] = None

    # ---- Borderline early-stop (return the minimal-imprint checkpoint that already refuses) ----
    # The plain objective (Eq. 1) is *unbounded* -- PGD keeps painting the target deeper long
    # after the moderator has flipped, adding visible sexual content with no refusal benefit.
    # These knobs stop at the *boundary*: return the FIRST checkpoint whose gate metric crosses
    # a threshold, giving the least-visible perturbation that still triggers moderation. Reuses
    # the same "evaluate a moderator mid-run and keep a checkpoint" idea as Sec. 4.3, inverted
    # (earliest-crossing instead of highest-scoring). Both default to None (disabled).
    stop_at_objective: Optional[float] = None  # stop at first checkpoint with S(x) >= this.
                                                # Calibration: S~5.9 refused, S~3.4 did not.
    stop_at_violate: Optional[float] = None     # or gate on mean ShieldGemma-2 P(violate) >= this.
    gate_check_every: int = 25                  # how often (steps) to evaluate the stop gate.

    # ---- Runtime ----
    device: str = "auto"               # "auto" | "cpu" | "cuda"
    seed: int = 0
    log_every: int = 50
    dtype: Literal["float32", "float16"] = "float32"

    def step_size_at(self, t: int) -> float:
        """Cosine-scheduled step size at PGD iteration t (0-indexed), per Table 5."""
        if self.step_schedule == "constant":
            return self.step_size
        import math

        return self.step_size * 0.5 * (1.0 + math.cos(math.pi * t / max(1, self.steps)))


# CPU-friendly config: tiny ensemble, few steps, small views. For correctness checks only
# (not a paper configuration -- use MirageConfig() for the real thing on a GPU).
DEMO_CONFIG = MirageConfig(
    budget=16 / 255,
    steps=40,
    ensemble=list(DEMO_ENSEMBLE),
    patch_size=64,
    num_patches=4,
    top_k=2,
    surrogate_drop_prob=0.0,
    min_active_surrogates=2,
    model_dropout=False,
    use_secant_cache=False,
    checkpoint_every=20,
    log_every=5,
    target_categories=["sexual"],
)
