"""CLI: immunize an image with MIRAGE.

Examples
--------
Full run (needs a GPU to be practical):
    python scripts/immunize.py --input photo.jpg --output photo_immunized.png

CPU smoke test (tiny ensemble, few steps):
    python scripts/immunize.py --input photo.jpg --output out.png --demo

Options let you sweep the budget, steps, target category, and ensemble knobs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make the `mirage` package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mirage import DEMO_CONFIG, MirageConfig, immunize  # noqa: E402
from mirage.config import DEMO_ENSEMBLE, VLM_MODERATOR_ENSEMBLE  # noqa: E402
from mirage.utils import load_image, save_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIRAGE image immunization")
    p.add_argument("--input", required=True, help="path to source image")
    p.add_argument("--output", required=True, help="path to write the immunized image")
    p.add_argument("--demo", action="store_true", help="tiny CPU-friendly config")
    p.add_argument("--budget", type=float, default=None, help="L-inf budget in /255 units")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-patches", type=int, default=None,
                   help="local-view patches per step (Table 5: 16). Fewer = faster.")
    p.add_argument("--lambda-local", type=float, default=None,
                   help="weight of the local-patch alignment (Table 5: 0.25). Lower = less "
                        "high-res 'painting' of target content (cleaner), still triggers "
                        "moderation via the global + ShieldGemma signal.")
    p.add_argument("--local-text-only", action="store_true",
                   help="use ONLY text targets for the local patches (image targets stay on "
                        "the global view). Removes explicit high-res painting from the "
                        "perturbation while keeping the refusal signal.")
    p.add_argument("--fast", action="store_true",
                   help="speed preset: steps=2500, num_patches=8, top_k=4 (~4x faster; "
                        "ε=16/255 still gives ~100%% immunization per Table 4)")
    p.add_argument("--categories", nargs="+", default=None,
                   help="target categories, e.g. sexual violence")
    p.add_argument("--image-targets", default=None, help="dir of image targets (optional)")
    p.add_argument("--no-global-local", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--vlm-moderators", action="store_true",
                   help="add Llama Guard 3 Vision on top of the default ensemble "
                        "(ShieldGemma-2 is already default per Table 5); "
                        "GPU-only, license-gated on HF; opt-in")
    p.add_argument("--openai-validate", default=None, metavar="API_KEY",
                   help="enable public-moderation-API checkpoint selection (Sec 4.3)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> MirageConfig:
    cfg = DEMO_CONFIG if args.demo else MirageConfig()
    # Copy so we don't mutate the module-level DEMO_CONFIG.
    cfg = MirageConfig(**{**cfg.__dict__})
    if args.demo:
        cfg.ensemble = list(DEMO_ENSEMBLE)
    if args.fast:
        cfg.steps = 2500
        cfg.num_patches = 8
        cfg.top_k = 4
    if args.budget is not None:
        cfg.budget = args.budget if args.budget < 1 else args.budget / 255.0
    if args.steps is not None:
        cfg.steps = args.steps
    if args.num_patches is not None:
        cfg.num_patches = args.num_patches
        cfg.top_k = min(cfg.top_k, args.num_patches)
    if args.lambda_local is not None:
        cfg.lambda_local = args.lambda_local
    if args.local_text_only:
        cfg.local_text_only = True
    if args.categories is not None:
        cfg.target_categories = args.categories
    if args.image_targets is not None:
        cfg.image_targets_dir = args.image_targets
    if args.vlm_moderators:
        cfg.ensemble = list(VLM_MODERATOR_ENSEMBLE)
    if args.no_global_local:
        cfg.use_global_local = False
    if args.no_augment:
        cfg.use_augmentations = False
    if args.openai_validate:
        cfg.select_by_public_api = True
        cfg.openai_moderation_key = args.openai_validate
    cfg.device = args.device
    cfg.seed = args.seed
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    img = load_image(args.input)
    print(f"[MIRAGE] loaded {args.input}  shape={tuple(img.shape)}  "
          f"budget={cfg.budget * 255:.1f}/255  steps={cfg.steps}  "
          f"ensemble={len(cfg.ensemble)} models")

    t0 = time.time()
    result = immunize(img, cfg)
    dt = time.time() - t0

    save_image(result.image, args.output)
    print("-" * 60)
    print(f"[MIRAGE] done in {dt:.1f}s")
    print(f"  objective S(x_hat) = {result.objective:.4f}")
    print(f"  L-inf perturbation = {result.linf_255:.1f}/255")
    print(f"  PSNR               = {result.psnr_db:.2f} dB")
    print(f"  saved -> {args.output}")


if __name__ == "__main__":
    main()
