"""Evaluation harness for MIRAGE.

Since freely hammering commercial editing APIs is costly, this harness reports a local
**moderation proxy**: the mean cosine similarity of an image to the unsafe target set across
the surrogate ensemble. A successful immunization should sharply raise this proxy relative
to the original -- that increase is exactly what transfers to the black-box moderator C.

It also measures weak-adversary robustness (Fig. 5): re-score the immunized image after
classical transforms (blur, JPEG, resize-down, greyscale, screenshot) and report how much
of the proxy gain survives.

Optionally, if you pass --openai-key, it queries OpenAI's public omni-moderation endpoint
on the original and immunized images as an *external* black-box sanity check. (This is the
public C-tilde in the paper, distinct from any editor's internal C.)
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import sys
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mirage.config import DEMO_ENSEMBLE, MirageConfig  # noqa: E402
from mirage.ensemble import attach_targets, build_ensemble  # noqa: E402
from mirage.objective import total_objective  # noqa: E402
from mirage.targets import build_target_set  # noqa: E402
from mirage.utils import load_image, resolve_device  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
import numpy as np  # noqa: E402


def moderation_proxy(encoders, image: torch.Tensor, device) -> float:
    """Mean global cosine similarity to targets across the ensemble (higher = 'unsafer')."""
    cfg = MirageConfig(use_global_local=False)
    with torch.no_grad():
        s = total_objective(encoders, image.to(device).clamp(0, 1), cfg).item()
    return s / max(1, len(encoders))


# ---- weak-adversary transforms (operate on a (1,3,H,W) tensor, return one) ----
def _to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def t_blur(t):       return _to_tensor(_to_pil(t).filter(ImageFilter.GaussianBlur(radius=1)))
def t_grey(t):       return _to_tensor(_to_pil(t).convert("L").convert("RGB"))
def t_resize(t):
    im = _to_pil(t); w, h = im.size
    return _to_tensor(im.resize((w // 2, h // 2)).resize((w, h)))
def t_jpeg(t):
    buf = io.BytesIO(); _to_pil(t).save(buf, format="JPEG", quality=50); buf.seek(0)
    return _to_tensor(Image.open(buf))
def t_screenshot(t):
    im = _to_pil(t); long = max(im.size); s = 1080 / long
    im = im.resize((int(im.size[0] * s), int(im.size[1] * s))).filter(ImageFilter.GaussianBlur(0.6))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=75); buf.seek(0)
    return _to_tensor(Image.open(buf))


WEAK_ADVERSARIES = {
    "gaussian_blur": t_blur, "greyscale": t_grey, "resize_down": t_resize,
    "jpeg_q50": t_jpeg, "screenshot": t_screenshot,
}


def openai_moderation(image_path: str, api_key: str) -> dict:
    """Query OpenAI's public omni-moderation endpoint on an image (opt-in)."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({
        "model": "omni-moderation-latest",
        "input": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/moderations", data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    scores = res["results"][0]["category_scores"]
    return {"flagged": res["results"][0]["flagged"], "max_score": max(scores.values())}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MIRAGE immunization")
    p.add_argument("--original", required=True)
    p.add_argument("--immunized", required=True)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--categories", nargs="+", default=["sexual"])
    p.add_argument("--robustness", action="store_true", help="run weak-adversary transforms")
    p.add_argument("--openai-key", default=None, help="opt-in external moderation check")
    args = p.parse_args()

    device = resolve_device("auto")
    ensemble_spec = list(DEMO_ENSEMBLE) if args.demo else MirageConfig().ensemble
    print(f"[eval] building ensemble ({len(ensemble_spec)} models) on {device} ...")
    encoders = attach_targets(
        build_ensemble(ensemble_spec, device),
        build_target_set(args.categories, use_text=True),
    )

    orig = load_image(args.original)
    imm = load_image(args.immunized)
    s_orig = moderation_proxy(encoders, orig, device)
    s_imm = moderation_proxy(encoders, imm, device)
    print("=" * 60)
    print(f"moderation proxy (mean cosine-sim to unsafe targets):")
    print(f"  original  : {s_orig:.4f}")
    print(f"  immunized : {s_imm:.4f}   (delta = +{s_imm - s_orig:.4f})")

    if args.robustness:
        print("-" * 60)
        print("weak-adversary robustness (proxy after transform; want it to stay high):")
        for name, fn in WEAK_ADVERSARIES.items():
            s = moderation_proxy(encoders, fn(imm), device)
            retained = (s - s_orig) / max(1e-6, s_imm - s_orig) * 100
            print(f"  {name:<14}: {s:.4f}   ({retained:5.1f}% of the gain retained)")

    if args.openai_key:
        print("-" * 60)
        print("external check via OpenAI omni-moderation:")
        for label, path in [("original", args.original), ("immunized", args.immunized)]:
            try:
                r = openai_moderation(path, args.openai_key)
                print(f"  {label:<10}: flagged={r['flagged']}  max_score={r['max_score']:.4f}")
            except Exception as e:  # noqa: BLE001
                print(f"  {label:<10}: API error: {e!r}")


if __name__ == "__main__":
    main()
