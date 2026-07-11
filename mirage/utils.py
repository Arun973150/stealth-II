"""Small helpers: device resolution, seeding, and image <-> tensor conversion."""

from __future__ import annotations

import base64
import io
import json
import random
import urllib.request

import numpy as np
import torch
from PIL import Image


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: str) -> torch.Tensor:
    """Load an image as a float tensor in [0, 1] with shape (1, 3, H, W)."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t


def save_image(t: torch.Tensor, path: str) -> None:
    """Save a (1,3,H,W) or (3,H,W) tensor in [0,1] to disk as PNG/JPEG."""
    if t.dim() == 4:
        t = t[0]
    arr = (t.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0)
    arr = arr.round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    """L-inf distance in 0-255 units, for reporting perturbation magnitude."""
    return float((a - b).abs().max().item() * 255.0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def tensor_to_png_bytes(t: torch.Tensor) -> bytes:
    if t.dim() == 4:
        t = t[0]
    arr = (t.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def openai_moderation_score(
    image: torch.Tensor, api_key: str, model: str = "omni-moderation-latest", timeout: int = 30
) -> float:
    """Return the max category score from OpenAI's public omni-moderation endpoint.

    This is the public black-box moderator C-tilde used for optional checkpoint selection
    (Sec. 4.3). Note it is *distinct* from any editor's internal moderator C; the paper finds
    that a high C-tilde does not guarantee a high C, and that omitting this selection performs
    comparably or better -- so it is off by default.
    """
    b64 = base64.b64encode(tensor_to_png_bytes(image)).decode()
    payload = json.dumps({
        "model": model,
        "input": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/moderations", data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read())
    return float(max(res["results"][0]["category_scores"].values()))
