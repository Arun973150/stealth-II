"""Moderation-VLM surrogates: ShieldGemma-2 and Llama Guard 3 Vision (Sec. 4.3).

The paper additionally optimizes against open moderation *models* that output a safe/unsafe
probability, and names **ShieldGemma [68]** (plus a second, unnamed model cited as [15]).
This module wires the two large vision-language moderators as differentiable surrogates:

  * ``google/shieldgemma-2-4b-it``            -- confirmed in the paper (ref [68]).
  * ``meta-llama/Llama-Guard-3-11B-Vision``   -- a *plausible* candidate for the unnamed
                                                 ref [15]; NOT named in the paper text.

Both are generative VLMs: they emit an affirmative/negative token ("Yes"/"No" or
"unsafe"/"safe") rather than exposing a classifier head. We turn that into a differentiable
scalar in [0,1] by reading the logits at the answer position and taking a two-way softmax
over the {violating, safe} token ids -> P(violating), which MIRAGE maximizes.

Getting gradients to the pixels requires a differentiable image path. The processors do
non-differentiable PIL preprocessing, so we run the processor ONCE on a dummy square image
at the model's native resolution to capture the exact prompt template (input_ids,
attention_mask, and any model-specific tensors such as Mllama's aspect-ratio ids) plus the
expected pixel_values shape. At each step we keep those cached text/meta tensors fixed and
substitute our own differentiable, normalized pixel_values (they don't change the template
because the image size is fixed). See ``_VLMModerator``.

IMPORTANT (honesty): both models are license-gated on Hugging Face (accept Google/Meta terms
and authenticate with an HF token) and large (4B / 11B). These loaders are written to the
documented API but are **GPU-only and untested in a CPU-only environment**; a shape/template
tweak may be needed for your exact transformers version. They are opt-in and are NOT part of
the default ensemble, which uses a lightweight NSFW classifier instead.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .config import ModelSpec
from .ensemble import Encoder


class _VLMModerator:
    """Wraps a generative moderation VLM as a differentiable unsafe-probability function."""

    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        prompt_messages: List[dict],
        violate_words: List[str],
        safe_words: List[str],
    ):
        self.model = model
        self.processor = processor
        self.device = device
        tok = getattr(processor, "tokenizer", processor)

        # 1) Build the template ONCE on a dummy image to capture text/meta tensors + shapes.
        dummy = self._dummy_image(processor)
        enc = self._encode(processor, prompt_messages, dummy)
        self._pixel_shape = tuple(enc["pixel_values"].shape)          # e.g. (1,3,896,896)
        self._cached = {k: v.to(device) for k, v in enc.items() if k != "pixel_values"}

        # 2) Resolve the {violating, safe} first-token ids for the two-way softmax.
        self.violate_id = _first_token_id(tok, violate_words)
        self.safe_id = _first_token_id(tok, safe_words)

    # ---- helpers ----
    @staticmethod
    def _dummy_image(processor) -> Image.Image:
        r = _native_resolution(processor)
        return Image.fromarray(np.zeros((r, r, 3), dtype=np.uint8))

    @staticmethod
    def _encode(processor, messages, image) -> Dict[str, torch.Tensor]:
        # Prefer the chat template (handles image tokens correctly per model family).
        try:
            out = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", images=[image],
            )
            if "pixel_values" in out:
                return dict(out)
        except Exception:
            pass
        # Fallback: flatten the message text and call the processor directly.
        text = " ".join(
            c["text"] for m in messages for c in m["content"] if c.get("type") == "text"
        )
        return dict(processor(images=image, text=text, return_tensors="pt"))

    def unsafe_score(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B,3,r,r) differentiable in [normalized] space -> (B,) P(violating)."""
        b = pixels.shape[0]
        scores = []
        for i in range(b):  # B is 1 for global-only VLM moderators
            pv = pixels[i : i + 1].reshape(self._pixel_shape).to(self.device)
            kwargs = {k: v for k, v in self._cached.items()}
            out = self.model(pixel_values=pv, **kwargs)
            last = out.logits[:, -1, :]                      # (1, vocab) first answer token
            pair = torch.stack([last[0, self.violate_id], last[0, self.safe_id]])
            scores.append(torch.softmax(pair, dim=0)[0])     # P(violating)
        return torch.stack(scores)


def _first_token_id(tokenizer, words: List[str]) -> int:
    for w in words:
        for cand in (w, " " + w):
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if ids:
                return ids[0]
    raise ValueError(f"Could not resolve a token id for any of {words}")


def _native_resolution(processor) -> int:
    ip = getattr(processor, "image_processor", processor)
    size = getattr(ip, "size", {}) or {}
    return int(size.get("height") or size.get("width") or size.get("shortest_edge") or 896)


def _image_norm(processor):
    ip = getattr(processor, "image_processor", processor)
    mean = getattr(ip, "image_mean", (0.5, 0.5, 0.5))
    std = getattr(ip, "image_std", (0.5, 0.5, 0.5))
    return mean, std


def _load_vlm(spec: ModelSpec, device, prompt, violate, safe) -> Encoder:
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except Exception:  # older transformers
        from transformers import AutoModelForCausalLM as _AutoVLM

    processor = AutoProcessor.from_pretrained(spec.name)
    model = _AutoVLM.from_pretrained(spec.name, torch_dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mod = _VLMModerator(model, processor, device, prompt, violate, safe)
    mean, std = _image_norm(processor)
    return Encoder(
        key=spec.key(), model=model, resolution=_native_resolution(processor),
        mean=mean, std=std, device=device, score_kind="moderation",
        unsafe_score=mod.unsafe_score, use_local=False,  # global view only
    )


_SHIELDGEMMA_PROMPT = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": (
            "You are a content-safety classifier. Does this image contain sexually "
            "explicit content, nudity, graphic violence, or gore? Answer Yes or No."
        )},
    ],
}]

_LLAMAGUARD_PROMPT = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": (
            "Classify this image for content-policy violations (sexual content, violence, "
            "gore). Respond with 'safe' or 'unsafe'."
        )},
    ],
}]


def load_shieldgemma2(spec: ModelSpec, device: torch.device) -> Encoder:
    """ShieldGemma-2 (paper ref [68]). Gated: accept Google's license + HF auth."""
    return _load_vlm(spec, device, _SHIELDGEMMA_PROMPT, ["Yes"], ["No"])


def load_llamaguard_vision(spec: ModelSpec, device: torch.device) -> Encoder:
    """Llama Guard 3 Vision (candidate for unnamed ref [15]). Gated: accept Meta's license."""
    return _load_vlm(spec, device, _LLAMAGUARD_PROMPT, ["unsafe"], ["safe"])
