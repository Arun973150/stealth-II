"""Surrogate ensemble (Sec. 4.3).

Each surrogate is wrapped in a common ``Encoder`` interface exposing a *fully
differentiable* per-view score, so PGD can backpropagate gradients from the objective all
the way to the input pixels. open_clip's built-in ``preprocess`` transforms operate on PIL
images and are NOT differentiable, so we reconstruct preprocessing ourselves.

Three surrogate kinds, all reduced to a single ``view_score(x) -> (B,)`` signal:

  * "openclip"   -- CLIP / OpenCLIP / MetaCLIP / EVA / DataComp ViTs. score = mean cosine
                    similarity of the image embedding to the (text/image) target embeddings.
  * "dino"       -- self-supervised DINOv2 (timm). Image targets only (no text encoder).
                    score = mean cosine similarity to image-target embeddings.
  * "moderation" -- an open-source image safety classifier (e.g. an NSFW ViT, standing in
                    for ShieldGemma). score = the differentiable "unsafe" probability, which
                    we maximize directly (Sec. 4.3: "open-source moderation models which
                    return a scalar in [0,1] ... we maximize this score").

Unifying all three as ``view_score`` lets the objective (Eq. 2) treat similarity and
moderation-probability identically.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import CenterCrop, Normalize, Resize

from .config import ModelSpec
from .targets import TargetSet

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Encoder:
    """Unified wrapper around a single surrogate model."""

    def __init__(
        self,
        key: str,
        model: torch.nn.Module,
        resolution: int,
        mean,
        std,
        device: torch.device,
        score_kind: str,                 # "embedding" | "moderation"
        supports_text: bool = False,
        tokenizer=None,
        text_encode: Optional[Callable] = None,
        image_encode: Optional[Callable] = None,
        unsafe_score: Optional[Callable] = None,  # moderation: preprocessed_x -> (B,)
        use_local: bool = True,          # False for heavy VLM moderators (global view only)
    ):
        self.key = key
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.resolution = int(resolution)
        self.device = device
        self.score_kind = score_kind
        self.supports_text = supports_text
        self.use_local = use_local
        self._tokenizer = tokenizer
        self._text_encode = text_encode
        self._image_encode = image_encode
        self._unsafe_score = unsafe_score
        self._mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self._target_cache: Optional[torch.Tensor] = None

    # ---- differentiable preprocessing ----
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        r = self.resolution
        if x.shape[-2:] != (r, r):
            x = F.interpolate(x, size=(r, r), mode="bicubic", align_corners=False)
        x = x.clamp(0, 1)
        return (x - self._mean) / self._std

    def embed_image(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._image_encode(self._preprocess(x))
        return F.normalize(feats, dim=-1)

    # ---- unified per-view score (differentiable) ----
    def view_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return a per-image score (B,) in a maximize-me direction (higher = 'unsafer')."""
        if self.score_kind == "moderation":
            return self._unsafe_score(self._preprocess(x))
        # embedding: mean cosine similarity to targets
        emb = self.embed_image(x)                       # (B, D)
        return (emb @ self._target_cache.t()).mean(dim=1)  # (B,)

    # ---- target embeddings (embedding kind only; computed once, no grad) ----
    @torch.no_grad()
    def precompute_targets(self, targets: TargetSet) -> Optional[torch.Tensor]:
        if self.score_kind == "moderation":
            return None  # moderation models need no targets
        embs: List[torch.Tensor] = []
        if self.supports_text and targets.texts:
            tokens = self._tokenizer(targets.texts).to(self.device)
            embs.append(F.normalize(self._text_encode(tokens), dim=-1))
        for img in targets.images:
            arr = torch.from_numpy(
                np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)
            embs.append(self.embed_image(arr))
        self._target_cache = torch.cat(embs, dim=0) if embs else None
        return self._target_cache

    @property
    def target_embeddings(self) -> Optional[torch.Tensor]:
        return self._target_cache


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def _openclip_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.name, pretrained=spec.pretrained
    )
    tokenizer = open_clip.get_tokenizer(spec.name)
    resolution = _openclip_resolution(model, preprocess)
    mean, std = _extract_norm(preprocess, _CLIP_MEAN, _CLIP_STD)
    return Encoder(
        key=spec.key(), model=model, resolution=resolution, mean=mean, std=std,
        device=device, score_kind="embedding", supports_text=True, tokenizer=tokenizer,
        text_encode=lambda tok: model.encode_text(tok),
        image_encode=lambda px: model.encode_image(px),
    )


def _dino_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    import timm

    model = timm.create_model(spec.name, pretrained=True, num_classes=0)
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    input_size = cfg.get("input_size", (3, 518, 518))
    return Encoder(
        key=spec.key(), model=model, resolution=input_size[-1],
        mean=cfg.get("mean", _IMAGENET_MEAN), std=cfg.get("std", _IMAGENET_STD),
        device=device, score_kind="embedding", supports_text=False,
        image_encode=lambda px: model(px),
    )


def _moderation_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    """Load an open-source image safety classifier; score = differentiable unsafe prob."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(spec.name)
    proc = AutoImageProcessor.from_pretrained(spec.name)
    size = getattr(proc, "size", {}) or {}
    resolution = size.get("height") or size.get("shortest_edge") or 224
    mean = getattr(proc, "image_mean", _IMAGENET_MEAN)
    std = getattr(proc, "image_std", _IMAGENET_STD)

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    unsafe_idx = next(
        (i for i, l in id2label.items()
         if any(w in l.lower() for w in ("nsfw", "porn", "unsafe", "sexual", "explicit"))),
        1 if len(id2label) > 1 else 0,
    )

    def unsafe(px: torch.Tensor) -> torch.Tensor:
        logits = model(pixel_values=px).logits
        return torch.softmax(logits, dim=-1)[:, unsafe_idx]

    return Encoder(
        key=spec.key(), model=model, resolution=resolution, mean=mean, std=std,
        device=device, score_kind="moderation", supports_text=False, unsafe_score=unsafe,
    )


def _openclip_resolution(model, preprocess) -> int:
    img_size = getattr(getattr(model, "visual", None), "image_size", None)
    if img_size is not None:
        return img_size[0] if isinstance(img_size, (tuple, list)) else int(img_size)
    for t in preprocess.transforms:
        if isinstance(t, (Resize, CenterCrop)):
            s = t.size
            return s[0] if isinstance(s, (tuple, list)) else int(s)
    return 224


def _extract_norm(preprocess, default_mean, default_std):
    for t in preprocess.transforms:
        if isinstance(t, Normalize):
            return tuple(t.mean), tuple(t.std)
    return default_mean, default_std


def _shieldgemma2_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    from .moderation_vlms import load_shieldgemma2  # lazy: avoids heavy import unless used
    return load_shieldgemma2(spec, device)


def _llamaguard_vision_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    from .moderation_vlms import load_llamaguard_vision
    return load_llamaguard_vision(spec, device)


_LOADERS = {
    "openclip": _openclip_encoder,
    "dino": _dino_encoder,
    "moderation": _moderation_encoder,
    "shieldgemma2": _shieldgemma2_encoder,
    "llamaguard_vision": _llamaguard_vision_encoder,
}


def build_ensemble(specs: List[ModelSpec], device: torch.device) -> List[Encoder]:
    encoders: List[Encoder] = []
    for spec in specs:
        loader = _LOADERS.get(spec.kind)
        if loader is None:
            warnings.warn(f"Unknown model kind '{spec.kind}'; skipping {spec.key()}.")
            continue
        try:
            encoders.append(loader(spec, device))
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Failed to load surrogate {spec.key()}: {e!r}. Skipping.")
    if not encoders:
        raise RuntimeError("No surrogate encoders could be loaded.")
    return encoders


def attach_targets(encoders: List[Encoder], targets: TargetSet) -> List[Encoder]:
    """Precompute target embeddings; keep moderation models and any encoder with a target."""
    usable: List[Encoder] = []
    for enc in encoders:
        if enc.score_kind == "moderation":
            usable.append(enc)
            continue
        if enc.precompute_targets(targets) is None:
            warnings.warn(f"{enc.key} has no usable target (no text encoder + no image "
                          "targets); dropping from the objective.")
            continue
        usable.append(enc)
    if not usable:
        raise RuntimeError("No encoder has a usable target/score.")
    return usable
