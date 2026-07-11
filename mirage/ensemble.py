"""Surrogate ensemble (Sec. 4.3).

Each surrogate is wrapped in a common ``Encoder`` interface exposing a *fully
differentiable* per-view score, so PGD can backpropagate gradients from the objective all
the way to the input pixels. open_clip's built-in ``preprocess`` transforms operate on PIL
images and are NOT differentiable, so we reconstruct preprocessing ourselves.

Surrogate kinds, all reduced to a single ``view_score(x) -> (B,)`` signal. Table 5's exact
main-experiment ensemble uses "hf_dinov2", "hf_siglip", "openclip" (x5), and "shieldgemma2":

  * "openclip"         -- OpenCLIP ViTs (LAION / DataComp / DFN checkpoints per Table 5).
                          score = mean cosine similarity of the image embedding to targets.
  * "hf_dinov2"        -- self-supervised DINOv2 (`facebook/dinov2-base` via transformers).
                          Image targets only (no text encoder).
  * "hf_siglip"        -- SigLIP (`google/siglip-base-patch16-224` via transformers). Has
                          both a text and image tower, like the OpenCLIP entries.
  * "dino"             -- legacy timm-based DINOv2 loader; kept for flexibility, not used
                          by the default Table-5 ensemble.
  * "moderation"        -- a generic open-source image safety classifier (e.g. an NSFW ViT).
                          score = the differentiable "unsafe" probability, maximized directly.
  * "shieldgemma2"      -- ShieldGemma-2 (paper ref [68]); see moderation_vlms.py.
  * "llamaguard_vision" -- Llama Guard 3 Vision (paper ref [15]); opt-in extra, not part of
                          Table 5's main-experiment ensemble. See moderation_vlms.py.

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


def _as_tensor(x):
    """Normalize an encoder's output to a plain tensor.

    Some HF methods that are documented to return a pooled tensor (e.g.
    ``get_text_features`` / ``get_image_features``) return a raw model-output object
    (``BaseModelOutputWithPooling``) instead, depending on the installed ``transformers``
    version. Handle both so a version difference doesn't crash the whole ensemble.
    """
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state[:, 0, :]  # CLS-token fallback
    raise TypeError(f"Unexpected encoder output type: {type(x)}")


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
        feats = _as_tensor(self._image_encode(self._preprocess(x)))
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
            embs.append(F.normalize(_as_tensor(self._text_encode(tokens)), dim=-1))
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
# Checkpoint families trained with QuickGELU activation. When such a tag is loaded under a
# plain (non-quickgelu) architecture name, open_clip warns of a mismatch and silently uses
# standard GELU -- the wrong activation -- degrading the embeddings. We force QuickGELU for
# these. (We do this via force_quick_gelu instead of a "-quickgelu" arch name because e.g.
# the tag `dfn2b_s39b` is registered only under the plain `ViT-L-14`, not `ViT-L-14-quickgelu`.)
_QUICKGELU_TAGS = ("dfn", "openai", "metaclip")


def _needs_quick_gelu(pretrained: Optional[str]) -> bool:
    return bool(pretrained) and any(t in pretrained for t in _QUICKGELU_TAGS)


def _openclip_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.name, pretrained=spec.pretrained,
        force_quick_gelu=_needs_quick_gelu(spec.pretrained),
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
    """Legacy timm-based DINOv2 loader (kept for flexibility; Table 5 uses the HF loader
    below, ``hf_dinov2``, for the actual paper ensemble)."""
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


def _hf_dinov2_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    """Table 5's DINOv2 surrogate: ``facebook/dinov2-base`` via HuggingFace transformers.

    No text encoder (self-supervised, image-only), so it only ever aligns to image targets
    -- consistent with the paper's statement that DINOv2 uses image embeddings only.
    """
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(spec.name)
    proc = AutoImageProcessor.from_pretrained(spec.name)
    size = getattr(proc, "size", {}) or {}
    resolution = size.get("height") or size.get("shortest_edge") or 224
    mean = getattr(proc, "image_mean", _IMAGENET_MEAN)
    std = getattr(proc, "image_std", _IMAGENET_STD)

    def image_encode(px: torch.Tensor) -> torch.Tensor:
        out = model(pixel_values=px)
        return out.last_hidden_state[:, 0, :]  # CLS token

    return Encoder(
        key=spec.key(), model=model, resolution=resolution, mean=mean, std=std,
        device=device, score_kind="embedding", supports_text=False, image_encode=image_encode,
    )


def _hf_siglip_encoder(spec: ModelSpec, device: torch.device) -> Encoder:
    """Table 5's SigLIP surrogate: ``google/siglip-base-patch16-224`` (paper ref [69]).

    Loaded via transformers.SiglipModel so both the image and text towers are usable, matching
    how CLIP-style entries in T (text or image) are handled generically.
    """
    from transformers import AutoProcessor, SiglipModel

    model = SiglipModel.from_pretrained(spec.name)
    processor = AutoProcessor.from_pretrained(spec.name)
    ip = processor.image_processor
    size = getattr(ip, "size", {}) or {}
    resolution = size.get("height") or size.get("width") or 224
    mean = getattr(ip, "image_mean", _IMAGENET_MEAN)
    std = getattr(ip, "image_std", _IMAGENET_STD)

    def tokenizer(texts: List[str]) -> torch.Tensor:
        enc = processor.tokenizer(
            texts, padding="max_length", truncation=True, return_tensors="pt"
        )
        return enc["input_ids"]

    return Encoder(
        key=spec.key(), model=model, resolution=resolution, mean=mean, std=std,
        device=device, score_kind="embedding", supports_text=True, tokenizer=tokenizer,
        text_encode=lambda ids: model.get_text_features(input_ids=ids),
        image_encode=lambda px: model.get_image_features(pixel_values=px),
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
    "hf_dinov2": _hf_dinov2_encoder,
    "hf_siglip": _hf_siglip_encoder,
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
