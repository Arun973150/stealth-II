"""Spatial masks that restrict WHERE the perturbation may be applied (Sec. 3.1 cites
masking [16] as a related immunization technique).

The mask is 1 where delta is allowed to be non-zero and 0 where the original pixels must be
kept. Constraining the perturbation to the background (or an outer frame) keeps the subject
(e.g. a person) visually clean -- the target "painting" lands on the background instead.

Two modes, both using only torchvision (no new dependency):

  * "background": segment the salient subject with DeepLabV3 and allow perturbation only
    *outside* it (with a small feather so the subject's edge is also protected).
  * "border:<frac>": allow perturbation only within an outer frame of relative width `frac`
    (e.g. "border:0.25"). No segmentation -- crude but instant.

Note (honesty): confining the perturbation shrinks the signal the moderator sees, so refusal
may weaken relative to a full-image perturbation. This is an experimental knob.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_VOC_PERSON_CLASS = 15  # DeepLabV3 (VOC labels): "person"


def build_mask(spec: str, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return a (1,1,H,W) float mask in {0,1} for image (1,3,H,W) in [0,1]."""
    if spec.startswith("border"):
        frac = float(spec.split(":", 1)[1]) if ":" in spec else 0.25
        return _border_mask(image, frac).to(device)
    if spec == "background":
        return _background_mask(image, device)
    raise ValueError(f"Unknown mask spec '{spec}'. Use 'background' or 'border:<frac>'.")


def _border_mask(image: torch.Tensor, frac: float) -> torch.Tensor:
    _, _, h, w = image.shape
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    m = torch.zeros(1, 1, h, w)
    m[..., :bh, :] = 1
    m[..., h - bh :, :] = 1
    m[..., :, :bw] = 1
    m[..., :, w - bw :] = 1
    return m


@torch.no_grad()
def _background_mask(image: torch.Tensor, device: torch.device, feather: int = 9) -> torch.Tensor:
    """Perturb only the background: 1 outside the segmented person, 0 on the person."""
    import torchvision

    seg = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
    seg = seg.to(device).eval()

    _, _, h, w = image.shape
    # DeepLabV3 is happiest around 512px on the long side.
    scale = 520 / max(h, w)
    rh, rw = max(1, int(h * scale)), max(1, int(w * scale))
    x = F.interpolate(image.to(device), size=(rh, rw), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    logits = seg((x - mean) / std)["out"]                    # (1, 21, rh, rw)
    person = (logits.argmax(1, keepdim=True) == _VOC_PERSON_CLASS).float()

    # Dilate the person region so its silhouette edge is also protected (feathering).
    if feather > 1:
        person = F.max_pool2d(person, kernel_size=feather, stride=1, padding=feather // 2)
    person = F.interpolate(person, size=(h, w), mode="nearest")
    background = 1.0 - person
    del seg
    return background  # (1,1,H,W)
