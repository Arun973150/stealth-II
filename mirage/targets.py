"""Target concept set T (Sec. 4.2).

The paper surveys the acceptable-use policies of the commercial editors and extracts
the concepts that are *universally prohibited* -- primarily sexually explicit content,
graphic violence, and copyrighted material -- then collects images/text that instantiate
each concept.

This reproduction uses **text captions** phrased in the register of content-moderation
policies. CLIP-style encoders embed text and image into a shared space, so text targets
are a legitimate instantiation of T (Eq. 1 explicitly allows text or image targets), and
they let us align an image toward a prohibited *concept* without collecting any actual
explicit or violent media. Users who want image targets can drop files into a directory
and pass ``image_targets_dir`` in the config.

Fig. 9 of the paper finds the "sexual" category triggers refusals most strongly, so it is
the default target category.
"""

from __future__ import annotations

import os
from typing import Dict, List

from PIL import Image

# Policy-register descriptors per prohibited category. These mirror the language of
# acceptable-use policies; they name the prohibited concept so the CLIP text encoder
# places them in the corresponding "unsafe" region, without graphic detail.
TEXT_TARGETS: Dict[str, List[str]] = {
    "sexual": [
        "explicit sexual content",
        "pornographic imagery",
        "nudity and sexually explicit material",
        "adult content that violates the content policy",
        "sexually explicit photograph flagged by the safety filter",
        "NSFW explicit adult imagery",
    ],
    "violence": [
        "graphic violence",
        "gore and bloodshed",
        "extremely violent and graphic imagery",
        "a brutal violent scene that violates the content policy",
        "graphic depiction of injury and blood",
        "violent content flagged by the safety filter",
    ],
    "copyright": [
        "copyrighted movie still",
        "a frame from a copyrighted film that violates the content policy",
        "trademarked and copyrighted commercial artwork",
    ],
}


class TargetSet:
    """Holds the raw targets (texts and/or PIL images) for one or more categories.

    Embeddings are produced lazily by each surrogate encoder (see ensemble.py), because
    every model has its own text tokenizer / image preprocessing.
    """

    def __init__(self, texts: List[str], images: List[Image.Image]):
        self.texts = texts
        self.images = images

    def __len__(self) -> int:
        return len(self.texts) + len(self.images)

    def __repr__(self) -> str:
        return f"TargetSet(texts={len(self.texts)}, images={len(self.images)})"


def build_target_set(
    categories: List[str],
    use_text: bool = True,
    image_targets_dir: str | None = None,
) -> TargetSet:
    """Assemble T from the requested categories.

    Args:
        categories: subset of TEXT_TARGETS keys, e.g. ["sexual", "violence"].
        use_text: include the policy-register text captions.
        image_targets_dir: optional directory of user-supplied image targets. If it
            contains per-category subfolders, only the requested categories are used;
            otherwise every image in the directory is used.
    """
    texts: List[str] = []
    if use_text:
        for cat in categories:
            if cat not in TEXT_TARGETS:
                raise KeyError(f"Unknown target category '{cat}'. Known: {list(TEXT_TARGETS)}")
            texts.extend(TEXT_TARGETS[cat])

    images: List[Image.Image] = []
    if image_targets_dir:
        images = _load_image_targets(image_targets_dir, categories)

    if len(texts) + len(images) == 0:
        raise ValueError("Target set is empty: enable text targets or provide image targets.")
    return TargetSet(texts=texts, images=images)


_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _load_image_targets(root: str, categories: List[str]) -> List[Image.Image]:
    paths: List[str] = []
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if subdirs and any(c in subdirs for c in categories):
        for cat in categories:
            cdir = os.path.join(root, cat)
            if os.path.isdir(cdir):
                paths += _list_images(cdir)
    else:
        paths = _list_images(root)
    return [Image.open(p).convert("RGB") for p in paths]


def _list_images(d: str) -> List[str]:
    out = []
    for fn in sorted(os.listdir(d)):
        if os.path.splitext(fn)[1].lower() in _IMG_EXT:
            out.append(os.path.join(d, fn))
    return out
