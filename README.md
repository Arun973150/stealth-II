# MIRAGE — Moderation Induced Resistance Against Generative Editing

A faithful, from-scratch reproduction of **MIRAGE**, an image-immunization method that
protects personal photos from unauthorized AI-powered editing (GPT-Image, Gemini / "Nano
Banana", Grok Imagine).

Instead of attacking the black-box image *editor* (impossible without its weights), MIRAGE
adds an imperceptible perturbation that makes the editor's **pre-generation safety
moderator** flag the image as policy-violating — triggering a **prompt-agnostic refusal**.
It does this by maximizing the similarity of the (perturbed) image to a set of "unsafe"
concept targets across an **ensemble of open-source encoders**, using global + local views,
augmentations, and PGD with model-dropout + secant gradient caching.

```
Original image ──▶ [MIRAGE perturbation] ──▶ Immunized image
                                                   │
                        submit to GPT-Image/Gemini/Grok with ANY edit prompt
                                                   ▼
                              "I'm sorry, I can't help with that."  (refusal)
```

All hyperparameters below are set to match **Appendix B, Table 5** of the paper
("Hyper-parameters for MIRAGE used in the main experiments") verbatim — the authoritative
source, which overrides the looser prose description in Sec. 4.3 where the two disagree.

---

## Method → code map

| Paper component (Sec. 4)                                   | File |
|-----------------------------------------------------------|------|
| Target concept set `T` (policy-violating concepts)        | [mirage/targets.py](mirage/targets.py) |
| Surrogate ensemble (Table 5's exact 8 models)              | [mirage/ensemble.py](mirage/ensemble.py) |
| Moderation VLMs (ShieldGemma-2, Llama Guard 3 Vision)     | [mirage/moderation_vlms.py](mirage/moderation_vlms.py) |
| Differentiable preprocessing + unified `view_score`       | [mirage/ensemble.py](mirage/ensemble.py) |
| Global + shared Local views, top-k patches (Eq. 2)        | [mirage/views.py](mirage/views.py), [mirage/objective.py](mirage/objective.py) |
| Augmentations + straight-through JPEG                     | [mirage/augment.py](mirage/augment.py) |
| Ensemble objective `S(x)=Σ_i S_i(x)` (Eq. 1 & 2)          | [mirage/objective.py](mirage/objective.py) |
| Model dropout (prob+floor) + secant gradient caching      | [mirage/secant.py](mirage/secant.py) |
| PGD + cosine step schedule + public-API validation        | [mirage/attack.py](mirage/attack.py) |
| CLI                                                       | [scripts/immunize.py](scripts/immunize.py) |
| Evaluation (moderation proxy + weak-adversary robustness) | [scripts/evaluate.py](scripts/evaluate.py) |

---

## The exact ensemble (Table 5)

```
hf_dinov2:  facebook/dinov2-base                         (self-supervised, image targets only)
hf_siglip:  google/siglip-base-patch16-224                (ref [69])
open_clip:  ViT-B-32 / laion2b_s34b_b79k
open_clip:  ViT-B-16 / laion2b_s34b_b88k
open_clip:  ViT-L-14 / datacomp_xl_s13b_b90k
open_clip:  ViT-L-14 / dfn2b_s39b                          (DFN, ref [21])
open_clip:  ViT-H-14 / dfn5b                               (DFN, ref [21])
shieldgemma2: google/shieldgemma-2-4b-it (target: sexual)  (ref [68])
```

This is `mirage.config.DEFAULT_ENSEMBLE` — 8 models exactly as listed in Table 5, no more,
no less. (An earlier draft of this repo used a reconstructed 10-model ensemble before Table 5
was available; this has been corrected.)

`facebook/dinov2-base` and `google/siglip-base-patch16-224` are **not gated** and download
directly. `google/shieldgemma-2-4b-it` **is gated** — accept Google's license on Hugging Face
and run `huggingface-cli login` before using the default ensemble.

**A note on DINOv2 and targets:** DINOv2 has no text encoder, so per the paper it can only
align to *image* targets — text captions are structurally invisible to it. If you only supply
text targets (the default, for responsible-use reasons — see below), DINOv2 contributes
nothing and is dropped from the optimization with a warning. Supply image targets via
`--image-targets` (see the folder-structure note below) to activate it.

### Llama Guard 3 Vision — real citation, not in the main-experiment ensemble

The Sec. 4.3 prose names two open moderation models: **ShieldGemma [68]** and a second one
cited only as **[15]**. Reference [15] resolves to **J. Chi et al., "Llama Guard 3 Vision:
Safeguarding human-AI image understanding conversations," arXiv:2411.10414** — a real paper,
not a guess. However, **Table 5's actual main-experiment ensemble only includes ShieldGemma-2**
— Llama Guard 3 Vision is not part of the 8-model list above. It's wired here as an **opt-in
extra** (`--vlm-moderators`) on top of the Table-5 ensemble, not a verified part of the
headline results.

> ⚠️ Both ShieldGemma-2 and Llama Guard 3 Vision are **license-gated** on Hugging Face and
> **large (4B / 11B)** — GPU-only. Llama Guard 3 Vision's loader follows the documented API
> but is **experimental / untested**; your exact `transformers` version may need a small
> prompt-template or pixel-shape tweak.

---

## Install (GPU)

The default config is Table 5's exact main-experiment config and is meant for a CUDA GPU.

```bash
# 1) Install a CUDA build of torch that matches your driver, e.g. CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2) The rest:
pip install -r requirements.txt
# 3) ShieldGemma-2 is gated -- accept the license at the link below, then:
huggingface-cli login
```

Gated model link (accept the license before running the default config):
https://huggingface.co/google/shieldgemma-2-4b-it

The first run downloads the ensemble weights (~10–12 GB, dominated by ShieldGemma-2 4B) from
Hugging Face and caches them.

**Runtime (paper-reported):** ~15 minutes to immunize a single 1024×1024 image at 5000 steps
on one A40 (48 GB) GPU.

---

## Usage — full paper config (GPU)

`MirageConfig()` with no arguments **is** Table 5's configuration: the 8-model ensemble above,
`steps=5000`, `budget=16/255`, cosine step-size schedule, model dropout (30% drop probability,
floor of 3 active), secant caching, global + shared-local views (16 patches, fixed 256×256
crop, top-k=8, λ=0.25), and target category `sexual`. `device="auto"` selects CUDA
automatically.

```bash
python scripts/immunize.py --input photo.jpg --output photo_immunized.png
```

From Python:

```python
from mirage import immunize, MirageConfig
from mirage.utils import load_image, save_image

img = load_image("photo.jpg")
result = immunize(img, MirageConfig())      # Table 5 defaults, uses GPU if present
save_image(result.image, "photo_immunized.png")
print(result.objective, result.linf_255, result.psnr_db)
```

### Common options

```bash
# Sweep the L-inf budget (Table 4): 2/4/8/16 over 255
python scripts/immunize.py --input p.jpg --output o.png --budget 8

# Add the violence category alongside sexual (paper's main config uses "sexual" only)
python scripts/immunize.py --input p.jpg --output o.png --categories sexual violence

# Ablate global-local (Table 2 "No Global-local" row) or augmentations
python scripts/immunize.py --input p.jpg --output o.png --no-global-local

# Optional: public-moderation-API checkpoint selection (Sec 4.3, Table 2 "No C-tilde" row;
# off by default -- the paper finds omitting it performs comparably or better)
python scripts/immunize.py --input p.jpg --output o.png --openai-validate sk-...

# Add your own image targets (a folder of images, optionally per-category subfolders) --
# needed to activate DINOv2 (see note above)
python scripts/immunize.py --input p.jpg --output o.png --image-targets ./targets

# Add Llama Guard 3 Vision on top of the default ensemble (GPU-only, license-gated, opt-in)
python scripts/immunize.py --input p.jpg --output o.png --vlm-moderators
```

### Local image-target folder structure

Gitignored by default — never commit real target images to a public repo.

```
targets/
  sexual/       <- images for this repo instance/local testing only
  violence/
  copyright/
```

Only subfolders matching `--categories` are loaded (see `mirage/targets.py`).

---

## Evaluate

Because freely querying commercial editors is costly, the harness reports a local
**moderation proxy** (mean cosine similarity to the unsafe target set across the ensemble);
a good immunization sharply raises it. It also runs the **weak-adversary robustness** suite
(Table 3): blur, JPEG, resize-down, greyscale, screenshot.

```bash
python scripts/evaluate.py --original photo.jpg --immunized photo_immunized.png --robustness
# optional external black-box check (public C-tilde):
python scripts/evaluate.py --original photo.jpg --immunized out.png --openai-key sk-...
```

> To reproduce the paper's headline immunization-rate percentages (Table 1: GPT-Image
> 100.0%, Gemini 88.5%, Grok 94.0% at ε=16/255) you must submit the immunized images to the
> real GPT-Image / Gemini / Grok edit endpoints and count refusals — that requires your own
> paid API keys. This repo produces the immunized images and the local proxy; it does not
> ship commercial-editor clients.

---

## Configuration reference (`mirage/config.py`)

| Field | Table 5 value | Meaning |
|-------|---------------|---------|
| `budget` | `16/255` | L∞ perturbation bound `B` (also swept: 2, 4, 8/255) |
| `steps` | `5000` | PGD iterations |
| `step_size`, `step_schedule` | `1.0/255`, `"cosine"` | peak step size, cosine-annealed to ~0 |
| `ensemble` | 8 models (see above) | surrogate models |
| `patch_size`, `num_patches` | `256`, `16` | shared local-view crop size / count |
| `top_k`, `lambda_local` | `8`, `0.25` | top-k pooling, local-vs-global weight (Eq. 2) |
| `target_categories` | `["sexual"]` | which concepts form `T` |
| `use_augmentations`, `aug_prob` | `True`, `0.9` | stochastic augmentations |
| `eot_samples` | `1` | augmented views per step (no averaging) |
| `model_dropout`, `surrogate_drop_prob`, `min_active_surrogates` | `True`, `0.3`, `3` | gradient subset per step |
| `use_secant_cache` | `True` | secant approx. for dropped models |
| `select_by_public_api` | `False` | public-API checkpoint selection |

Resolution is preserved: the immunized image is saved at the **same width/height** as the
input (e.g. 1024×1024 → 1024×1024). The shared 256×256 local-view crop is sampled once per
step from the full-resolution image and used only to compute the loss — every surrogate
scores the *same* 16 regions each step (not independently resampled per encoder), then each
resizes them internally to its own native resolution.

---

## Notes on targets and responsible use

* This reproduction instantiates `T` with **policy-register text captions** (e.g. "explicit
  sexual content"), not actual explicit or violent media. CLIP-style encoders share a
  text/image embedding space, so text targets are a legitimate instantiation of `T`, and this
  avoids collecting harmful media. You may supply your own image targets via
  `--image-targets` (needed for DINOv2 to contribute, see above).
* MIRAGE is a **defensive privacy tool**: it makes *your own* images refuse to be edited.
  It causes editors to *refuse* — it does not generate or unlock any prohibited content.

---

## Scope

This repository implements the **MIRAGE method** (Sec. 4, Table 5) and its own evaluation
(moderation proxy + robustness). It does not include competing immunization baselines or
commercial-editor API clients.
