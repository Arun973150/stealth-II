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

---

## Method → code map

| Paper component (Sec. 4)                                   | File |
|-----------------------------------------------------------|------|
| Target concept set `T` (policy-violating concepts)        | [mirage/targets.py](mirage/targets.py) |
| Surrogate ensemble (CLIP/OpenCLIP + DINOv2 + moderation)  | [mirage/ensemble.py](mirage/ensemble.py) |
| Moderation VLMs (ShieldGemma-2, Llama Guard 3 Vision)     | [mirage/moderation_vlms.py](mirage/moderation_vlms.py) |
| Differentiable preprocessing + unified `view_score`       | [mirage/ensemble.py](mirage/ensemble.py) |
| Global + Local views, top-k patches (Eq. 2)               | [mirage/views.py](mirage/views.py) |
| Augmentations + straight-through JPEG                     | [mirage/augment.py](mirage/augment.py) |
| Ensemble objective `S(x)=Σ_i S_i(x)` (Eq. 1 & 2)          | [mirage/objective.py](mirage/objective.py) |
| Model dropout + secant gradient caching                   | [mirage/secant.py](mirage/secant.py) |
| PGD / iterated FGSM immunizer + public-API validation     | [mirage/attack.py](mirage/attack.py) |
| CLI                                                       | [scripts/immunize.py](scripts/immunize.py) |
| Evaluation (moderation proxy + weak-adversary robustness) | [scripts/evaluate.py](scripts/evaluate.py) |

---

## Install (GPU)

The default config is the full paper config and is meant for a CUDA GPU.

```bash
# 1) Install a CUDA build of torch that matches your driver, e.g. CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2) The rest:
pip install -r requirements.txt
```

The first run downloads the ensemble weights (~2–3 GB) from Hugging Face and caches them.

---

## Usage — full paper config (GPU)

`MirageConfig()` with no arguments **is** the paper configuration:
10-model ensemble (8 CLIP-style + DINOv2 + moderation classifier), `steps=5000`,
`budget=16/255`, model dropout (4 active/step), secant caching, global+local views
(16 patches, top-k=4), and augmentations. `device="auto"` selects CUDA automatically.

```bash
python scripts/immunize.py --input photo.jpg --output photo_immunized.png
```

From Python:

```python
from mirage import immunize, MirageConfig
from mirage.utils import load_image, save_image

img = load_image("photo.jpg")
result = immunize(img, MirageConfig())      # full paper defaults, uses GPU if present
save_image(result.image, "photo_immunized.png")
print(result.objective, result.linf_255, result.psnr_db)
```

### Common options

```bash
# Sweep the L-inf budget (Fig. 4): 2/4/8/16 over 255
python scripts/immunize.py --input p.jpg --output o.png --budget 8

# Choose the target category (Fig. 9: "sexual" triggers most strongly)
python scripts/immunize.py --input p.jpg --output o.png --categories sexual violence

# Ablate global-local (Fig. 8) or augmentations
python scripts/immunize.py --input p.jpg --output o.png --no-global-local

# Optional: public-moderation-API checkpoint selection (Sec. 4.3; off by default)
python scripts/immunize.py --input p.jpg --output o.png --openai-validate sk-...

# Add your own image targets (a folder of images, optionally per-category subfolders)
python scripts/immunize.py --input p.jpg --output o.png --image-targets ./targets

# Add the large moderation VLMs to the ensemble (GPU-only, license-gated — see below)
python scripts/immunize.py --input p.jpg --output o.png --vlm-moderators
```

### Moderation-VLM surrogates (ShieldGemma-2 + Llama Guard 3 Vision)

The paper optimizes against open moderation *models* that output a safe/unsafe probability
(Sec. 4.3). What the paper text actually names:

* **ShieldGemma** — explicitly named, citation **[68]**. Wired here as
  `google/shieldgemma-2-4b-it` (the image variant).
* A **second, unnamed** moderation model, cited only as **[15]**. The popular "Llama Guard 3
  Vision" attribution is an *inference*, not stated in the paper. It is wired here as
  `meta-llama/Llama-Guard-3-11B-Vision` and clearly labeled as a candidate for [15].

These are turned into differentiable surrogates by reading the answer-token logits and taking
a two-way softmax over the {violating, safe} tokens → P(violating), which MIRAGE maximizes.
They run **global-view only** (patch forwards through a 4B/11B model per step are infeasible).

> ⚠️ Both are **license-gated** on Hugging Face (accept Google's / Meta's terms and run
> `huggingface-cli login`) and **large (4B / 11B)** — GPU-only. The loaders follow the
> documented API but are **experimental / untested in a CPU-only environment**; your exact
> `transformers` version may need a small prompt-template or pixel-shape tweak. They are
> opt-in (`--vlm-moderators`); the default ensemble uses a lightweight NSFW classifier so the
> repo runs out-of-the-box.

---

## Evaluate

Because freely querying commercial editors is costly, the harness reports a local
**moderation proxy** (mean cosine similarity to the unsafe target set across the ensemble);
a good immunization sharply raises it. It also runs the **weak-adversary robustness** suite
(Fig. 5): blur, JPEG, resize-down, greyscale, screenshot.

```bash
python scripts/evaluate.py --original photo.jpg --immunized photo_immunized.png --robustness
# optional external black-box check (public C-tilde):
python scripts/evaluate.py --original photo.jpg --immunized out.png --openai-key sk-...
```

> To reproduce the paper's headline immunization-rate percentages (~88–100%) you must submit
> the immunized images to the real GPT-Image / Gemini / Grok edit endpoints and count refusals
> — that requires your own paid API keys. This repo produces the immunized images and the
> local proxy; it does not ship commercial-editor clients.

---

## Configuration reference (`mirage/config.py`)

| Field | Paper default | Meaning |
|-------|---------------|---------|
| `budget` | `16/255` | L∞ perturbation bound `B` |
| `steps` | `5000` | PGD iterations |
| `ensemble` | 8 CLIP + DINOv2 + moderation | surrogate models |
| `use_global_local`, `num_patches`, `top_k`, `lambda_local` | `True`, `16`, `4`, `1.0` | Eq. 2 |
| `target_categories` | `["sexual","violence"]` | which concepts form `T` |
| `use_augmentations`, `aug_prob` | `True`, `0.9` | stochastic augmentations |
| `model_dropout`, `active_models_per_step` | `True`, `4` | gradient subset per step |
| `use_secant_cache` | `True` | secant approx. for dropped models |
| `select_by_public_api` | `False` | public-API checkpoint selection |

Resolution is preserved: the immunized image is saved at the **same width/height** as the
input (e.g. 1024×1024 → 1024×1024). Global/local views are used only to compute the loss.

---

## Notes on targets and responsible use

* This reproduction instantiates `T` with **policy-register text captions** (e.g. "explicit
  sexual content"), not actual explicit or violent media. CLIP-style encoders share a
  text/image embedding space, so text targets are a legitimate instantiation of `T` (Eq. 1
  allows text or image targets) and avoid collecting harmful media. You may supply your own
  image targets via `--image-targets`.
* MIRAGE is a **defensive privacy tool**: it makes *your own* images refuse to be edited.
  It causes editors to *refuse* — it does not generate or unlock any prohibited content.

---

## Scope

This repository implements the **MIRAGE method** (Sec. 4) and its own evaluation
(moderation proxy + robustness). It does not include competing immunization baselines.
