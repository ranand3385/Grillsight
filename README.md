# GrillSight — Real-Time Meat Doneness Classifier
**ECE 570 Course Project · Track 2: Product Prototype**

A real-time computer-vision system that classifies the doneness level of meat
(raw, rare, medium_rare, medium, medium_well, well_done) from a live webcam or
video feed. Built on EfficientNet-B0 + class-adaptive offline augmentation.

- **Test accuracy:** 98.6% (vs. 83.3% baseline)
- **Raw-class F1:** 0.96 (vs. 0.00 baseline — class was completely failing)
- **CPU inference:** 26.6 FPS, no GPU required

---

## Repository Structure

```
570-Project/
├── src/
│   ├── model.py          # EfficientNet-B0 + custom doneness head
│   ├── dataset.py        # ImageFolder pipeline, transforms, class weights
│   ├── train.py          # training loop (AdamW, cosine LR, early stop, weighted loss)
│   ├── inference.py      # real-time webcam overlay
│   └── evaluate.py       # confusion matrix + FPS benchmark
├── scripts/
│   ├── augment_dataset.py    # offline class-adaptive augmentation (CP2)
│   └── download_dataset.py   # Roboflow or synthetic demo generator
├── paper/                # ICLR-style final paper (paper.tex, paper.bib, figures/)
├── data/                 # Dataset in ImageFolder layout (auto-generated demo or Roboflow)
├── checkpoints_v2/       # Trained model (best_model.pt, history.json)
├── generate_slides.py        # Checkpoint-1 slide generator (MeatVision baseline)
├── generate_slides_cp2.py    # Checkpoint-2 slide generator (GrillSight)
├── checkpoint1_slides.pdf
├── checkpoint2_slides.pdf
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Required packages:** `torch`, `torchvision`, `opencv-python`, `Pillow`,
`numpy`, `matplotlib`, `scikit-learn`, `tqdm`, `requests`. All are installable
from PyPI; no custom wheels or system packages are required.

No GPU is required — the final model runs at 26.6 FPS on a standard CPU.

---

## Dataset

The dataset follows a standard PyTorch `ImageFolder` layout:

```
data/
  train/   val/   test/
    raw/  rare/  medium_rare/  medium/  medium_well/  well_done/
      *.jpg
```

### Option A — Automatic synthetic demo (default, no external data required)

The project ships with a synthetic-demo generator that runs without any API keys
or external downloads. It creates procedurally-generated colour-gradient images
per doneness class:

```bash
python scripts/download_dataset.py demo --dest data --n 60
```

This produces 60 train / 12 val / 12 test images per class (504 total). This
is the dataset used to produce all numbers in the paper and slides.

### Option B — Real meat images via Roboflow Universe

For production use, real meat images can be downloaded from Roboflow Universe.
Create a free account at <https://roboflow.com>, find a steak-doneness dataset,
then:

```bash
python scripts/download_dataset.py roboflow \
    --api-key  YOUR_API_KEY \
    --workspace YOUR_WORKSPACE \
    --project   YOUR_PROJECT \
    --version   1 \
    --dest      data
```

Requires `pip install roboflow`. The class-weighting in `src/train.py` auto-adapts
to unequal class counts, so no code changes are needed for real data.

---

## Quick-Start Reproduction (end-to-end)

```bash
# 1. Generate / fetch the dataset
python scripts/download_dataset.py demo --dest data --n 60

# 2. Expand training set 5x via class-adaptive offline augmentation
python scripts/augment_dataset.py --data data --factor 5

# 3. Train (20 epochs, lr=5e-4, CPU-OK, ~8 minutes)
python src/train.py --data data --epochs 20 --batch 32 \
    --lr 5e-4 --workers 0 --output checkpoints_v2

# 4. Evaluate
python src/evaluate.py --checkpoint checkpoints_v2/best_model.pt --data data

# 5. Run live webcam inference (press 'q' to quit)
python src/inference.py --checkpoint checkpoints_v2/best_model.pt
```

---

## Core Commands

| Task | Command |
|------|---------|
| Generate demo dataset | `python scripts/download_dataset.py demo --dest data --n 60` |
| Offline augment (5×) | `python scripts/augment_dataset.py --data data --factor 5` |
| Train | `python src/train.py --data data --epochs 20 --workers 0 --output checkpoints_v2` |
| Evaluate | `python src/evaluate.py --checkpoint checkpoints_v2/best_model.pt --data data` |
| Live webcam | `python src/inference.py --checkpoint checkpoints_v2/best_model.pt` |
| Single image | `python src/inference.py --checkpoint checkpoints_v2/best_model.pt --source img.jpg --image` |
| Video file | `python src/inference.py --checkpoint checkpoints_v2/best_model.pt --source video.mp4` |
| Regenerate CP1 slides | `python generate_slides.py` |
| Regenerate CP2 slides | `python generate_slides_cp2.py` |

---

## Code Authorship Statement

All source code in this repository was written by the author (Rishith Anand)
for this project. An LLM assistant (Anthropic Claude) was consulted during
development for scaffolding and refactoring suggestions; no code was copied
from any external public repository. Library usage (PyTorch, torchvision,
OpenCV, scikit-learn) is standard and unmodified.

### Files written entirely by the author

| File | Lines | Role |
|------|------:|------|
| `src/model.py` | 1–123 | EfficientNet-B0 wrapper, custom MLP head, parameter counting |
| `src/dataset.py` | 1–153 | Transform pipelines, ImageFolder builders, class-weight helper |
| `src/train.py` | 1–172 | Training loop, early stopping, checkpointing |
| `src/inference.py` | 1–204 | OpenCV real-time loop, overlay renderer, single-image and video modes |
| `src/evaluate.py` | 1–97 | Classification report, confusion matrix, inference-speed benchmark |
| `scripts/augment_dataset.py` | 1–84 | Offline class-adaptive augmentation (**new in CP2**) |
| `scripts/download_dataset.py` | 1–147 | Roboflow downloader + synthetic-demo generator |
| `generate_slides.py` | 1–466 | Checkpoint-1 slide PDF generator |
| `generate_slides_cp2.py` | 1–343 | Checkpoint-2 slide PDF generator |
| `paper/paper.tex` | 1–– | ICLR-style final paper |
| `paper/paper.bib` | 1–– | Bibliography |

### Changes made to prior code between checkpoints

Prior code refers to code carried forward from Checkpoint 1. All edits are the
author's own.

| File | Line range | Change |
|------|-----------:|--------|
| `src/dataset.py` | 139–153 | Added `get_class_weights()` — computes inverse-frequency class weights from the train-split filesystem layout. |
| `src/train.py` | 21 | Imported `get_class_weights` from `dataset`. |
| `src/train.py` | 96–99 | Replaced unweighted `nn.CrossEntropyLoss(label_smoothing=0.1)` with class-weighted variant that auto-adapts to imbalanced data. |
| `src/evaluate.py` | 84 | Replaced Unicode arrow in the benchmark print with ASCII `->` for Windows `cp1252`-codec compatibility. |
| `src/*` | all docstrings | Trimmed multi-paragraph docstrings to one line each; removed verbose inline comments that duplicated the code. |

### No external code copied

No code in this repository was copied, adapted, or forked from any external
public repository, tutorial, or sample project. All library calls follow
official documentation (PyTorch, torchvision, OpenCV, scikit-learn). Pre-trained
ImageNet weights for EfficientNet-B0 are downloaded automatically by
`torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`
on first run.

---

## LLM Usage Disclosure

An LLM assistant (Anthropic Claude) was used during development for:
- Scaffolding the initial project structure and argparse wiring.
- Drafting docstrings and the LaTeX structure of the final paper.
- Proposing the outline of the offline-augmentation approach.
- Slide-deck generation helper code.

All modelling decisions, the per-class jitter magnitudes, the class-weighting
formulation, the experimental protocol, the final authored paper, and all
reported numbers are the author's own.

---

## Paper

The final paper is in `paper/paper.tex`. Compile with:

```bash
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Or paste the files into <https://www.overleaf.com> as a new blank project.

---

## Results (reproduced on synthetic demo data)

| Metric | CP1 baseline | GrillSight (CP2) |
|--------|-------------:|-----------------:|
| Overall test accuracy | 83.3% | **98.6%** |
| Macro F1 | 0.78 | **0.99** |
| Raw-class F1 | 0.00 | **0.96** |
| Rare-class F1 | 0.67 | **0.96** |
| Training images | 360 | 2,160 |
| Epochs trained | 15 | 20 |
| CPU inference (FPS) | 36.6 | 26.6 |

See `paper/paper.pdf` for full analysis.
