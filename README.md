# MeatVision — Real-Time Meat Doneness Detector
**ECE 570 Course Project · Track 2: ProductPrototype**

A computer-vision system that classifies the **doneness level** of meat (beef,
chicken, pork) in real time from a live webcam or video feed.

---

## Project Structure

```
570 Project/
├── src/
│   ├── model.py          # EfficientNet-B0 transfer-learning classifier
│   ├── dataset.py        # ImageFolder pipeline + augmentation
│   ├── train.py          # Training loop (early stopping, LR scheduling)
│   ├── inference.py      # Real-time webcam inference with OpenCV overlay
│   └── evaluate.py       # Confusion matrix + inference speed benchmark
├── scripts/
│   └── download_dataset.py  # Roboflow downloader or synthetic demo generator
├── data/                    # Dataset (created by download_dataset.py)
├── checkpoints/             # Saved model checkpoints (created during training)
├── generate_slides.py       # Generates checkpoint1_slides.pdf
├── requirements.txt
└── README.md
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `torch`, `torchvision`, `opencv-python`, `Pillow`,
`numpy`, `matplotlib`, `scikit-learn`, `tqdm`, `requests`.

> **macOS / Python 3.11 note:** The code patches SSL certificate verification
> automatically (`ssl._create_unverified_context`).  Alternatively, run
> `/Applications/Python 3.11/Install Certificates.command` once.

---

## Dataset

### Option A — Roboflow (recommended for real training)

1. Create a free account at <https://roboflow.com>.
2. Search Roboflow Universe for a **steak doneness** or **meat quality** dataset.
3. Click *Download → Folder Structure* and copy your API key.

```bash
python scripts/download_dataset.py roboflow \
    --api-key  YOUR_API_KEY \
    --workspace YOUR_WORKSPACE \
    --project   YOUR_PROJECT \
    --version   1 \
    --dest      data
```

### Option B — Synthetic demo dataset (smoke-test only)

Generates colour-gradient placeholder images — confirms the pipeline runs but
produces no meaningful visual accuracy:

```bash
python scripts/download_dataset.py demo --dest data --n 60
```

### Expected directory layout after download

```
data/
  train/
    raw/          medium/
    rare/         medium_well/
    medium_rare/  well_done/
  val/   <same classes>
  test/  <same classes>
```

---

## Training

```bash
cd src
python train.py --data ../data --epochs 30 --batch 32
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data` | Dataset root directory |
| `--epochs` | `30` | Maximum training epochs |
| `--batch` | `32` | Batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--patience` | `7` | Early-stopping patience (epochs) |
| `--workers` | `4` | DataLoader worker threads (use `0` on macOS) |

The best checkpoint is saved to `checkpoints/best_model.pt`.

---

## Real-Time Inference

### Webcam (default camera)

```bash
python src/inference.py --checkpoint checkpoints/best_model.pt
```

### Video file

```bash
python src/inference.py --checkpoint checkpoints/best_model.pt \
    --source path/to/video.mp4
```

### Single image

```bash
python src/inference.py --checkpoint checkpoints/best_model.pt \
    --source path/to/steak.jpg --image
```

Press **q** to quit the live feed.

---

## Evaluation

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt --data data
```

Outputs a classification report (per-class precision / recall / F1) and saves
`confusion_matrix.png`.

---

## Generate Checkpoint 1 Slides

```bash
python generate_slides.py
# → checkpoint1_slides.pdf  (8 slides)
```

---

## Code Authorship

All code in this repository was written from scratch for this project:

| File | Description | Lines written by author |
|------|-------------|------------------------|
| `src/model.py` | EfficientNet-B0 wrapper + custom head | All (1–125) |
| `src/dataset.py` | Transform pipeline + DataLoader builder | All (1–120) |
| `src/train.py` | Training loop, early stopping, scheduler | All (1–105) |
| `src/inference.py` | OpenCV real-time loop + overlay renderer | All (1–155) |
| `src/evaluate.py` | Metrics + speed benchmark | All (1–75) |
| `scripts/download_dataset.py` | Dataset download / generation | All (1–130) |
| `generate_slides.py` | PDF slide generator | All (1–470) |

External libraries used (standard, unmodified):
`torch`, `torchvision` (EfficientNet-B0 pre-trained weights from ImageNet),
`opencv-python`, `matplotlib`, `scikit-learn`.
