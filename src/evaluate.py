# GrillSight: classification report, confusion matrix, and CPU inference benchmark.

import argparse
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import build_dataloaders
from model import get_model, CLASS_DISPLAY_NAMES


def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint and class names.
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt['class_names']
    num_classes  = len(class_names)

    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, _, test_loader, _ = build_dataloaders(
        args.data, batch_size=args.batch, num_workers=2)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Per-class classification report.
    display_names = [CLASS_DISPLAY_NAMES.get(c, c) for c in class_names]
    report = classification_report(
        all_labels, all_preds, target_names=display_names)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix figure.
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=display_names)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('GrillSight - Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to confusion_matrix.png")

    # CPU inference-speed benchmark.
    print("\nBenchmarking inference speed ...")
    dummy = torch.randn(1, 3, 224, 224).to(device)
    # Warmup passes.
    for _ in range(10):
        model(dummy)
    t0 = time.perf_counter()
    reps = 200
    for _ in range(reps):
        model(dummy)
    elapsed = time.perf_counter() - t0
    ms_per_frame = elapsed / reps * 1000
    fps = reps / elapsed
    print(f"  Avg latency: {ms_per_frame:.2f} ms/frame  ->  {fps:.1f} FPS  (device={device})")

    return all_preds, all_labels, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data',       default='data')
    parser.add_argument('--batch',      type=int, default=32)
    args = parser.parse_args()
    evaluate(args)
