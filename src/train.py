# GrillSight: training loop with validation, early stopping, and cosine LR.

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import build_dataloaders, get_class_weights
from model import get_model, count_parameters


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    # Run one epoch of training or validation; return (loss, accuracy).
    model.train() if train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, labels in tqdm(loader, desc='Train' if train else 'Val', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss    += loss.item() * imgs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


class EarlyStopping:
    # Stop training when val loss has not improved for `patience` epochs.

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter   = 0
        self.stop      = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Build data loaders.
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_root=args.data,
        batch_size=args.batch,
        num_workers=args.workers,
    )
    num_classes = len(class_names)

    # Build the model.
    model = get_model(num_classes=num_classes, device=device)
    params = count_parameters(model)
    print(f"Parameters - Total: {params['total']:,}  Trainable: {params['trainable']:,}")

    # Class-weighted cross-entropy loss with label smoothing.
    class_weights = get_class_weights(args.data, class_names, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stop = EarlyStopping(patience=args.patience)

    # Prepare output directory.
    ckpt_dir = Path(args.output)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"\nTraining for up to {args.epochs} epochs ...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f} | "
              f"{elapsed:.1f}s")

        # Save the best checkpoint.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, ckpt_dir / 'best_model.pt')

        if early_stop.step(val_loss):
            print(f"Early stopping at epoch {epoch}.")
            break

    # Persist training history.
    with open(ckpt_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Test-set evaluation.
    print("\nEvaluating on test set ...")
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False)
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GrillSight classifier')
    parser.add_argument('--data',     default='data',       help='Dataset root')
    parser.add_argument('--output',   default='checkpoints',help='Checkpoint dir')
    parser.add_argument('--epochs',   type=int, default=30)
    parser.add_argument('--batch',    type=int, default=32)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--workers',  type=int, default=4)
    args = parser.parse_args()

    train(args)
