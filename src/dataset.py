# GrillSight: ImageFolder loaders, transforms, and class-weight helper.

from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# ImageNet normalisation statistics for transfer learning.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# EfficientNet-B0 input resolution.
IMAGE_SIZE = 224


def get_transforms(split: str) -> transforms.Compose:
    # Return the training or val/test transform pipeline.
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # Deterministic pipeline for val and test.
        return transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    # Build train, val, and test DataLoaders from an ImageFolder layout.
    root = Path(data_root)

    train_dataset = datasets.ImageFolder(
        root=str(root / 'train'),
        transform=get_transforms('train'),
    )
    val_dataset = datasets.ImageFolder(
        root=str(root / 'val'),
        transform=get_transforms('val'),
    )
    test_dataset = datasets.ImageFolder(
        root=str(root / 'test'),
        transform=get_transforms('test'),
    )

    class_names = train_dataset.classes

    # Weighted sampler balances class frequency across batches.
    sampler = None
    if use_weighted_sampler:
        targets = torch.tensor(train_dataset.targets)
        class_counts = torch.bincount(targets)
        weights = 1.0 / class_counts.float()
        sample_weights = weights[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


def get_inference_transform() -> transforms.Compose:
    # Single-frame transform used at inference time.
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_class_weights(data_root: str, class_names: list, device: str) -> torch.Tensor:
    # Compute inverse-frequency class weights from the train split.
    train_dir = Path(data_root) / 'train'
    counts = torch.tensor(
        [len(list((train_dir / c).glob('*'))) for c in class_names],
        dtype=torch.float,
    )
    weights = counts.sum() / (len(class_names) * counts)
    weights = weights / weights.sum() * len(class_names)
    print(f"Class weights: { {c: f'{w:.3f}' for c, w in zip(class_names, weights.tolist())} }")
    return weights.to(device)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data', help='Path to dataset root')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    loaders = build_dataloaders(args.data, batch_size=args.batch)
    train_loader, val_loader, test_loader, classes = loaders
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels: {labels}")
