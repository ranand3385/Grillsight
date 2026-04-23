"""GrillSight: Offline augmentation to expand the training split.

Reads every image in data/train/<class>/ and writes N augmented copies
alongside the originals.  Val and test splits are left untouched.

Usage:
    python scripts/augment_dataset.py --data data --factor 5
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageFilter
from torchvision import transforms


# Per-class augmentation strength — harder-to-distinguish classes get
# stronger colour jitter so the model sees more variation.
CLASS_JITTER = {
    'raw':         dict(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.12),
    'rare':        dict(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.12),
    'medium_rare': dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
    'medium':      dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    'medium_well': dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    'well_done':   dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
}

BASE_AUGMENTS = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),
    transforms.RandomGrayscale(p=1.0),
    transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
]


def augment_image(img: Image.Image, cls: str, seed: int) -> Image.Image:
    random.seed(seed)
    jitter = transforms.ColorJitter(**CLASS_JITTER[cls])
    pipeline = transforms.Compose([
        transforms.RandomApply([BASE_AUGMENTS[seed % len(BASE_AUGMENTS)]], p=0.9),
        jitter,
        transforms.Resize((224, 224)),
    ])
    return pipeline(img)


def expand_split(data_root: Path, factor: int):
    train_dir = data_root / 'train'
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    total_written = 0
    for cls in classes:
        cls_dir = train_dir / cls
        originals = sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.png'))

        written = 0
        for img_path in originals:
            img = Image.open(img_path).convert('RGB')
            for k in range(factor):
                seed = hash((img_path.name, k)) & 0xFFFFFF
                aug = augment_image(img, cls, seed)
                out_name = f"{img_path.stem}_aug{k:03d}{img_path.suffix}"
                aug.save(cls_dir / out_name)
                written += 1

        print(f"  {cls:14s}: {len(originals)} originals -> "
              f"{len(originals) + written} total ({written} added)")
        total_written += written

    print(f"\nDone. {total_written} augmented images written to {train_dir}")


def main():
    parser = argparse.ArgumentParser(description='Offline training-set augmentation')
    parser.add_argument('--data',   default='data', help='Dataset root')
    parser.add_argument('--factor', type=int, default=5,
                        help='Augmented copies per original image')
    args = parser.parse_args()

    print(f"Expanding train split by factor {args.factor} ...")
    expand_split(Path(args.data), args.factor)


if __name__ == '__main__':
    main()
