# GrillSight: download from Roboflow or generate a synthetic demo dataset.

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path


def download_from_roboflow(api_key: str, workspace: str, project: str,
                           version: int, dest: str = 'data'):
    # Download a dataset from Roboflow via the export API.
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("roboflow package not found. Install with: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download(
        "folder", location=dest, overwrite=True)
    print(f"Dataset downloaded to: {dest}")
    return dataset


def create_demo_dataset(dest: str = 'data', images_per_class: int = 30):
    # Generate a synthetic colour-gradient dataset for smoke testing.
    try:
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        raise SystemExit("numpy and Pillow required: pip install numpy Pillow")

    CLASSES = ['raw', 'rare', 'medium_rare', 'medium', 'medium_well', 'well_done']

    # RGB colour centre per doneness level.
    CLASS_COLORS_RGB = {
        'raw':         (190,  60,  60),
        'rare':        (170,  50,  50),
        'medium_rare': (160,  80,  60),
        'medium':      (150, 100,  70),
        'medium_well': (130, 100,  80),
        'well_done':   (100,  70,  50),
    }

    rng = np.random.default_rng(42)

    for split in ('train', 'val', 'test'):
        n = images_per_class if split == 'train' else max(5, images_per_class // 5)
        for cls in CLASSES:
            out_dir = Path(dest) / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            r0, g0, b0 = CLASS_COLORS_RGB[cls]
            for i in range(n):
                # Add noise to simulate texture variation.
                noise = rng.integers(-30, 30, (224, 224, 3), dtype=np.int32)
                img_array = np.clip(
                    np.array([r0, g0, b0], dtype=np.int32) + noise, 0, 255
                ).astype(np.uint8)
                PILImage.fromarray(img_array).save(out_dir / f"{cls}_{i:04d}.jpg")

    print(f"Demo dataset written to '{dest}' ({images_per_class} imgs/class train).")
    print("WARNING: synthetic colour-gradient dataset for smoke-testing only.")

    # Write a manifest describing the generated dataset.
    manifest = {
        'type': 'synthetic_demo',
        'classes': CLASSES,
        'images_per_class_train': images_per_class,
    }
    with open(Path(dest) / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Download or generate the GrillSight dataset')
    subparsers = parser.add_subparsers(dest='command')

    # Roboflow subcommand.
    rf_parser = subparsers.add_parser('roboflow', help='Download from Roboflow Universe')
    rf_parser.add_argument('--api-key',   required=True)
    rf_parser.add_argument('--workspace', required=True)
    rf_parser.add_argument('--project',   required=True)
    rf_parser.add_argument('--version',   type=int, default=1)
    rf_parser.add_argument('--dest',      default='data')

    # Synthetic demo subcommand.
    demo_parser = subparsers.add_parser('demo', help='Create a synthetic demo dataset')
    demo_parser.add_argument('--dest', default='data')
    demo_parser.add_argument('--n',    type=int, default=30,
                              help='Images per class in the train split')

    args = parser.parse_args()

    if args.command == 'roboflow':
        download_from_roboflow(args.api_key, args.workspace, args.project, args.version, args.dest)
    elif args.command == 'demo':
        create_demo_dataset(args.dest, args.n)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
