
"""GrillSight: EfficientNet-B0 meat doneness classifier."""

import ssl
import torch
import torch.nn as nn
from torchvision import models

# Fix macOS Python 3.11 SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context


# Doneness classes for beef/steak (most granular scale)
BEEF_CLASSES = ['raw', 'rare', 'medium_rare', 'medium', 'medium_well', 'well_done']

# Safety-focused classes for poultry and pork
POULTRY_PORK_CLASSES = ['raw', 'cooked']

# Unified display names for UI
CLASS_DISPLAY_NAMES = {
    'raw':         'Raw',
    'rare':        'Rare',
    'medium_rare': 'Medium Rare',
    'medium':      'Medium',
    'medium_well': 'Medium Well',
    'well_done':   'Well Done',
    'cooked':      'Cooked',
}

# Color coding for doneness levels (BGR for OpenCV)
CLASS_COLORS = {
    'raw':         (0, 0, 200),     # Red  - danger
    'rare':        (0, 100, 255),   # Orange-red
    'medium_rare': (0, 180, 255),   # Orange
    'medium':      (0, 200, 100),   # Yellow-green
    'medium_well': (0, 220, 0),     # Green
    'well_done':   (0, 140, 0),     # Dark green
    'cooked':      (0, 220, 0),     # Green
}


class MeatDonennessClassifier(nn.Module):
    """EfficientNet-B0 backbone with custom 6-class doneness head."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Load EfficientNet-B0 pre-trained on ImageNet
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)

        # Freeze early convolutional layers; fine-tune later blocks
        for name, param in backbone.named_parameters():
            if 'features.0' in name or 'features.1' in name:
                param.requires_grad = False

        # Replace the classifier head for meat doneness
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, x: torch.Tensor):
        """Return (class_index, confidence) for a single batch."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred, conf, probs


def get_model(num_classes: int = 6, checkpoint_path: str = None,
              device: str = 'cpu') -> MeatDonennessClassifier:
    """Build and optionally load a MeatDonennessClassifier."""
    model = MeatDonennessClassifier(num_classes=num_classes)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model.to(device)


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    model = get_model(num_classes=6)
    params = count_parameters(model)
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    # Quick sanity-check forward pass
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: (2, 6)
