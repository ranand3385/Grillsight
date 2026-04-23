# GrillSight: EfficientNet-B0 meat doneness classifier.

import ssl
import torch
import torch.nn as nn
from torchvision import models

# Workaround for macOS Python 3.11 SSL cert verification.
ssl._create_default_https_context = ssl._create_unverified_context


# 6-class beef doneness scale.
BEEF_CLASSES = ['raw', 'rare', 'medium_rare', 'medium', 'medium_well', 'well_done']

# Binary safety scale for poultry and pork.
POULTRY_PORK_CLASSES = ['raw', 'cooked']

# Display labels for the UI overlay.
CLASS_DISPLAY_NAMES = {
    'raw':         'Raw',
    'rare':        'Rare',
    'medium_rare': 'Medium Rare',
    'medium':      'Medium',
    'medium_well': 'Medium Well',
    'well_done':   'Well Done',
    'cooked':      'Cooked',
}

# BGR colour per class for OpenCV drawing.
CLASS_COLORS = {
    'raw':         (0, 0, 200),
    'rare':        (0, 100, 255),
    'medium_rare': (0, 180, 255),
    'medium':      (0, 200, 100),
    'medium_well': (0, 220, 0),
    'well_done':   (0, 140, 0),
    'cooked':      (0, 220, 0),
}


class MeatDonennessClassifier(nn.Module):
    # EfficientNet-B0 backbone with custom 6-class doneness head.

    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # ImageNet-pretrained backbone.
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)

        # Freeze features.0 and features.1; fine-tune the rest.
        for name, param in backbone.named_parameters():
            if 'features.0' in name or 'features.1' in name:
                param.requires_grad = False

        # Custom doneness classification head.
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
        # Return (class_index, confidence, full_probs) for a batch.
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred, conf, probs


def get_model(num_classes: int = 6, checkpoint_path: str = None,
              device: str = 'cpu') -> MeatDonennessClassifier:
    # Build and optionally load a classifier on the given device.
    model = MeatDonennessClassifier(num_classes=num_classes)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model.to(device)


def count_parameters(model: nn.Module) -> dict:
    # Count total and trainable parameters.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    model = get_model(num_classes=6)
    params = count_parameters(model)
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    # Sanity-check forward pass.
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
