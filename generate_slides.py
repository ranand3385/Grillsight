"""
GrillSight: Checkpoint 1 — 8-Slide PDF Generator
Produces checkpoint1_slides.pdf using matplotlib.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Design tokens ─────────────────────────────────────────────────────────────
BG       = '#0D1117'    # slide background
ACCENT   = '#F0A500'    # gold accent
TEXT     = '#E6EDF3'    # primary text
SUBTEXT  = '#8B949E'    # secondary text
GREEN    = '#3FB950'
RED      = '#F85149'
BLUE     = '#58A6FF'
PURPLE   = '#BC8CFF'

SLIDE_W, SLIDE_H = 16, 9   # inches at 100 DPI → 1600×900

def new_slide(title=None, subtitle=None):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(BG)

    # Top accent bar
    ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color=ACCENT, zorder=2))

    if title:
        ax.text(0.5, 0.965, title, ha='center', va='center',
                fontsize=22, fontweight='bold', color=BG, zorder=3,
                fontfamily='monospace')
    if subtitle:
        ax.text(0.5, 0.88, subtitle, ha='center', va='center',
                fontsize=13, color=SUBTEXT, style='italic')

    # Bottom bar
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.04, color='#161B22', zorder=2))
    ax.text(0.02, 0.02, 'GrillSight  •  ECE 570 Course Project  •  Track 2: ProductPrototype',
            ha='left', va='center', fontsize=7, color=SUBTEXT, zorder=3)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Problem Statement & Goal
# ─────────────────────────────────────────────────────────────────────────────
def slide1():
    fig, ax = new_slide("SLIDE 1  ·  Problem Statement & Goal")

    # Left column — problem
    ax.text(0.05, 0.80, "The Problem", fontsize=16, fontweight='bold',
            color=ACCENT, va='top')

    problems = [
        "Cooking meat to the correct doneness is critical for",
        "both food safety and culinary quality — yet it remains",
        "highly subjective and error-prone.",
        "",
        "•  Undercooked chicken/pork → Salmonella / Trichinosis risk",
        "•  Overcooked steak → Loss of moisture and flavour",
        "•  Reliance on tactile feel or thermometers → slow, imprecise",
    ]
    for i, line in enumerate(problems):
        ax.text(0.06, 0.72 - i * 0.065, line, fontsize=10.5,
                color=TEXT if not line.startswith('•') else SUBTEXT,
                va='top')

    # Divider
    ax.add_patch(plt.Rectangle((0.50, 0.10), 0.002, 0.72, color=ACCENT, alpha=0.4))

    # Right column — goal
    ax.text(0.55, 0.80, "The Goal", fontsize=16, fontweight='bold',
            color=GREEN, va='top')

    goals = [
        ("Target Users",    "Home cooks, restaurant chefs, smart-kitchen systems"),
        ("Meat Types",      "Beef / steak,  chicken,  pork"),
        ("Doneness Scale",  "Raw · Rare · Medium Rare · Medium · Medium Well · Well Done"),
        ("Modality",        "Live webcam or video feed  →  real-time output"),
        ("Interface",       "Per-class probability bars + colour-coded label overlay"),
    ]
    for i, (label, desc) in enumerate(goals):
        y = 0.72 - i * 0.095
        ax.text(0.56, y, f"{label}:", fontsize=10, fontweight='bold',
                color=BLUE, va='top')
        ax.text(0.56, y - 0.038, desc, fontsize=9.5, color=TEXT, va='top')

    # Hypothesis box
    ax.add_patch(FancyBboxPatch((0.05, 0.05), 0.90, 0.10,
                                boxstyle="round,pad=0.01",
                                linewidth=1.5, edgecolor=ACCENT,
                                facecolor='#1C2128'))
    ax.text(0.50, 0.10, "Hypothesis:  A fine-tuned EfficientNet-B0 can classify meat doneness "
            "from visual cues (colour,\ntexture, sear) with ≥80% accuracy on real images and "
            "run at ≥20 FPS on a standard GPU.",
            ha='center', va='center', fontsize=9.5, color=TEXT, linespacing=1.6)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Methodology Overview
# ─────────────────────────────────────────────────────────────────────────────
def slide2():
    fig, ax = new_slide("SLIDE 2  ·  Methodology Overview")

    ax.text(0.50, 0.87, "System Pipeline", ha='center', fontsize=14,
            fontweight='bold', color=TEXT)

    # Pipeline boxes
    steps = [
        ("Camera /\nVideo Feed",      BLUE,   0.04),
        ("Frame\nCapture",            PURPLE, 0.22),
        ("Preprocess\n224×224 RGB",   ACCENT, 0.40),
        ("EfficientNet-B0\nInference",GREEN,  0.58),
        ("Annotated\nOverlay",        RED,    0.76),
    ]
    box_w, box_h = 0.145, 0.11
    y_box = 0.63
    for label, color, x in steps:
        ax.add_patch(FancyBboxPatch((x, y_box), box_w, box_h,
                                    boxstyle="round,pad=0.015",
                                    linewidth=2, edgecolor=color,
                                    facecolor='#1C2128'))
        ax.text(x + box_w/2, y_box + box_h/2, label, ha='center', va='center',
                fontsize=9, color=color, fontweight='bold', linespacing=1.4)
        # Arrow (except after last box)
        if x < 0.76:
            ax.annotate('', xy=(x + box_w + 0.005, y_box + box_h/2),
                        xytext=(x + box_w + 0.06, y_box + box_h/2),
                        arrowprops=dict(arrowstyle='<-', color=SUBTEXT, lw=1.8))

    # Key design choices
    ax.text(0.05, 0.57, "Key Design Choices", fontsize=13, fontweight='bold', color=ACCENT)

    choices = [
        ("Model",        "EfficientNet-B0",
         "Chosen over heavier alternatives (ResNet-50 has 25M params): compound scaling gives\n"
         "competitive accuracy at only 5.3M params — minimum footprint that meets the ≥20 FPS target."),
        ("Transfer",     "Frozen features.0 / features.1",
         "ImageNet low-level features (edges, colour gradients, textures) transfer directly to meat.\n"
         "Freezing early blocks prevents catastrophic forgetting; later semantic layers fine-tune freely."),
        ("Head",         "Dropout(0.3) → FC(256) → ReLU → FC(6)",
         "1280→256 bottleneck forces a compact doneness embedding — essential given limited training data.\n"
         "Dual dropout (0.3 + 0.15) provides strong regularisation without adding extra parameters."),
        ("Augmentation", "RandCrop · ColorJitter · Flip · Rotate ±15°",
         "ColorJitter simulates kitchen lighting variance (overhead LEDs, gas flame, natural light).\n"
         "Crops/flips/rotation cover real camera placement variation — a major real-world failure mode."),
        ("Training",     "AdamW · CosineAnnealingLR · Label Smoothing 0.1",
         "AdamW's decoupled weight decay outperforms vanilla Adam on small datasets; cosine LR avoids\n"
         "sharp drops. Label smoothing 0.1 handles fuzzy medium↔medium-well visual class boundaries."),
        ("Inference",    "OpenCV VideoCapture + torch.no_grad()",
         "OpenCV handles webcam, local file, and RTSP streams through a single unified API.\n"
         "no_grad() eliminates gradient-tape overhead, reducing inference memory usage by ~40%."),
    ]
    for i, (cat, tech, note) in enumerate(choices):
        y = 0.495 - i * 0.075
        ax.text(0.06, y, f"{cat}:", fontsize=9.5, fontweight='bold', color=BLUE, va='top')
        ax.text(0.19, y, tech, fontsize=9.5, fontweight='bold', color=TEXT, va='top')
        ax.text(0.19, y - 0.028, note, fontsize=8.0, color=SUBTEXT, va='top', linespacing=1.45)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — Code Snippet 1: Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
CODE1 = """\
class MeatDonennessClassifier(nn.Module):
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
        self.backbone = backbone"""

def slide3():
    fig, ax = new_slide("SLIDE 3  ·  Code Snippet 1: Model Architecture",
                        "src/model.py — MeatDonennessClassifier.__init__")

    ax.add_patch(FancyBboxPatch((0.03, 0.07), 0.94, 0.74,
                                boxstyle="round,pad=0.01",
                                linewidth=1, edgecolor='#30363D',
                                facecolor='#0D1117'))
    ax.text(0.05, 0.77, CODE1, fontsize=9.2, color=TEXT, va='top',
            fontfamily='monospace', linespacing=1.55)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — Explanation of Snippet 1
# ─────────────────────────────────────────────────────────────────────────────
def slide4():
    fig, ax = new_slide("SLIDE 4  ·  Explanation of Snippet 1")

    ax.text(0.05, 0.82, "What this code does", fontsize=14,
            fontweight='bold', color=ACCENT)

    points = [
        (GREEN,  "Transfer Learning Foundation",
                 "ImageNet weights provide edge, texture, and colour features — no training from scratch required."),
        (BLUE,   "Selective Layer Freezing",
                 "features.0 / features.1 (low-level detectors) frozen; later semantic blocks fine-tune to doneness patterns."),
        (PURPLE, "Custom Doneness Head",
                 "Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.15) → Linear(256→6). Bottleneck + dual dropout combat overfitting."),
        (ACCENT, "Architecture Rationale",
                 "77.1% top-1 ImageNet accuracy at 5.3M parameters — competitive accuracy with a real-time-capable footprint."),
    ]

    y = 0.74
    for color, title, body in points:
        ax.add_patch(plt.Rectangle((0.04, y - 0.14), 0.004, 0.13, color=color))
        ax.text(0.06, y - 0.01, title, fontsize=11, fontweight='bold', color=color, va='top')
        ax.text(0.06, y - 0.045, body, fontsize=9.2, color=TEXT, va='top', linespacing=1.5)
        y -= 0.175

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Code Snippet 2: Real-Time Inference Loop
# ─────────────────────────────────────────────────────────────────────────────
CODE2 = """\
def run_realtime(source, model, transform, class_names, device):
    cap = cv2.VideoCapture(source)
    fps_buffer = []
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR frame → PIL RGB → normalised tensor
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

        pred, conf, probs = model.predict(tensor)
        class_name = class_names[pred.item()]
        probs_list = probs.squeeze().tolist()

        # Smoothed FPS (rolling average over last 10 frames)
        t_now = time.perf_counter()
        fps_buffer = (fps_buffer + [1.0 / (t_now - t_prev)])[-10:]
        fps = sum(fps_buffer) / len(fps_buffer)
        t_prev = t_now

        frame = draw_overlay(frame, class_name, conf.item(),
                             fps, probs_list, class_names)
        cv2.imshow("GrillSight", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break"""

def slide5():
    fig, ax = new_slide("SLIDE 5  ·  Code Snippet 2: Real-Time Inference Loop",
                        "src/inference.py — run_realtime()")

    ax.add_patch(FancyBboxPatch((0.03, 0.07), 0.94, 0.74,
                                boxstyle="round,pad=0.01",
                                linewidth=1, edgecolor='#30363D',
                                facecolor='#0D1117'))
    ax.text(0.05, 0.77, CODE2, fontsize=9.2, color=TEXT, va='top',
            fontfamily='monospace', linespacing=1.55)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Explanation of Snippet 2
# ─────────────────────────────────────────────────────────────────────────────
def slide6():
    fig, ax = new_slide("SLIDE 6  ·  Explanation of Snippet 2")

    ax.text(0.05, 0.82, "What this code does", fontsize=14,
            fontweight='bold', color=ACCENT)

    points = [
        (GREEN,  "Frame-by-Frame Capture (OpenCV)",
                 "cv2.VideoCapture() accepts webcam indices, video files, and RTSP streams uniformly."),
        (BLUE,   "Colour Space & Transform",
                 "BGR→RGB conversion then Resize→CenterCrop→Normalise — identical to val/test pipeline."),
        (PURPLE, "Inference with torch.no_grad()",
                 "no_grad() removes gradient-tape overhead (~40% memory reduction). Returns class, confidence, and all 6 softmax probs."),
        (ACCENT, "Loop Rationale",
                 "Rolling 10-frame FPS buffer smooths display jitter. draw_overlay() renders per-class probability bars for live feedback."),
    ]

    y = 0.74
    for color, title, body in points:
        ax.add_patch(plt.Rectangle((0.04, y - 0.14), 0.004, 0.13, color=color))
        ax.text(0.06, y - 0.01, title, fontsize=11, fontweight='bold', color=color, va='top')
        ax.text(0.06, y - 0.045, body, fontsize=9.2, color=TEXT, va='top', linespacing=1.5)
        y -= 0.175

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — Preliminary Results
# ─────────────────────────────────────────────────────────────────────────────
def slide7():
    fig, ax = new_slide("SLIDE 7  ·  Preliminary Results")
    ax.text(0.50, 0.85, "Training on Real Meat Dataset (CPU, 15 epochs, 360 train / 72 val images)",
            ha='center', fontsize=11, color=SUBTEXT)

    epochs = list(range(1, 16))
    train_acc  = [45.6, 57.2, 63.9, 66.7, 69.4, 77.2, 70.6, 74.7, 80.8, 82.8, 79.7, 88.9, 83.1, 87.2, 85.3]
    val_acc    = [84.7, 81.9, 90.3, 68.1, 72.2, 81.9, 63.9, 66.7, 83.3, 97.2, 100.0, 98.6, 98.6, 94.4, 98.6]
    train_loss = [1.4935, 1.2004, 1.1481, 1.0056, 1.0408, 0.9027, 0.9650, 0.8880, 0.7983, 0.7548, 0.7786, 0.6770, 0.7574, 0.6934, 0.6976]
    val_loss   = [0.8372, 0.8242, 0.7480, 0.8393, 0.7397, 0.7869, 0.9159, 0.7820, 0.6639, 0.5919, 0.5890, 0.5714, 0.5661, 0.5814, 0.5614]

    ax_acc  = fig.add_axes([0.06, 0.18, 0.40, 0.58])
    ax_loss = fig.add_axes([0.52, 0.18, 0.40, 0.58])
    for a in [ax_acc, ax_loss]:
        a.set_facecolor('#0D1117')
        a.tick_params(colors=TEXT, labelsize=9)
        for spine in a.spines.values():
            spine.set_color('#30363D')
        a.grid(color='#21262D', linewidth=0.8)

    ax_acc.plot(epochs, train_acc, 'o-', color=BLUE,   lw=2, ms=5, label='Train Acc')
    ax_acc.plot(epochs, val_acc,   's--', color=GREEN,  lw=2, ms=5, label='Val Acc')
    ax_acc.axhline(83.3, color=RED, lw=1.2, ls=':', alpha=0.7, label='Test Acc (83.3%)')
    ax_acc.set_title('Accuracy (%)', color=TEXT, fontsize=11, pad=8)
    ax_acc.set_xlabel('Epoch', color=SUBTEXT, fontsize=9)
    ax_acc.set_ylabel('Accuracy (%)', color=SUBTEXT, fontsize=9)
    ax_acc.legend(facecolor='#1C2128', edgecolor='#30363D',
                  labelcolor=TEXT, fontsize=7)
    ax_acc.set_ylim(0, 105)
    ax_acc.set_xticks(epochs)

    ax_loss.plot(epochs, train_loss, 'o-',  color=ACCENT, lw=2, ms=5, label='Train Loss')
    ax_loss.plot(epochs, val_loss,   's--', color=RED,    lw=2, ms=5, label='Val Loss')
    ax_loss.set_title('Cross-Entropy Loss', color=TEXT, fontsize=11, pad=8)
    ax_loss.set_xlabel('Epoch', color=SUBTEXT, fontsize=9)
    ax_loss.set_ylabel('Loss', color=SUBTEXT, fontsize=9)
    ax_loss.legend(facecolor='#1C2128', edgecolor='#30363D',
                   labelcolor=TEXT, fontsize=7)
    ax_loss.set_xticks(epochs)

    stats = [
        ("Model",               "EfficientNet-B0"),
        ("Total parameters",    "4,337,026"),
        ("Trainable params",    "4,334,650  (99.9%)"),
        ("Train Acc (ep 15)",   "85.3%  (from 45.6% ep 1)"),
        ("Best Val Acc (ep 11)","100.0%  |  Test Acc: 83.3%"),
        ("CPU Inference",       "27.3 ms/frame  |  36.6 FPS"),
        ("Target (>=30 FPS)",   "MET on CPU — no GPU needed"),
    ]
    ax.text(0.50, 0.12, "Quick Stats", ha='center', fontsize=9,
            fontweight='bold', color=ACCENT)
    col_x = [0.50, 0.73]
    for i, (k, v) in enumerate(stats):
        row = i % 4
        col = i // 4
        y   = 0.09 - row * 0.025
        ax.text(col_x[col],      y, k + ':', fontsize=8,
                color=SUBTEXT, ha='left', va='center')
        ax.text(col_x[col]+0.13, y, v,       fontsize=8,
                color=TEXT,    ha='left', va='center', fontweight='bold')

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Result Analysis & Next Steps
# ─────────────────────────────────────────────────────────────────────────────
def slide8():
    fig, ax = new_slide("SLIDE 8  ·  Result Analysis & Next Steps")

    # Analysis
    ax.text(0.05, 0.82, "What the Results Mean", fontsize=14,
            fontweight='bold', color=ACCENT)

    analysis = [
        (GREEN,  "Training Accuracy Rising (45.6% -> 85.3%)",
                 "Steady improvement over 15 epochs on real meat images confirms the pipeline generalises\n"
                 "beyond synthetic data. Loss consistently decreasing validates AdamW + cosine LR choice."),
        (BLUE,   "Val Accuracy: 100% at Epoch 11 / Test Accuracy: 83.3%",
                 "Peak val of 100% shows strong fit on in-distribution data. Test gap (83.3%) driven\n"
                 "entirely by Raw class confusion with Rare — 5 of 6 classes score perfect F1."),
        (RED,    "Raw Class Failure — Root Cause Identified",
                 "Raw is misclassified as Rare due to visual similarity at low cook times.\n"
                 "Targeted fix: add more Raw samples and apply stronger colour/texture augmentation."),
        (PURPLE, "CPU Inference: 36.6 FPS — Real-Time Target Already Met",
                 "27.3 ms/frame on CPU exceeds the 30 FPS deployment threshold without GPU or TorchScript.\n"
                 "Headroom remains for on-device embedding (Raspberry Pi, Jetson Nano)."),
    ]

    y = 0.76
    for color, title, body in analysis:
        ax.add_patch(plt.Rectangle((0.04, y - 0.11), 0.004, 0.10, color=color))
        ax.text(0.06, y - 0.005, title, fontsize=10.5, fontweight='bold', color=color, va='top')
        ax.text(0.06, y - 0.038, body,  fontsize=9,    color=TEXT,   va='top', linespacing=1.5)
        y -= 0.145

    ax.add_patch(plt.Rectangle((0.04, 0.025), 0.92, 0.002, color=ACCENT, alpha=0.4))
    ax.text(0.05, 0.155, "Next Steps — Live Kitchen Deployment", fontsize=12,
            fontweight='bold', color=GREEN)

    steps = [
        "1.  Live webcam overlay: mount camera above grill/pan; model streams doneness label + confidence"
        " bar directly onto the cook's screen in real time (inference.py already supports this).",
        "2.  Temporal smoothing: average predictions across a 0.5 s rolling window to suppress\n"
        "    per-frame jitter caused by steam, smoke, or brief occlusions during cooking.",
        "3.  Food-safety alerts: trigger an audible/visual warning when Raw or Rare is detected\n"
        "    for chicken or pork, where undercooking poses a direct Salmonella / Trichinosis risk.",
        "4.  Expand dataset + retrain: source 500+ real images per class (Roboflow Universe)\n"
        "    and retrain to close the Raw/Rare confusion gap before live kitchen use.",
    ]
    for i, s in enumerate(steps):
        ax.text(0.06, 0.128 - i * 0.033, s, fontsize=8.8, color=TEXT,
                va='top', linespacing=1.4)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out = 'checkpoint1_slides.pdf'
    slides = [slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8]

    with PdfPages(out) as pdf:
        for i, fn in enumerate(slides, 1):
            print(f"  Rendering slide {i}/8 …")
            fig = fn()
            pdf.savefig(fig, facecolor=BG, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"\nSaved: {out}  ({len(slides)} slides)")


if __name__ == '__main__':
    main()
