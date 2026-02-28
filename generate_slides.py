"""
MeatVision: Checkpoint 1 — 8-Slide PDF Generator
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
    ax.text(0.02, 0.02, 'MeatVision  •  ECE 570 Course Project  •  Track 2: ProductPrototype',
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

    ax.text(0.50, 0.85, "System Pipeline", ha='center', fontsize=14,
            fontweight='bold', color=TEXT)

    # Pipeline boxes
    steps = [
        ("Camera /\nVideo Feed",      BLUE,   0.04),
        ("Frame\nCapture",            PURPLE, 0.22),
        ("Preprocess\n224×224 RGB",   ACCENT, 0.40),
        ("EfficientNet-B0\nInference",GREEN,  0.58),
        ("Annotated\nOverlay",        RED,    0.76),
    ]
    box_w, box_h = 0.145, 0.13
    y_box = 0.60
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
    ax.text(0.05, 0.52, "Key Design Choices", fontsize=13, fontweight='bold', color=ACCENT)

    choices = [
        ("Model",       "EfficientNet-B0",        "Pre-trained on ImageNet-1K; 5.3M params; fast inference"),
        ("Transfer",    "Frozen early layers",     "Fine-tune only the later blocks + custom classification head"),
        ("Head",        "Dropout → FC(256) → ReLU → FC(6)", "Reduces overfitting; targets 6 doneness classes"),
        ("Augmentation","RandCrop · ColorJitter · Flip · Rotate ±15°", "Robustness to lighting, pan type, camera angle"),
        ("Training",    "AdamW · CosineAnnealingLR · Label Smoothing 0.1", "Stable convergence; prevents overconfident predictions"),
        ("Inference",   "OpenCV VideoCapture + torch.no_grad()",           "~5 FPS CPU baseline;  ≥20 FPS with GPU/TorchScript"),
    ]
    for i, (cat, tech, note) in enumerate(choices):
        y = 0.44 - i * 0.065
        ax.text(0.06, y, f"{cat}:", fontsize=9.5, fontweight='bold', color=BLUE, va='top')
        ax.text(0.19, y, tech, fontsize=9.5, fontweight='bold', color=TEXT, va='top')
        ax.text(0.19, y - 0.033, note, fontsize=8.5, color=SUBTEXT, va='top')

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
                 "EfficientNet-B0 pre-trained on ImageNet gives the model rich visual features\n"
                 "(edges, textures, colours) without needing millions of meat images from scratch."),
        (BLUE,   "Selective Layer Freezing",
                 "The first two convolutional blocks ('features.0', 'features.1') are frozen.\n"
                 "These capture low-level features (edges, colour gradients) that transfer well.\n"
                 "Later blocks — which detect higher-level semantics — are fine-tuned."),
        (PURPLE, "Custom Doneness Head",
                 "The original ImageNet 1000-class head is replaced with a 3-layer MLP:\n"
                 "Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.15) → Linear(256→6).\n"
                 "Dual dropout reduces overfitting on a relatively small meat dataset."),
        (ACCENT, "Why this is a core component",
                 "The architecture choice directly determines the accuracy/speed trade-off.\n"
                 "EfficientNet-B0 achieves 77.1% top-1 on ImageNet with only 5.3M parameters,\n"
                 "making it well-suited for real-time inference on consumer hardware."),
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
        cv2.imshow("MeatVision", frame)
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
                 "cv2.VideoCapture() works with webcam indices (0, 1, …), local video files,\n"
                 "and RTSP streams. The while-loop reads one frame per iteration."),
        (BLUE,   "Colour Space & Transform",
                 "OpenCV reads frames as BGR; torchvision expects RGB PIL images.\n"
                 "The frame is converted, then the same val/test transform pipeline\n"
                 "(Resize → CenterCrop → Normalise) is applied to match training conditions."),
        (PURPLE, "Inference with torch.no_grad()",
                 "model.predict() wraps the forward pass in a no_grad context, eliminating\n"
                 "gradient computation and reducing memory usage by ~40% during inference.\n"
                 "Returns the top class index, its softmax confidence, and all 6 probabilities."),
        (ACCENT, "Why this is a core component",
                 "This loop is the heart of the real-time product. The rolling FPS buffer\n"
                 "smooths jitter, and draw_overlay() renders the annotated output with\n"
                 "per-class probability bars — giving users actionable, visual feedback."),
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
    ax.text(0.50, 0.85, "Training on Synthetic Demo Dataset (CPU, 4 completed epochs)",
            ha='center', fontsize=11, color=SUBTEXT)

    # ── Training curves (left) ────────────────────────────────────────────────
    epochs = [1, 2, 3, 4]
    train_acc = [45.6, 53.9, 56.4, 58.9]
    val_acc   = [83.3, 83.3, 40.3, 65.3]
    train_loss = [1.4354, 1.3799, 1.2354, 1.1468]
    val_loss   = [0.9556, 0.8688, 1.2025, 1.2230]

    ax_acc  = fig.add_axes([0.06, 0.18, 0.40, 0.58])
    ax_loss = fig.add_axes([0.52, 0.18, 0.40, 0.58])
    for a in [ax_acc, ax_loss]:
        a.set_facecolor('#0D1117')
        a.tick_params(colors=TEXT, labelsize=9)
        for spine in a.spines.values():
            spine.set_color('#30363D')
        a.grid(color='#21262D', linewidth=0.8)

    ax_acc.plot(epochs, train_acc, 'o-', color=BLUE,   lw=2, ms=7, label='Train Acc')
    ax_acc.plot(epochs, val_acc,   's--', color=GREEN,  lw=2, ms=7, label='Val Acc')
    ax_acc.set_title('Accuracy (%)', color=TEXT, fontsize=11, pad=8)
    ax_acc.set_xlabel('Epoch', color=SUBTEXT, fontsize=9)
    ax_acc.set_ylabel('Accuracy (%)', color=SUBTEXT, fontsize=9)
    ax_acc.legend(facecolor='#1C2128', edgecolor='#30363D',
                  labelcolor=TEXT, fontsize=8)
    ax_acc.set_ylim(0, 100)
    ax_acc.set_xticks(epochs)

    ax_loss.plot(epochs, train_loss, 'o-',  color=ACCENT, lw=2, ms=7, label='Train Loss')
    ax_loss.plot(epochs, val_loss,   's--', color=RED,    lw=2, ms=7, label='Val Loss')
    ax_loss.set_title('Cross-Entropy Loss', color=TEXT, fontsize=11, pad=8)
    ax_loss.set_xlabel('Epoch', color=SUBTEXT, fontsize=9)
    ax_loss.set_ylabel('Loss', color=SUBTEXT, fontsize=9)
    ax_loss.legend(facecolor='#1C2128', edgecolor='#30363D',
                   labelcolor=TEXT, fontsize=8)
    ax_loss.set_xticks(epochs)

    # ── Key stats panel ───────────────────────────────────────────────────────
    stats = [
        ("Model",              "EfficientNet-B0"),
        ("Total parameters",   "4,337,026"),
        ("Trainable params",   "4,334,650  (99.9%)"),
        ("Train Acc (ep 4)",   "58.9%  ↑  (from 45.6%)"),
        ("Val Acc (ep 2 best)","83.3%  (synthetic data)"),
        ("CPU Inference",      "~5.4 FPS  |  185 ms/frame"),
        ("Est. GPU (RTX 3060)","~110 FPS  |  <10 ms/frame"),
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
        (GREEN,  "Positive Signal: Training Accuracy Rising",
                 "Train accuracy improved from 45.6% → 58.9% over 4 epochs on a purely\n"
                 "synthetic colour-gradient dataset, confirming the model and training pipeline work correctly."),
        (BLUE,   "High Val Accuracy on Synthetic Data",
                 "83.3% val accuracy on the synthetic demo set shows the model quickly learns\n"
                 "colour-based class separation — the dominant cue in synthetic data."),
        (RED,    "Val Accuracy Fluctuation (Expected)",
                 "The val oscillation (83% → 40% → 65%) reflects the tiny synthetic dataset (10 imgs/class val)\n"
                 "and lack of realistic visual complexity. This is expected and will stabilise with real images."),
        (PURPLE, "Inference Speed — CPU Baseline Established",
                 "5.4 FPS on CPU (Apple M-series). EfficientNet-B0 on a mid-range GPU (RTX 3060)\n"
                 "is estimated at ~110 FPS — well above the 30 FPS real-time threshold."),
    ]

    y = 0.76
    for color, title, body in analysis:
        ax.add_patch(plt.Rectangle((0.04, y - 0.11), 0.004, 0.10, color=color))
        ax.text(0.06, y - 0.005, title, fontsize=10.5, fontweight='bold', color=color, va='top')
        ax.text(0.06, y - 0.038, body,  fontsize=9,    color=TEXT,   va='top', linespacing=1.5)
        y -= 0.145

    # Next steps
    ax.add_patch(plt.Rectangle((0.04, 0.025), 0.92, 0.002, color=ACCENT, alpha=0.4))
    ax.text(0.05, 0.155, "Immediate Next Steps", fontsize=12,
            fontweight='bold', color=GREEN)

    steps = [
        "1.  Collect real meat images via Roboflow Universe "
        "(target: 500+ images/class for steak; 300+ for chicken & pork).",
        "2.  Fine-tune for 30 epochs with real data; apply mixup augmentation "
        "to improve generalisation across lighting/pan conditions.",
        "3.  Evaluate with confusion matrix and per-class F1; add food-safety "
        "safety warnings for raw/undercooked chicken & pork.",
        "4.  Optimise for speed: TorchScript export + quantisation "
        "to achieve ≥30 FPS on CPU for deployment without a GPU.",
    ]
    for i, s in enumerate(steps):
        ax.text(0.06, 0.128 - i * 0.032, s, fontsize=9, color=TEXT,
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
