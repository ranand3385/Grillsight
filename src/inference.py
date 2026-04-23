# GrillSight: real-time meat doneness detection from webcam, video, or single image.

import argparse
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

from dataset import get_inference_transform
from model import (
    CLASS_COLORS,
    CLASS_DISPLAY_NAMES,
    BEEF_CLASSES,
    get_model,
)


def draw_overlay(frame, class_name: str, confidence: float, fps: float,
                 all_probs: list, class_names: list):
    # Render doneness label, probability bars, and FPS onto a BGR frame in-place.
    h, w = frame.shape[:2]
    color = CLASS_COLORS.get(class_name, (200, 200, 200))
    display = CLASS_DISPLAY_NAMES.get(class_name, class_name)

    # Translucent header banner.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Prediction label.
    label = f"{display}  {confidence:.0%}"
    cv2.putText(frame, label, (15, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 3, cv2.LINE_AA)

    # FPS counter.
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    # Per-class probability bars at the bottom.
    bar_h     = 22
    bar_pad   = 8
    panel_top = h - (len(class_names) * (bar_h + bar_pad) + bar_pad + 30)
    overlay2  = frame.copy()
    cv2.rectangle(overlay2, (0, panel_top), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)

    for i, (cls, prob) in enumerate(zip(class_names, all_probs)):
        y = panel_top + bar_pad + i * (bar_h + bar_pad) + bar_h
        bar_color = CLASS_COLORS.get(cls, (160, 160, 160))
        bar_width = int((w - 220) * prob)

        # Bar background.
        cv2.rectangle(frame, (160, y - bar_h + 4), (w - 20, y + 4),
                      (60, 60, 60), -1)
        # Filled portion of the bar.
        if bar_width > 0:
            cv2.rectangle(frame, (160, y - bar_h + 4),
                          (160 + bar_width, y + 4), bar_color, -1)

        cls_label = CLASS_DISPLAY_NAMES.get(cls, cls)
        cv2.putText(frame, f"{cls_label}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{prob:.0%}", (w - 65, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def predict_image(image_path: str, model, transform, class_names, device):
    # Run inference on a single image file and display the result.
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    pred, conf, probs = model.predict(tensor)
    class_name = class_names[pred.item()]
    probs_list = probs.squeeze().tolist()

    print(f"\nPrediction: {CLASS_DISPLAY_NAMES.get(class_name, class_name)}")
    print(f"Confidence: {conf.item():.1%}")
    for cls, p in zip(class_names, probs_list):
        bar = '#' * int(p * 30)
        print(f"  {CLASS_DISPLAY_NAMES.get(cls, cls):14s} {bar:<30} {p:.1%}")

    # Render annotated image via OpenCV.
    frame = cv2.imread(image_path)
    frame = draw_overlay(frame, class_name, conf.item(), 0.0, probs_list, class_names)
    cv2.imshow("GrillSight - Single Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_realtime(source, model, transform, class_names, device):
    # Capture and annotate frames from a webcam index, video file, or RTSP URL.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print("GrillSight live feed started. Press 'q' to quit.")

    fps_buffer = []
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        tensor = transform(pil).unsqueeze(0).to(device)

        pred, conf, probs = model.predict(tensor)
        class_name = class_names[pred.item()]
        probs_list = probs.squeeze().tolist()

        # 10-frame rolling FPS average.
        t_now = time.perf_counter()
        fps_buffer.append(1.0 / max(t_now - t_prev, 1e-6))
        if len(fps_buffer) > 10:
            fps_buffer.pop(0)
        fps    = sum(fps_buffer) / len(fps_buffer)
        t_prev = t_now

        frame = draw_overlay(frame, class_name, conf.item(), fps,
                             probs_list, class_names)
        cv2.imshow("GrillSight - Live Doneness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live feed closed.")


def main():
    parser = argparse.ArgumentParser(description='GrillSight real-time inference')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--source', default=0,
                        help='Video source: 0 (webcam), path to file, or RTSP URL')
    parser.add_argument('--image', action='store_true',
                        help='Treat --source as a single image file')
    parser.add_argument('--device', default=None,
                        help='Device override: cpu or cuda (auto-detected if omitted)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Load trained checkpoint.
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt.get('class_names', BEEF_CLASSES)
    num_classes = len(class_names)

    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    transform = get_inference_transform()

    # Coerce numeric webcam indices from string.
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if args.image:
        predict_image(str(source), model, transform, class_names, device)
    else:
        run_realtime(source, model, transform, class_names, device)


if __name__ == '__main__':
    main()
