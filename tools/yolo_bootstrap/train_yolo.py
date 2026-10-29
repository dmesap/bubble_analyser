"""Train a lightweight YOLOv8-seg model on the pseudo-labelled bubble dataset.

This trains yolov8n-seg for bubble segmentation and outputs the weights to
bubble_analyser/weights/yolo_bubble_seg.pt.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "build" / "yolo_bootstrap_dataset" / "data.yaml"
WEIGHTS_DEST = REPO_ROOT / "bubble_analyser" / "weights" / "yolo_bubble_seg.pt"


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for training YOLO models. Run: pip install ultralytics"
        ) from exc
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_YAML}. Run make_pseudo_labels.py first.")

    print("Loading pretrained base model: yolov8n-seg.pt...")
    model = YOLO("yolov8n-seg.pt")

    print("Starting training...")
    # 20 epochs is enough for transfer learning on 12 bubble images with good convergence
    _ = model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=640,
        batch=4,
        task="segment",
        device=0,  # GPU 0
        project=str(REPO_ROOT / "build" / "yolo_train"),
        name="bubble_run",
        exist_ok=True,
        verbose=True,
    )

    best_weights = REPO_ROOT / "build" / "yolo_train" / "bubble_run" / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = REPO_ROOT / "build" / "yolo_train" / "bubble_run" / "weights" / "last.pt"

    if best_weights.exists():
        WEIGHTS_DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, WEIGHTS_DEST)
        print(f"Successfully trained and saved model weights to: {WEIGHTS_DEST}")
        print(f"Weights size: {WEIGHTS_DEST.stat().st_size / (1024*1024):.2f} MB")
    else:
        raise FileNotFoundError(f"Could not find trained weights at {best_weights}")


if __name__ == "__main__":
    main()
