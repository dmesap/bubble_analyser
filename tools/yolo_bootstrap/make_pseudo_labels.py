"""Build a YOLO segmentation dataset by pseudo-labelling images with BubMask.

No public YOLO model trained on physical (gas-liquid) bubbles exists, so this script
bootstraps one: the BubMask Mask R-CNN (already shipped with Bubble Analyser) labels
the images, and the labels are written in Ultralytics YOLO segmentation format.

Usage:
    python tools/yolo_bootstrap/make_pseudo_labels.py --images example_imgs/sample_bubble_images \
        --out build/yolo_bootstrap_dataset --val 2

The resulting dataset is only as good as BubMask on these images. Review/correct it
(e.g. with the export feature of the GUI) before using it for anything serious.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskConfig, BubMaskDetector  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def mask_to_polygon(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    """Return the largest external contour of a binary mask as an (N, 2) float array."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    epsilon = 0.003 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    if len(contour) < 3:
        return None
    return contour.reshape(-1, 2).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="Folder with bubble images")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset folder")
    parser.add_argument(
        "--weights", type=Path, default=REPO_ROOT / "bubble_analyser" / "weights" / "mask_rcnn_bubble.h5"
    )
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--val", type=int, default=2, help="Number of images held out for validation")
    parser.add_argument("--max-width", type=int, default=2048, help="Downscale stored images to this width")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {args.images}")
    random.Random(args.seed).shuffle(images)
    val_set = set(images[: args.val])

    config = BubMaskConfig.for_high_quality(confidence_threshold=args.confidence)
    detector = BubMaskDetector(str(args.weights), config)

    total = 0
    for image_path in images:
        split = "val" if image_path in val_set else "train"
        bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip (unreadable): {image_path.name}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]

        result = detector.detect_bubbles(rgb, return_masks=True)
        masks = result.get("masks")
        scores = np.asarray(result.get("scores", []))
        lines: list[str] = []
        if masks is not None and np.size(masks):
            masks = np.asarray(masks)
            min_area = 0.00002 * height * width  # drop specks
            for index in range(masks.shape[-1]):
                if index < len(scores) and scores[index] < args.confidence:
                    continue
                polygon = mask_to_polygon(masks[..., index], min_area)
                if polygon is None:
                    continue
                normalised = polygon / np.array([width, height])
                normalised = np.clip(normalised, 0.0, 1.0)
                lines.append("0 " + " ".join(f"{v:.6f}" for v in normalised.ravel()))

        image_dir = args.out / "images" / split
        label_dir = args.out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        scale = min(1.0, args.max_width / width)
        stored = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        out_image = image_dir / f"{image_path.stem}.jpg"
        cv2.imencode(".jpg", stored, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(out_image))
        (label_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total += len(lines)
        print(f"{split:5s} {image_path.name}: {len(lines)} bubbles")

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: bubble\n",
        encoding="utf-8",
    )
    print(f"Done: {total} bubble polygons written to {args.out}")


if __name__ == "__main__":
    main()
