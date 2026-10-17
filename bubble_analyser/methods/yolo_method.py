"""YOLO segmentation integration method for Bubble Analyser.

This module adapts Ultralytics YOLO segmentation models to the same internal
label-mask format used by the existing Bubble Analyser segmentation methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy import typing as npt

from bubble_analyser.processing.image_postprocess import overlay_labels_on_rgb


def normalize_yolo_device(device: object) -> str:
    """Convert user-facing CPU/GPU labels to Ultralytics device values."""
    device_text = str(device).strip()
    device_lower = device_text.lower()
    if device_lower == "cpu":
        return "cpu"
    if device_lower == "gpu":
        return "0"
    return device_text


class YOLOSegmentation:
    """Segment bubbles with an Ultralytics YOLO segmentation model."""

    def __init__(self, params: dict[str, float | int | str]) -> None:
        self.name = "YOLO Segmentation"
        self.description = (
            "Detection of bubbles using an Ultralytics YOLO segmentation model. "
            "The predicted masks are converted to Bubble Analyser labels and then "
            "processed by the existing ellipse/polygon fitting workflow."
        )

        base_dir = Path(__file__).resolve().parents[2]
        self.weights_path = str(base_dir / "bubble_analyser" / "weights" / "yolo_bubble_seg.pt")
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.50
        self.resample = 1.0
        self.max_bubbles = 300
        self.device = "CPU"
        self.alpha = 0.5

        self.model: Any | None = None
        self._loaded_weights: Path | None = None
        self.img_grey: npt.NDArray[np.uint8]
        self.img_rgb: npt.NDArray[np.uint8]
        self.labels_watershed: npt.NDArray[np.int32]

        self.update_params(cast(dict[str, object], params))

    def get_needed_params(self) -> dict[str, float | int | str]:
        """Return parameters exposed in the Bubble Analyser GUI."""
        return {
            "weights_path": self.weights_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "resample": self.resample,
            "max_bubbles": self.max_bubbles,
            "device": self.device,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Return GUI tooltip descriptions for YOLO parameters."""
        return {
            "weights_path": "Absolute path to a YOLO segmentation model (.pt).",
            "confidence_threshold": "Minimum confidence required to accept a YOLO mask.",
            "iou_threshold": "YOLO non-maximum suppression IoU threshold.",
            "resample": "Display/processing scale for the image before segmentation.",
            "max_bubbles": "Maximum number of YOLO masks to keep in one image.",
            "device": (
                "Compute device for YOLO inference. CPU is always compatible; GPU uses NVIDIA CUDA when available."
            ),
        }

    def update_params(self, params: dict[str, object]) -> None:
        """Update method parameters from the GUI table."""
        if "weights_path" in params:
            self.weights_path = str(params["weights_path"])
        if "confidence_threshold" in params:
            self.confidence_threshold = float(cast(float, params["confidence_threshold"]))
        if "iou_threshold" in params:
            self.iou_threshold = float(cast(float, params["iou_threshold"]))
        if "resample" in params:
            self.resample = float(cast(float, params["resample"]))
        if "max_bubbles" in params:
            self.max_bubbles = int(cast(float, params["max_bubbles"]))
        if "device" in params:
            self.device = str(params["device"])

    def initialize_processing(
        self,
        params: dict[str, Any],
        img_grey: npt.NDArray[np.uint8],
        img_rgb: npt.NDArray[np.uint8],
        if_bknd_img: bool,
        px2mm: float,
        bknd_img: npt.NDArray[np.uint8] | None = None,
        cnn_model: Any | None = None,
    ) -> None:
        """Initialize images, parameters, and the YOLO model."""
        del if_bknd_img, px2mm, bknd_img
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.update_params(cast(dict[str, object], params))

        if cnn_model is not None and hasattr(cnn_model, "predict"):
            self.model = cnn_model
            self._loaded_weights = None
            return

        weights = Path(self.weights_path)
        # Reuse the model already in memory (batch processing calls this once per image).
        if self.model is not None and self._loaded_weights == weights:
            return

        if not weights.is_file():
            raise FileNotFoundError(
                f"YOLO weights file not found. Set 'weights_path' to a valid .pt segmentation model: {weights}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "Ultralytics is not installed. Install it with 'pip install ultralytics' "
                "or choose another segmentation method."
            ) from error

        self.model = YOLO(str(weights))
        self._loaded_weights = weights

    def _predict_masks(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Run YOLO and return masks plus confidence scores."""
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized. Call initialize_processing first.")

        results = self.model.predict(
            source=self.img_rgb,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=normalize_yolo_device(self.device),
            verbose=False,
        )
        if not results:
            return (
                np.zeros((0, *self.img_grey.shape), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        result = results[0]
        if getattr(result, "masks", None) is None or result.masks is None:
            return (
                np.zeros((0, *self.img_grey.shape), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        masks_data = result.masks.data
        masks_np = masks_data.detach().cpu().numpy().astype(np.float32)
        scores_np = np.ones((masks_np.shape[0],), dtype=np.float32)
        if getattr(result, "boxes", None) is not None and result.boxes is not None:
            conf = getattr(result.boxes, "conf", None)
            if conf is not None:
                scores_np = conf.detach().cpu().numpy().astype(np.float32)
        return masks_np, scores_np

    def _convert_masks_to_labels(
        self,
        masks: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int32]:
        """Convert YOLO masks to Bubble Analyser integer labels."""
        height, width = self.img_grey.shape
        labels = np.zeros((height, width), dtype=np.int32)
        if masks.size == 0:
            return labels

        keep = np.where(scores >= self.confidence_threshold)[0]
        if keep.size == 0:
            return labels
        keep = keep[np.argsort(scores[keep])[::-1]][: self.max_bubbles]

        next_label = 2
        for index in keep:
            mask = masks[index]
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            binary = mask > 0.5
            if not np.any(binary):
                continue
            labels[binary] = next_label
            next_label += 1
        return labels

    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], None]:
        """Run YOLO segmentation and return Bubble Analyser-compatible outputs."""
        logging.info("Running YOLO segmentation...")
        masks, scores = self._predict_masks()
        self.labels_watershed = self._convert_masks_to_labels(masks, scores)
        labels_on_img = overlay_labels_on_rgb(self.img_rgb, self.labels_watershed, alpha=self.alpha)
        logging.info("YOLO detected %s bubble masks", int(max(0, self.labels_watershed.max() - 1)))
        return labels_on_img.astype(np.uint8), self.labels_watershed, None
