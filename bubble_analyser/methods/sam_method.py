"""SAM 2.1 (Segment Anything) Integration Method for Bubble Analyser.

This module implements a detection method that uses Meta's SAM 2.1 foundation
model (Apache 2.0) with automatic mask generation to detect bubbles, instead
of traditional watershed segmentation or the BubMask Mask R-CNN model. It
follows the same interface as the other methods in the Bubble Analyser.

Classes:
    SamSegmentation: Zero-shot bubble detection using SAM 2.1.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy import typing as npt

from bubble_analyser.core.param_guide import MethodGuide, guide_tooltips
from bubble_analyser.processing.image_postprocess import overlay_labels_on_rgb

SAM_GUIDE: MethodGuide = {
    "summary": (
        "Meta's general-purpose segmentation model. Needs no bubble training; a GPU is recommended. "
        "It spreads a grid of points over the image and keeps the masks it is confident about."
    ),
    "params": {
        "weights_path": {
            "what": (
                "SAM 2.1 checkpoint. sam2.1_hiera_large.pt is the default and most accurate; the base_plus, small "
                "and tiny checkpoints (e.g. sam2.1_hiera_small.pt) are faster and lighter but a bit less accurate."
            ),
        },
        "points_per_side": {
            "what": (
                "The grid has points_per_side x points_per_side points; a bubble that no point lands on is missed. "
                "Point spacing = image width / points_per_side (4096 / 32 = 128 px)."
            ),
            "higher": "48 or 64 finds smaller bubbles, but the time grows roughly with the square.",
            "lower": "Faster, but small bubbles are missed.",
        },
        "pred_iou_thresh": {
            "what": "SAM's own quality score for each mask.",
            "higher": "Only crisp, confident masks.",
            "lower": (
                "0.7-0.8 keeps more masks, including faint or blurred bubbles, but also background patches and "
                "half-bubbles."
            ),
        },
        "stability_score_thresh": {
            "what": "How stable a mask's outline is.",
            "higher": "Only sharp-edged bubbles.",
            "lower": "0.85-0.9 keeps bubbles with soft, out-of-focus edges, plus more junk.",
        },
        "min_mask_region_area": {
            "what": (
                "In px of the resampled image (0 = off). Fills holes, such as the bright highlight in a bubble's "
                "centre, and removes specks smaller than this inside each mask, which helps ring-like masks pass "
                "the solidity filter."
            ),
            "tip": "Slower, and it rarely changes the bubble count. Typical values are 50-300.",
        },
        "resample": {
            "what": "SAM always looks at a 1024 x 1024 copy, so resample mainly sets how finely outlines are traced.",
            "higher": (
                "Sharper outlines, and it sometimes finds bubbles it missed (one test image: 3 masks at 0.4, all 5 "
                "at 1.0), but it needs much more GPU memory and time."
            ),
            "lower": "Less GPU memory.",
        },
    },
}

# Mapping from SAM 2.1 checkpoint filename to the model config bundled with
# the SAM-2 package. The checkpoint and the config must describe the same
# architecture, so the pair is resolved from the weights filename.
MODEL_CONFIGS = {
    "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
}


def masks_to_labels(
    masks: npt.NDArray[np.bool_],
    shape: tuple[int, ...],
) -> npt.NDArray[np.int32]:
    """Convert a stack of boolean instance masks to a watershed-style label image.

    Follows the same convention as the BubMask method: background is 0 and
    bubbles are labelled from 2 onwards.

    Args:
        masks (npt.NDArray[np.bool_]): Stack of masks of shape (H, W, N).
        shape (tuple[int, ...]): Shape of the output label image (H, W).

    Returns:
        npt.NDArray[np.int32]: Label image with one integer id per bubble.
    """
    height, width = int(shape[0]), int(shape[1])
    labels = np.zeros((height, width), dtype=np.int32)

    for i in range(masks.shape[2]):
        m = masks[:, :, i]
        # Resize mask to match the original image size if needed
        if m.shape != (height, width):
            m_resized = cast(
                npt.NDArray[np.uint8],
                cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST),
            )
            m_final = m_resized > 0
        else:
            m_final = m

        # Add to labels (i+2 because 1 is reserved for background)
        labels[m_final] = i + 2

    return labels


class SamSegmentation:
    def __init__(self, params: dict[str, float | int]) -> None:
        """Initialize the SAM 2.1 segmentation method.

        Args:
            params (Dict[str, float | int]): Dictionary containing parameters for the method.
                May include 'weights_path', 'points_per_side', 'pred_iou_thresh',
                'stability_score_thresh', 'min_mask_region_area' and 'resample'.
        """
        self.name = "SAM 2.1 (Segment Anything)"
        self.description = (
            "Zero-shot detection of bubbles using Meta's SAM 2.1 foundation model with "
            "automatic mask generation (Ravi et al., 2024, https://arxiv.org/abs/2408.00714)."
        )
        base_dir = str(Path(__file__).resolve().parents[2])
        self.weights_path: str = os.path.join(base_dir, "bubble_analyser/weights/sam2.1_hiera_large.pt")
        self.points_per_side: int = 32
        self.pred_iou_thresh: float = 0.88
        self.stability_score_thresh: float = 0.95
        self.min_mask_region_area: int = 0
        self.resample: float = 0.4
        self.alpha: float = 0.5

        # Lazily built model components. Heavy imports (torch, sam2) are
        # deferred to _ensure_model so that app startup stays fast and the
        # method can be instantiated on machines without torch installed.
        self.model: Any | None = None
        self.device: str | None = None
        self.detection_results: dict[str, object] = {}
        self.labels_watershed: npt.NDArray[np.int32]
        self.img_grey: npt.NDArray[np.uint8]
        self.img_rgb: npt.NDArray[np.uint8]
        self.if_bknd_img: bool
        self.bknd_img: npt.NDArray[np.uint8] | None

        # Update with provided parameters
        self.update_params(cast(dict[str, object], params))

    def get_needed_params(self) -> dict[str, float | int | str]:
        """Get the parameters required for this segmentation method.

        Returns:
            Dict[str, float | int | str]: Dictionary containing the required parameters and their current values.
        """
        return {
            "weights_path": self.weights_path,
            "points_per_side": self.points_per_side,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "min_mask_region_area": self.min_mask_region_area,
            "resample": self.resample,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for each parameter for use in GUI tooltips.

        Returns:
            Dict[str, str]: Dictionary mapping parameter names to their descriptions.
        """
        return guide_tooltips(SAM_GUIDE)

    def get_param_guide(self) -> MethodGuide:
        """Get the parameter guide shown in the GUI's parameter guide window.

        Returns:
            MethodGuide: What the method does and what each parameter changes.
        """
        return SAM_GUIDE

    def update_params(self, params: dict[str, object]) -> None:
        """Update the parameters for this segmentation method.

        Args:
            params (Dict[str, Any]): Dictionary containing parameters to update.
        """
        if "weights_path" in params:
            self.weights_path = str(params["weights_path"])
        if "points_per_side" in params:
            self.points_per_side = int(cast(float, params["points_per_side"]))
        if "pred_iou_thresh" in params:
            self.pred_iou_thresh = float(cast(float, params["pred_iou_thresh"]))
        if "stability_score_thresh" in params:
            self.stability_score_thresh = float(cast(float, params["stability_score_thresh"]))
        if "min_mask_region_area" in params:
            self.min_mask_region_area = int(cast(float, params["min_mask_region_area"]))
        if "resample" in params:
            self.resample = float(cast(float, params["resample"]))

    def _ensure_model(self) -> None:
        """Build the SAM 2.1 model once and cache it on the instance.

        Heavy dependencies (torch, sam2) are imported here so that module
        import stays cheap for GUI startup and unit tests.
        """
        if self.model is not None:
            return

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"SAM 2.1 checkpoint not found at: {self.weights_path}. "
                "Select the method in the GUI to trigger the download, or run "
                "bubble_analyser.weights.loader.get_weights_path(download_if_missing=True)."
            )

        model_cfg = MODEL_CONFIGS.get(os.path.basename(self.weights_path))
        if model_cfg is None:
            raise ValueError(
                f"Unrecognised SAM checkpoint filename: {self.weights_path}. "
                f"Expected one of: {', '.join(sorted(MODEL_CONFIGS))}."
            )

        try:
            import torch
            from sam2.build_sam import build_sam2
        except ImportError as e:
            raise RuntimeError(
                "SAM 2.1 requires PyTorch and the SAM-2 package. "
                "Install them with 'poetry install'."
            ) from e

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_sam2(model_cfg, self.weights_path, device=self.device)
        logging.info(f"SAM 2.1 model initialised on device: {self.device}")

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
        """Initialize the processing with input images and parameters.

        Args:
            params (Dict[str, float | int]): Dictionary containing parameters for the method.
            img_grey (npt.NDArray[np.uint8]): Grayscale input image.
            img_rgb (npt.NDArray[np.uint8]): RGB input image.
            if_bknd_img (bool): Flag indicating if background image is used.
            px2mm (float): Conversion factor from pixels to millimeters.
            bknd_img (npt.NDArray[np.uint8], optional): Background image if available.
            cnn_model (object, optional): Pre-initialised model. If it is a SAM
                model it is reused; otherwise it is ignored.
        """
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.if_bknd_img = if_bknd_img
        self.bknd_img = bknd_img
        self.px2mm = px2mm

        # Update parameters
        self.update_params(cast(dict[str, object], params))

        # Reuse a pre-built SAM model if one was handed over, otherwise
        # build lazily on first detection.
        if cnn_model is not None and hasattr(cnn_model, "image_encoder"):
            self.model = cnn_model

    def detect_bubbles(self) -> None:
        """Run SAM 2.1 automatic mask generation to detect bubbles."""
        self._ensure_model()

        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            mask_generator = SAM2AutomaticMaskGenerator(
                self.model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )

            logging.info("Running SAM 2.1 automatic mask generation...")
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        annotations = mask_generator.generate(self.img_rgb)
                else:
                    annotations = mask_generator.generate(self.img_rgb)

            # Sort by area (largest first) so smaller bubbles keep their
            # label pixels when masks marginally overlap.
            annotations = sorted(annotations, key=lambda ann: ann["area"], reverse=True)

            height, width = self.img_grey.shape[:2]
            if annotations:
                masks = np.stack([ann["segmentation"] for ann in annotations], axis=-1)
                scores = np.array([ann["predicted_iou"] for ann in annotations], dtype=np.float64)
            else:
                masks = np.zeros((height, width, 0), dtype=bool)
                scores = np.zeros((0,), dtype=np.float64)

            self.detection_results = {
                "masks": masks,
                "scores": scores,
                "bubble_count": len(annotations),
            }

            # Convert SAM masks to watershed format
            self._convert_sam_to_watershed()

            logging.info(f"SAM 2.1 detected {len(annotations)} bubbles")

        except Exception as e:
            logging.error(f"Error running SAM 2.1 detection: {e}")
            raise

    def _convert_sam_to_watershed(self) -> None:
        """Convert SAM detection results to watershed segmentation format."""
        if not self.detection_results or "masks" not in self.detection_results:
            # No bubbles detected, create empty labels
            self.labels_watershed = np.zeros(self.img_grey.shape, dtype=np.int32)
            return

        masks = cast(npt.NDArray[np.bool_], self.detection_results["masks"])
        self.labels_watershed = masks_to_labels(masks, self.img_grey.shape)

    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], None]:
        """Get the results image with overlaid bubble detection.

        Returns:
            tuple: (labels_on_img, labels_watershed, None) where labels_on_img is the
                  overlayed image with bubble masks and labels_watershed is just the masks.
        """
        start_time = time.time()
        self.detect_bubbles()
        end_time = time.time()
        logging.info(f"SAM 2.1 detection completed in {end_time - start_time:.2f} seconds")

        if not hasattr(self, "labels_watershed"):
            raise RuntimeError("SAM 2.1 detection not run yet. Call detect_bubbles first.")

        # Create transparent overlay using the standard project function
        labels_on_img = overlay_labels_on_rgb(self.img_rgb, self.labels_watershed, alpha=self.alpha)

        return labels_on_img.astype(np.uint8), self.labels_watershed, None

    def get_bubble_properties(self, pixel_to_mm: float = 1.0) -> list[dict[str, object]]:
        """Get detailed properties of detected bubbles.

        Args:
            pixel_to_mm (float): Conversion factor from pixels to millimeters.

        Returns:
            list[Dict]: List of dictionaries containing bubble properties.
        """
        masks = self.detection_results.get("masks")
        if not isinstance(masks, np.ndarray) or masks.size == 0:
            return []

        from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskDetector

        return BubMaskDetector.get_bubble_properties(cast(npt.NDArray[np.bool_], masks), pixel_to_mm)

    def get_detection_confidence(self) -> list[float]:
        """Get confidence scores for detected bubbles.

        Returns:
            list[float]: List of SAM predicted IoU scores for each detected bubble.
        """
        scores = self.detection_results.get("scores")
        if isinstance(scores, np.ndarray):
            return scores.tolist()  # type: ignore
        return []
