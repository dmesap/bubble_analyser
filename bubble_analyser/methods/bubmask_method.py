"""BubMask Integration Method for Bubble Analyser.

This module implements a watershed method that uses BubMask (Mask R-CNN) for bubble detection
instead of traditional watershed segmentation. It follows the same interface as other
watershed methods in the Bubble Analyser.

Classes:
    BubMaskWatershed: Watershed method using BubMask for detection.
"""

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
from numpy import typing as npt

if TYPE_CHECKING:
    from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskDetector

# from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskConfig, BubMaskDetector
from bubble_analyser.processing.image_postprocess import overlay_labels_on_rgb


class BubMaskWatershed:
    def __init__(self, params: dict[str, float | int]) -> None:
        """Initialize the BubMask watershed segmentation method.

        Args:
            params (Dict[str, float | int]): Dictionary containing parameters for the method.
                Must include 'weights_path', 'confidence_threshold', 'resample',
                'max_bubbles', and 'nms_threshold'.
        """
        self.name = "BubMask (Deep Learning)"
        self.description = (
            "Detection of bubbles using Mask R-CNN model from MULTIPHASE FLOW & FLOW VISUALIZATION LAB, "
            "https://doi.org/10.1038/s41598-021-88334-0 (Kim & Park, 2021)."
        )
        base_dir = str(Path(__file__).resolve().parents[2])
        self.weights_path: str = os.path.join(base_dir, "bubble_analyser/weights/mask_rcnn_bubble.h5")
        self.confidence_threshold: float = 0.9
        self.resample: float = 1.0
        self.target_width: int = 1000
        self.image_min_dim: int = 512
        self.image_max_dim: int = 1024
        self.alpha: float = 0.5
        self.max_bubbles: int = 500
        self.nms_threshold: float = 0.3

        self.detector: BubMaskDetector | None = None
        self.detection_results: dict[str, object] = {}
        self.labels_watershed: npt.NDArray[np.int32]
        self.img_grey: npt.NDArray[np.uint8]
        self.img_rgb: npt.NDArray[np.uint8]
        self.if_bknd_img: bool
        self.bknd_img: npt.NDArray[np.uint8] | None

        # Update with provided parameters
        self.update_params(cast(dict[str, object], params))

    def get_needed_params(self) -> dict[str, float | int | str]:
        """Get the parameters required for this watershed method.

        Returns:
            Dict[str, float | int | str]: Dictionary containing the required parameters and their current values.
        """
        return {
            "weights_path": self.weights_path,
            "confidence_threshold": self.confidence_threshold,
            "resample": self.resample,
            "image_min_dim": self.image_min_dim,
            "image_max_dim": self.image_max_dim,
            "max_bubbles": self.max_bubbles,
            "nms_threshold": self.nms_threshold,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for each parameter for use in GUI tooltips.

        Returns:
            Dict[str, str]: Dictionary mapping parameter names to their descriptions.
        """
        return {
            "weights_path": "Absolute path to the trained Mask R-CNN model weights (.h5 file).",
            "confidence_threshold": "Minimum probability (0.0 to 1.0) required for a detection to be accepted as a bubble.",  # noqa: E501
            "resample": "Resampling factor to scale the image before processing. Lower values increase speed but reduce detail.",  # noqa: E501
            "image_min_dim": "Minimum internal image dimension used by Mask R-CNN. Higher values improve small-bubble detection but increase runtime and memory use.",  # noqa: E501
            "image_max_dim": "Maximum internal image dimension used by Mask R-CNN. Higher values improve small-bubble detection but increase runtime and memory use.",  # noqa: E501
            "max_bubbles": "Maximum number of bubbles the model is allowed to detect in a single image.",
            "nms_threshold": "Non-Maximum Suppression threshold. Lower values allow less overlap between detected bubbles.",  # noqa: E501
        }

    def update_params(self, params: dict[str, object]) -> None:
        """Update the parameters for this watershed method.

        Args:
            params (Dict[str, Any]): Dictionary containing parameters to update.
        """
        # Update BubMask-specific parameters
        if "weights_path" in params:
            self.weights_path = str(params["weights_path"])
        if "confidence_threshold" in params:
            self.confidence_threshold = float(cast(float, params["confidence_threshold"]))
        if "resample" in params:
            self.resample = float(cast(float, params["resample"]))
        if "image_min_dim" in params:
            self.image_min_dim = int(cast(float, params["image_min_dim"]))
        if "image_max_dim" in params:
            self.image_max_dim = int(cast(float, params["image_max_dim"]))
        if "max_bubbles" in params:
            self.max_bubbles = int(cast(float, params["max_bubbles"]))
        if "nms_threshold" in params:
            self.nms_threshold = float(cast(float, params["nms_threshold"]))

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
            cnn_model (object, optional): Pre-initialized CNN model.
        """
        from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskConfig, BubMaskDetector

        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.if_bknd_img = if_bknd_img
        self.bknd_img = bknd_img
        self.px2mm = px2mm

        # Update parameters
        self.update_params(cast(dict[str, object], params))

        if cnn_model is not None:
            self.detector = cast(BubMaskDetector, cnn_model)
            # Ensure the pre-initialized model's config matches the current parameters
            self.detector.update_config(
                confidence_threshold=self.confidence_threshold,
                max_bubbles=self.max_bubbles,
                nms_threshold=self.nms_threshold,
            )

        # Initialize BubMask detector if not already done
        if self.detector is None and self.weights_path:
            try:
                config = BubMaskConfig(
                    confidence_threshold=self.confidence_threshold,
                    image_min_dim=self.image_min_dim,
                    image_max_dim=self.image_max_dim,
                    max_bubbles=self.max_bubbles,
                    nms_threshold=self.nms_threshold,
                )
                self.detector = BubMaskDetector(self.weights_path, config)
                logging.info("BubMask detector initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize BubMask detector: {e}")
                raise

    def batch_detect_bubbles(self, images: list[npt.NDArray[np.uint8]]) -> list[dict[str, object]]:
        """Run BubMask detection on a batch of images.

        Args:
            images: List of RGB image arrays.

        Returns:
            List of detection results.
        """
        if self.detector is None:
            raise RuntimeError("BubMask detector not initialized.")

        if not images:
            return []

        try:
            results: list[dict[str, Any]] = []
            if self.detector.model is None:
                raise RuntimeError("Model not loaded")

            batch_size = self.detector.model.config.BATCH_SIZE

            # Process in chunks of BATCH_SIZE
            for i in range(0, len(images), batch_size):
                chunk = images[i : i + batch_size]

                # Mask R-CNN strictly expects exactly BATCH_SIZE images
                # If the chunk is smaller than BATCH_SIZE, we must pad it
                pad_size = batch_size - len(chunk)
                if pad_size > 0:
                    # Pad with copies of the last image (simplest way to ensure shape match)
                    chunk.extend([chunk[-1]] * pad_size)

                # Convert to float32 as expected by model.detect type hints
                float_chunk = [img.astype(np.float32) for img in chunk]
                batch_results = self.detector.model.detect(float_chunk, verbose=0)

                # Only iterate over the actual requested images, ignoring the padded ones
                for j in range(batch_size - pad_size):
                    result = batch_results[j]
                    keep_indices = result["scores"] >= self.detector.config.confidence_threshold
                    filtered_result: dict[str, object] = {
                        "rois": result["rois"][keep_indices],
                        "class_ids": result["class_ids"][keep_indices],
                        "scores": result["scores"][keep_indices],
                        "bubble_count": int(np.sum(keep_indices)),
                        "masks": result["masks"][:, :, keep_indices],
                        "image_shape": images[i + j].shape,
                    }
                    results.append(filtered_result)

            return results

        except Exception as e:
            logging.error(f"Error running batch BubMask detection: {e}")
            raise

    def get_batch_results(
        self, images: list[npt.NDArray[np.uint8]], images_grey: list[npt.NDArray[np.uint8]]
    ) -> list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], None]]:
        """Get results for a batch of images.

        Args:
            images: List of RGB image arrays.
            images_grey: List of grayscale image arrays corresponding to the RGB images.

        Returns:
            List of tuples: (labels_on_img, labels_watershed, None)
        """
        start_time = time.time()
        logging.info(f"Running BubMask detection on batch of {len(images)} images...")

        batch_detection_results = self.batch_detect_bubbles(images)

        end_time = time.time()
        logging.info(f"BubMask batch detection completed in {end_time - start_time:.2f} seconds")

        final_results = []
        for i, detection_results in enumerate(batch_detection_results):
            self.detection_results = detection_results
            self.img_grey = images_grey[i]
            self.img_rgb = images[i]

            self._convert_bubmask_to_watershed()
            labels_on_img = overlay_labels_on_rgb(self.img_rgb, self.labels_watershed, alpha=self.alpha)

            final_results.append((labels_on_img.astype(np.uint8), self.labels_watershed.copy(), None))

        return final_results

    def detect_bubbles(self) -> None:
        """Run the BubMask detection process.

        This method replaces traditional watershed segmentation with BubMask detection.
        """
        if self.detector is None:
            raise RuntimeError("BubMask detector not initialized. Call initialize_processing first.")

        try:
            # Run BubMask detection directly on the image array to avoid disk I/O overhead
            self.detection_results = self.detector.detect_bubbles(self.img_rgb, return_masks=True, return_splash=False)

            # Convert BubMask results to watershed format
            self._convert_bubmask_to_watershed()

            logging.info(f"BubMask detected {self.detection_results.get('bubble_count', 0)} bubbles")

        except Exception as e:
            logging.error(f"Error running BubMask detection: {e}")
            raise

    def _convert_bubmask_to_watershed(self) -> None:
        """Convert BubMask detection results to watershed segmentation format."""
        if not self.detection_results or "masks" not in self.detection_results:
            # No bubbles detected, create empty labels
            self.labels_watershed = np.zeros(self.img_grey.shape, dtype=np.int32)
            return

        masks = cast(npt.NDArray[np.bool_], self.detection_results["masks"])
        height, width = self.img_grey.shape

        # Create labels array
        # Label 1 is background, bubbles start from 2
        self.labels_watershed = np.zeros((height, width), dtype=np.int32)

        # Convert each mask to a labeled region
        for i in range(masks.shape[2]):
            m = masks[:, :, i]
            # Resize mask to match original image size if needed
            if m.shape != (height, width):
                m_resized = cast(
                    npt.NDArray[np.uint8],
                    cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST),
                )
                m_final = m_resized > 0
            else:
                m_final = m

            # Add to labels (i+2 because 1 is background)
            self.labels_watershed[m_final] = i + 2

    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], None]:
        """Get the results image with overlaid bubble detection.

        Returns:
            tuple: (labels_on_img, labels_watershed, None) where labels_on_img is the
                  overlayed image with bubble masks and labels_watershed is just the masks.
        """
        start_time = time.time()
        logging.info("Running BubMask detection...")
        self.detect_bubbles()
        end_time = time.time()
        logging.info(f"BubMask detection completed in {end_time - start_time:.2f} seconds")

        if not hasattr(self, "labels_watershed"):
            raise RuntimeError("BubMask detection not run yet. Call detect_bubbles first.")

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
        if not self.detection_results or "masks" not in self.detection_results:
            return []

        if self.detector is None:
            return []

        masks = self.detection_results["masks"]
        if not isinstance(masks, np.ndarray):
            return []

        return self.detector.get_bubble_properties(cast(npt.NDArray[np.bool_], masks), pixel_to_mm)

    def get_detection_confidence(self) -> list[float]:
        """Get confidence scores for detected bubbles.

        Returns:
            list[float]: List of confidence scores for each detected bubble.
        """
        if not self.detection_results or "scores" not in self.detection_results:
            return []

        scores = self.detection_results["scores"]
        if isinstance(scores, np.ndarray):
            return scores.tolist()  # type: ignore
        return []

    def get_bounding_boxes(self) -> list[tuple[float, ...]]:
        """Get bounding boxes for detected bubbles.

        Returns:
            list[tuple]: List of bounding boxes as (y1, x1, y2, x2) tuples.
        """
        if not self.detection_results or "rois" not in self.detection_results:
            return []

        rois = self.detection_results["rois"]
        if isinstance(rois, np.ndarray):
            return [tuple(roi) for roi in rois]
        return []
