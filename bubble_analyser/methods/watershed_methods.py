"""Watershed Segmentation Methods for Bubble Analysis.

This module provides implementations of watershed segmentation algorithms for bubble analysis.
It includes two main approaches: Normal Watershed and Iterative Watershed. These methods
are designed to segment bubbles in images with different characteristics and requirements.

Classes:
    NormalWatershed: Implements the standard watershed algorithm with a single threshold.
    IterativeWatershed: Implements an advanced watershed algorithm that iteratively applies
                       thresholds to detect objects at different intensity levels.

Both classes inherit from the WatershedSegmentation parent class
"""

import logging
from typing import Any, cast

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy import typing as npt

from bubble_analyser.core.param_guide import (
    CONNECTIVITY_HELP,
    ELEMENT_SIZE_HELP,
    RESAMPLE_HELP,
    MethodGuide,
    guide_tooltips,
)
from bubble_analyser.processing.threshold_methods import ThresholdMethods
from bubble_analyser.processing.watershed_parent_class import WatershedSegmentation

EDGE_NOTE = "Bubbles touching the image edge are ignored."

DEFAULT_GUIDE: MethodGuide = {
    "summary": (
        "Classic watershed with three rounds of seeds (Mesa et al., 2022). Fast and runs on the CPU. "
        "Best for separate, high-contrast bubbles on a plain background. " + EDGE_NOTE
    ),
    "how_it_works": (
        "The image is split into bubbles and background automatically (or against the background image set in "
        "calibration), cleaned up, and turned into a distance map that peaks at each bubble's centre, higher for "
        "bigger bubbles. Seeds are taken in three rounds at high, mid and low fractions of the tallest peak; each "
        "seed becomes one bubble."
    ),
    "params": {
        "resample": RESAMPLE_HELP,
        "high_thresh": {
            "what": "First round of seeds: only the centres of the largest bubbles.",
            "higher": "Fewer bubbles get a seed in the first round; they are picked up in the later rounds.",
            "lower": "Two touching large bubbles can share one seed and merge.",
            "tip": "Usually has little effect. The app requires high_thresh > mid_thresh > low_thresh.",
        },
        "mid_thresh": {
            "what": "Second round of seeds, for medium bubbles.",
            "higher": "Fewer seeds and more merged bubbles.",
            "lower": (
                "More bubbles get their own seed, so touching bubbles are separated better. Too low and elongated "
                "bubbles get split in two."
            ),
        },
        "low_thresh": {
            "what": "Last round of seeds, which catches small bubbles.",
            "higher": "Less merging, but small or faint bubbles may be missed.",
            "lower": "Even tiny blobs get seeds, but large leftover seeds can join touching bubbles.",
        },
        "element_size": ELEMENT_SIZE_HELP,
        "connectivity": CONNECTIVITY_HELP,
    },
}

ITERATIVE_GUIDE: MethodGuide = {
    "summary": (
        "Like Default, but the seed level is lowered step by step from max_thresh to min_thresh instead of in "
        "three rounds. Better for mixed bubble sizes and clusters, but slower. " + EDGE_NOTE
    ),
    "params": {
        "resample": RESAMPLE_HELP,
        "element_size": ELEMENT_SIZE_HELP,
        "connectivity": CONNECTIVITY_HELP,
        "max_thresh": {
            "what": "Highest seed level, like high_thresh in Default.",
            "tip": "Usually has little effect. Must be greater than min_thresh.",
        },
        "min_thresh": {
            "what": "Lowest seed level: a bubble only gets a seed if its peak reaches it.",
            "higher": "Fewer fragments and faster, but small bubbles are missed.",
            "lower": "Smaller bubbles are found, but you get more small fragments.",
        },
        "step_size": {
            "what": "Gap between seed levels.",
            "higher": "Faster with fewer fragments, closer to how Default behaves.",
            "lower": "Finer separation, but many more fragments and much slower.",
        },
    },
}


class TestWatershed(WatershedSegmentation):
    """Standard watershed segmentation implementation.

    This class implements the standard watershed algorithm for image segmentation.
    It uses a single threshold value to separate foreground and background regions,
    followed by watershed segmentation to identify individual objects.

    Attributes:
        name (str): Name identifier for this watershed method.
        img_grey_dt_thresh (npt.NDArray[np.uint8]): Thresholded distance transform image.
        sure_fg (npt.NDArray[np.uint8]): Sure foreground regions.
        sure_bg (npt.NDArray[np.uint8]): Sure background regions.
        unknown (MatLike): Unknown regions (neither sure foreground nor background).
        threshold_value (float): Threshold value for distance transform.
        target_width (int) : Target width for image resizing.
        if_bknd_img (bool): Flag indicating if background image is used.
    """

    def __init__(self, params: dict[str, float | int]) -> None:
        """Initialize the NormalWatershed segmentation method.

        Args:
            params (dict[str, float | int]): Dictionary containing parameters for the watershed method.
                Must include 'target_width', 'element_size', 'connectivity', and 'threshold_value'.
        """
        self.name = "NEW method"
        self.description = "A new watershed method still in developing and testing (by Yiyang)."
        self.img_grey_dt_thresh: MatLike
        self.grad_img_rgb: MatLike
        self.sure_fg: npt.NDArray[np.uint8]
        self.sure_bg: npt.NDArray[np.uint8]
        self.unknown: MatLike
        self.threshold_value: float
        self.resample: float
        # self.target_width: int
        self.if_bknd_img: bool = False
        self.if_gaussianblur: bool = False
        self.ksize: int = 3
        self.update_params(params)

    def get_needed_params(self) -> dict[str, float | int]:
        """Get the parameters required for this watershed method.

        Returns:
            dict[str, float | int]: Dictionary containing the required parameters and their current values.
        """
        return {
            # "target_width": self.target_width,
            "resample": self.resample,
            "threshold_value": self.threshold_value,
            "if_gaussianblur": self.if_gaussianblur,
            "ksize": self.ksize,
            "max_thresh": getattr(self, "max_thresh", 0.9),
            "min_thresh": getattr(self, "min_thresh", 0.1),
            "step_size": getattr(self, "step_size", 0.1),
            "element_size": self.element_size,
            "connectivity": self.connectivity,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for each parameter for use in GUI tooltips.

        Returns:
            dict[str, str]: Dictionary mapping parameter names to their descriptions.
        """
        return {
            "resample": "Resampling factor to scale the image before processing. Lower values increase speed but reduce detail.",  # noqa: E501  # noqa: E501
            "threshold_value": "Static threshold value used to binarize the image (0.0 to 1.0).",
            "if_gaussianblur": "Whether to apply a Gaussian blur to the image to reduce high-frequency noise before segmentation.",  # noqa: E501
            "ksize": "Size of the Sobel edge filter (1, 3, 5 or 7). Larger values give smoother, thicker edges and less noise.",  # noqa: E501
            "max_thresh": "Upper limit for iterative thresholding.",
            "min_thresh": "Lower limit for iterative thresholding.",
            "step_size": "Increment size for the iterative thresholding process.",
            "element_size": "Size of the structuring element for morphological operations.",
            "connectivity": "Pixel neighborhood for region growing (4 or 8).",
        }

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
            params (dict[str, Any]): Dictionary containing parameters for the watershed method.
            img_grey (npt.NDArray[np.uint8]): Grayscale input image.
            img_rgb (npt.NDArray[np.uint8]): RGB input image.
            if_bknd_img (bool): Flag indicating if background image is used.
            px2mm (float): Conversion factor from pixels to millimeters.
            bknd_img (npt.NDArray[np.uint8], optional): Background image if available. Defaults to None.
            cnn_model (object, optional): Pre-initialized CNN model for deep learning methods. Defaults to None.
        """
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.bknd_img = bknd_img
        self.if_bknd_img = if_bknd_img
        self.cnn_model = cnn_model
        self.px2mm = px2mm
        self.update_params(params)
        super().__init__(
            img_grey,
            img_rgb,
            if_bknd_img=if_bknd_img,
            bknd_img=bknd_img,
            element_size=self.element_size,
            connectivity=self.connectivity,
        )

    def update_params(self, params: dict[str, float | int]) -> None:
        """Update the parameters for the watershed method.

        Args:
            params (dict[str, float | int]): Dictionary containing parameters to update.
                Must include 'target_width', 'element_size', 'connectivity', and 'threshold_value'.
        """
        # self.target_width = cast(int, params["target_width"])
        self.resample = cast(float, params.get("resample", 1.0))
        self.max_thresh = cast(float, params.get("max_thresh", 0.9))
        self.min_thresh = cast(float, params.get("min_thresh", 0.1))
        self.step_size = cast(float, params.get("step_size", 0.1))
        self.element_size = cast(int, params.get("element_size", 5))
        self.connectivity = cast(int, params.get("connectivity", 4))
        self.threshold_value = cast(float, params.get("threshold_value", 0.5))
        self.ksize = cast(int, params.get("ksize", 3))

        if params.get("if_gaussianblur") == "True":
            self.if_gaussianblur = True
            logging.info("Gausssian Blur activated")
        else:
            self.if_gaussianblur = False
            logging.info("Gausssian Blur deactivated")

    def __get_sure_fg_bg(
        self, target_image: npt.NDArray[np.uint8], dt_thresh_image: MatLike
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], MatLike]:
        """Determine sure foreground and background regions.

        Creates masks for sure foreground (from thresholded distance transform),
        sure background (from dilated original image), and unknown regions (the difference
        between sure background and sure foreground).
        """
        sure_fg_initial = dt_thresh_image.copy()

        sure_bg = np.array(
            cv2.dilate(target_image, np.ones((3, 3), np.uint8), iterations=1),
            dtype=np.uint8,
        )  # type: ignore
        sure_fg = np.array(sure_fg_initial, dtype=np.uint8)  # type: ignore
        unknown = cv2.subtract(sure_bg, sure_fg)  # type: ignore

        return sure_fg, sure_bg, unknown

    def _dilate_mask(self, mask: MatLike) -> MatLike:
        """Dilate the mask to enhance object boundaries.

        This method applies a morphological dilation operation to the input mask.
        The dilation operation enlarges the foreground regions, which helps in
        better defining the boundaries of objects.
        """
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)  # type: ignore
        return dilated_mask

    def __iterative_threshold(self, image: MatLike) -> tuple[MatLike, int]:
        """Apply iterative thresholding to detect objects at different intensity levels.

        This method iteratively applies decreasing thresholds to the distance transform image,
        detecting objects at each threshold level. It accumulates non-overlapping objects into
        a final mask, which is then used for watershed segmentation. This approach is effective
        for detecting objects with varying intensities or sizes that might be missed by a single
        threshold approach.

        The method starts at max_thresh and decreases by step_size until reaching min_thresh,
        keeping track of unique objects detected along the way.
        """
        logging.basicConfig(level=logging.INFO)

        image_uint8 = image.astype(np.uint8) if isinstance(image, np.ndarray) else image
        # Initialize the final mask to accumulate all detected objects
        output_mask = np.zeros_like(image_uint8, dtype=np.uint8)

        # Set the initial threshold
        current_thresh = self.max_thresh
        no_overlap_count = 0  # Reset counter

        while current_thresh >= self.min_thresh:
            # Apply binary thresholding
            _, thresholded = cv2.threshold(image_uint8, int(current_thresh * image_uint8.max()), 255, cv2.THRESH_BINARY)

            # Label the thresholded image
            num_labels, labels = cv2.connectedComponents(thresholded, connectivity=self.connectivity)
            logging.info(f"Threshold {current_thresh:.2f}: {num_labels} components found.")

            # Detect new objects by comparing with the final mask
            for label in range(1, num_labels):  # Skip label 0 (background)
                # Create a mask for the current label
                component_mask = (labels == label).astype(np.uint8) * 255

                # Check if the object is already in the final mask
                overlap = cv2.bitwise_and(output_mask, component_mask)

                if not np.any(overlap):  # If no overlap, it's a new object
                    no_overlap_count += 1
                    output_mask = cast(npt.NDArray[np.uint8], cv2.bitwise_or(output_mask, component_mask))

            # Decrease the threshold for the next iteration
            current_thresh -= self.step_size

        final_label_count, _ = cv2.connectedComponents(output_mask)
        logging.info(f"Total unique labels in output_mask_for_labels: {final_label_count}")
        logging.info(f"Total number of no overlap occurrences: {no_overlap_count}")
        return output_mask, final_label_count

    def _gradient_topography(self, img: MatLike) -> None:
        # Ensure the image is in uint8 format for cv2.Sobel compatibility
        g_final: npt.NDArray[np.uint8]
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                g_final = img.astype(np.uint8)
            else:
                g_final = cast(npt.NDArray[np.uint8], img)
        else:
            g_final = cast(npt.NDArray[np.uint8], img)

        if self.if_gaussianblur:
            g_blurred = cv2.GaussianBlur(g_final, (0, 0), 1.2)
            g_final = g_blurred.astype(np.uint8)

        # 2) Gradient magnitude = "topography"
        gx_f = cv2.Sobel(g_final, cv2.CV_32F, 1, 0, ksize=self.ksize)
        gy_f = cv2.Sobel(g_final, cv2.CV_32F, 0, 1, ksize=self.ksize)
        grad_f = cv2.magnitude(gx_f, gy_f)

        # normalize to 8-bit for display + watershed input
        grad_u8 = cv2.normalize(grad_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore
        grad_bgr = cv2.cvtColor(grad_u8, cv2.COLOR_GRAY2BGR)
        self.img_grey = grad_u8
        self.grad_img_rgb = grad_bgr

    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], npt.NDArray[np.uint8]]:
        """Execute the complete watershed segmentation process and return results."""
        self._gradient_topography(self.img_grey)
        threshold_method = ThresholdMethods()
        self.img_grey_thresholded = threshold_method.normal_threshold(self.img_grey, self.threshold_value)
        self.img_grey_morph, self.img_grey_morph_eroded = self._morph_process(~self.img_grey_thresholded)
        self.img_grey_dt = self._dist_transform(self.img_grey_morph)
        self.output_mask_for_labels, self.final_label_count = self.__iterative_threshold(self.img_grey_dt)

        self.dilated_mask = self._dilate_mask(self.output_mask_for_labels)
        self.labels = self._initialize_labels(self.dilated_mask)
        self.labels_watershed = self._watershed_segmentation(self.grad_img_rgb, self.labels)

        self.labels_on_img = self._overlay_labels_on_rgb(
            self.img_rgb, cast(npt.NDArray[np.int32], self.labels_watershed)
        )

        return (
            cast(npt.NDArray[np.uint8], self.labels_on_img),
            cast(npt.NDArray[np.int32], self.labels_watershed),
            cast(npt.NDArray[np.uint8], self.img_grey_morph_eroded),
        )


class IterativeWatershed(WatershedSegmentation):
    """Iterative watershed segmentation implementation."""

    def __init__(self, params: dict[str, float | int]) -> None:
        self.name = "Iterative Watershed"
        self.description = (
            "An advanced watershed method that iteratively applies thresholds "
            "to detect objects at different intensity levels (by Yiyang)."
        )
        self.max_thresh: float
        self.min_thresh: float
        self.step_size: float
        self.resample: float
        self.update_params(params)
        self.output_mask_for_labels: MatLike
        self.no_overlap_count: int = 0
        self.final_label_count: int = 0

    def get_needed_params(self) -> dict[str, float | int]:
        return {
            "resample": self.resample,
            "element_size": self.element_size,
            "connectivity": self.connectivity,
            "max_thresh": self.max_thresh,
            "min_thresh": self.min_thresh,
            "step_size": self.step_size,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for each parameter for use in GUI tooltips.

        Returns:
            dict[str, str]: Dictionary mapping parameter names to their descriptions.
        """
        return guide_tooltips(ITERATIVE_GUIDE)

    def get_param_guide(self) -> MethodGuide:
        """Get the parameter guide shown in the GUI's parameter guide window.

        Returns:
            MethodGuide: What the method does and what each parameter changes.
        """
        return ITERATIVE_GUIDE

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
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.bknd_img = bknd_img
        self.if_bknd_img = if_bknd_img
        self.update_params(params)
        super().__init__(
            img_grey,
            img_rgb,
            if_bknd_img=if_bknd_img,
            bknd_img=bknd_img,
            element_size=self.element_size,
            connectivity=self.connectivity,
        )

    def update_params(self, params: dict[str, float | int]) -> None:
        self.resample = cast(float, params.get("resample", 1.0))
        self.element_size = cast(int, params.get("element_size", 5))
        self.connectivity = cast(int, params.get("connectivity", 4))
        self.max_thresh = cast(float, params.get("max_thresh", 0.9))
        self.min_thresh = cast(float, params.get("min_thresh", 0.1))
        self.step_size = cast(float, params.get("step_size", 0.1))

    def _dilate_mask(self, mask: MatLike) -> MatLike:
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)  # type: ignore
        return dilated_mask

    def __iterative_threshold(self, image: MatLike) -> tuple[MatLike, int]:
        logging.basicConfig(level=logging.INFO)
        image_uint8 = image.astype(np.uint8) if isinstance(image, np.ndarray) else image
        output_mask = np.zeros_like(image_uint8, dtype=np.uint8)
        current_thresh = self.max_thresh
        no_overlap_count = 0

        while current_thresh >= self.min_thresh:
            _, thresholded = cv2.threshold(image_uint8, int(current_thresh * image_uint8.max()), 255, cv2.THRESH_BINARY)
            num_labels, labels = cv2.connectedComponents(thresholded, connectivity=self.connectivity)
            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8) * 255
                overlap = cv2.bitwise_and(output_mask, component_mask)
                if not np.any(overlap):
                    no_overlap_count += 1
                    output_mask = cast(npt.NDArray[np.uint8], cv2.bitwise_or(output_mask, component_mask))
            current_thresh -= self.step_size

        final_label_count, _ = cv2.connectedComponents(output_mask)
        return output_mask, final_label_count

    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], npt.NDArray[np.uint8]]:
        self.img_grey_thresholded = self._threshold(self.img_grey)
        self.img_grey_morph, self.img_grey_morph_eroded = self._morph_process(self.img_grey_thresholded)
        self.img_grey_dt = self._dist_transform(self.img_grey_morph)
        self.output_mask_for_labels, self.final_label_count = self.__iterative_threshold(self.img_grey_dt)
        self.dilated_mask = self._dilate_mask(self.output_mask_for_labels)
        self.labels = self._initialize_labels(self.dilated_mask)
        self.labels_watershed = self._watershed_segmentation(self.img_rgb, self.labels)
        self.labels_watershed_filled = self._fill_ellipses(self.labels_watershed)
        self.labels_on_img = self._overlay_labels_on_rgb(
            self.img_rgb, cast(npt.NDArray[np.int32], self.labels_watershed_filled)
        )
        return (
            cast(npt.NDArray[np.uint8], self.labels_on_img),
            cast(npt.NDArray[np.int32], self.labels_watershed_filled),
            cast(npt.NDArray[np.uint8], self.img_grey_morph_eroded),
        )


class NormalWatershed(WatershedSegmentation):
    """Standard watershed segmentation implementation."""

    def __init__(self, params: dict[str, float | int]) -> None:
        self.name = "Default"
        self.description = (
            "The default watershed method that applies thresholds by three times, "
            "improved based on the first version of Bubble Analyser (Mesa et al., 2022), "
            "https://doi.org/10.1016/j.mineng.2022.107497"
        )
        self.sure_fg: npt.NDArray[np.uint8]
        self.sure_bg: npt.NDArray[np.uint8]
        self.unknown: MatLike
        self.threshold_value: float
        self.resample: float
        self.target_width: int
        self.if_bknd_img: bool = False
        self.update_params(params)

    def get_needed_params(self) -> dict[str, float | int]:
        return {
            "resample": self.resample,
            "high_thresh": self.high_thresh,
            "mid_thresh": self.mid_thresh,
            "low_thresh": self.low_thresh,
            "element_size": self.element_size,
            "connectivity": self.connectivity,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for each parameter for use in GUI tooltips.

        Returns:
            dict[str, str]: Dictionary mapping parameter names to their descriptions.
        """
        return guide_tooltips(DEFAULT_GUIDE)

    def get_param_guide(self) -> MethodGuide:
        """Get the parameter guide shown in the GUI's parameter guide window.

        Returns:
            MethodGuide: What the method does and what each parameter changes.
        """
        return DEFAULT_GUIDE

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
        self.img_grey = img_grey
        self.img_rgb = img_rgb
        self.bknd_img = bknd_img
        self.if_bknd_img = if_bknd_img
        self.cnn_model = cnn_model
        self.update_params(params)
        super().__init__(
            img_grey,
            img_rgb,
            if_bknd_img=if_bknd_img,
            bknd_img=bknd_img,
            element_size=self.element_size,
            connectivity=self.connectivity,
        )

    def update_params(self, params: dict[str, float | int]) -> None:
        self.resample = cast(float, params.get("resample", 0.4))
        self.high_thresh = cast(float, params.get("high_thresh", 0.95))
        self.mid_thresh = cast(float, params.get("mid_thresh", 0.7))
        self.low_thresh = cast(float, params.get("low_thresh", 0.05))
        self.element_size = cast(int, params.get("element_size", 5))
        self.connectivity = cast(int, params.get("connectivity", 4))

    def __get_sure_fg_bg(
        self, target_image: npt.NDArray[np.uint8], dt_thresh_image: MatLike
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], MatLike]:
        sure_fg_initial = dt_thresh_image.copy()
        sure_bg = np.array(cv2.dilate(target_image, np.ones((3, 3), np.uint8), iterations=1), dtype=np.uint8)
        sure_fg = np.array(sure_fg_initial, dtype=np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)
        return sure_fg, sure_bg, unknown

    def _three_way_threshold(
        self, target_image: MatLike, high_thresh: float, mid_thresh: float, low_thresh: float
    ) -> tuple[MatLike, int]:
        image_float = target_image.astype(np.float32) if isinstance(target_image, np.ndarray) else target_image
        output_mask = np.zeros_like(image_float, dtype=np.uint8)
        thresh_list = [high_thresh, mid_thresh, low_thresh]
        no_overlap_count = 0
        max_val = float(image_float.max())

        for current_thresh in thresh_list:
            _, thresholded = cv2.threshold(image_float, current_thresh * max_val, 255, cv2.THRESH_BINARY)
            thresholded_uint8 = thresholded.astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(thresholded_uint8, connectivity=self.connectivity)

            for label in range(1, num_labels):
                component_mask = (labels == label).astype(np.uint8) * 255
                overlap = cv2.bitwise_and(output_mask, component_mask)
                if not np.any(overlap):
                    no_overlap_count += 1
                    output_mask = cast(npt.NDArray[np.uint8], cv2.bitwise_or(output_mask, component_mask))
        final_label_count, _ = cv2.connectedComponents(output_mask)
        return output_mask, final_label_count

    def get_results_img(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32], npt.NDArray[np.uint8]]:
        self.img_grey_thresholded = self._threshold(self.img_grey)
        self.img_grey_morph, self.img_grey_morph_eroded = self._morph_process(self.img_grey_thresholded)
        self.img_grey_dt = self._dist_transform(self.img_grey_morph)
        self.img_grey_dt_thresh, self.final_label_count = self._three_way_threshold(
            self.img_grey_dt, self.high_thresh, self.mid_thresh, self.low_thresh
        )
        self.sure_fg, self.sure_bg, self.unknown = self.__get_sure_fg_bg(self.img_grey, self.img_grey_dt_thresh)
        self.labels = self._initialize_labels(self.sure_fg)
        self.labels_watershed = self._watershed_segmentation(self.img_rgb, self.labels)
        self.labels_on_img = self._overlay_labels_on_rgb(
            self.img_rgb, cast(npt.NDArray[np.int32], self.labels_watershed)
        )
        return (
            cast(npt.NDArray[np.uint8], self.labels_on_img),
            cast(npt.NDArray[np.int32], self.labels_watershed),
            cast(npt.NDArray[np.uint8], self.img_grey_morph_eroded),
        )
