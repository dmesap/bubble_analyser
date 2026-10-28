"""Module for handling circle detection, filtering parameter management, and property calculation in images.

This module provides functionality for managing filtering parameters through the FilterParamHandler class,
processing labeled image regions, filtering them based on geometric properties, fitting ellipses to the
filtered regions, and calculating various properties of the detected ellipses. The filtering parameters
can be dynamically configured to control circle detection thresholds, size constraints, and analysis
settings. It is primarily used for bubble/circle analysis in scientific images.
"""

import logging
from typing import Any

import cv2
import numpy as np
from numpy import typing as npt
from skimage import measure


class FilterParamHandler:
    """Handles filtering parameters for circle detection and analysis.

    This class manages a set of parameters used to filter and analyze circular regions in images.
    It provides functionality to store, retrieve, and update filtering parameters such as
    eccentricity, solidity, circularity, and size thresholds.

    Attributes:
        filter_param_dict_1 (dict[str, float | str]): Dictionary containing filtering parameters:
            - max_eccentricity: Maximum allowed eccentricity for valid circles
            - min_solidity: Minimum required solidity for valid circles
            - min_circularity: Minimum required circularity for valid circles
            - min_size: Minimum required size for valid circles
            - max_size: Maximum allowed size for valid circles
    """

    def __init__(self, params_dict: dict[str, float | str]):
        """Initialize FilterParamHandler with configuration parameters.

        Args:
            params_dict: Dictionary containing filter parameters for circle detection
                        and analysis.
        """
        self.filter_param_dict_1: dict[str, float | str] = {
            "max_eccentricity": params_dict["max_eccentricity"],
            "min_solidity": params_dict["min_solidity"],
            "min_circularity": params_dict["min_circularity"],
            "min_size": params_dict["min_size"],
            "max_size": params_dict["max_size"],
        }

    def get_needed_params(self) -> dict[str, float | str]:
        """Retrieve the current filter parameters.

        Returns:
            Dictionary containing the current filter parameters.
        """
        return self.filter_param_dict_1

    def update_params_1(self, params: dict[str, float | str]) -> None:
        """Update the filter parameters with new values.

        Args:
            params: Dictionary containing new filter parameter values to update.
        """
        self.filter_param_dict_1 = params

    def get_param_descriptions(self) -> dict[str, str]:
        """Get descriptions for each parameter for use in GUI tooltips.

        Returns:
            dict[str, str]: Dictionary mapping parameter names to their descriptions.
        """
        return {
            "min_size": "Minimum allowable diameter for a bubble in millimeters.",
            "max_size": "Maximum allowable diameter for a bubble in millimeters.",
            "max_eccentricity": "Maximum allowed elongation. 0.0 is a perfect circle; higher values allow more oval shapes.",  # noqa: E501
            "min_solidity": "Minimum ratio of area to convex hull. Filters out bubbles with deep 'dents' or irregular holes.",  # noqa: E501
            "min_circularity": "Minimum roundness and edge smoothness. 1.0 is a perfect circle; lower values allow rougher edges.",  # noqa: E501
        }


class EllipseHandler:
    """Handles the detection, filtering, and analysis of circular regions in labeled images.

    This class provides methods for filtering labeled regions based on geometric properties,
    fitting ellipses to the filtered regions, overlaying the detected ellipses on images,
    and calculating various properties of the detected ellipses.

    Attributes:
        filter_param_dict (dict[str, float]): Dictionary of filtering parameters.
        img_rgb (npt.NDArray[np.int_]): The RGB image being processed.
        labels_before_filtering (npt.NDArray[np.int_]): The labeled image before filtering.
        labels_after_filtering (npt.NDArray[np.int_]): The labeled image after filtering.
        labels_for_calculations (npt.NDArray[np.int_]): Labels used for calculations.
        px2mm (float): Conversion factor from pixels to millimeters.
        ellipses (list[tuple[tuple[float, float], tuple[int, int], int]]): List of detected ellipses.
        ellipses_on_image (npt.NDArray[np.int_]): Image with ellipses overlaid.
        ellipses_properties (list[dict[str, float]]): Properties of detected ellipses.
    """

    def __init__(
        self,
        labels_before_filtering: npt.NDArray[np.int_] | None = None,
        img_rgb: npt.NDArray[np.int_] | None = None,
        px2mm_display: float = 90.0,
        resample: float = 0.5,
    ) -> None:
        """Initialize the CircleHandler with labeled image data and conversion factor.

        Args:
            labels_before_filtering (npt.NDArray[np.int_]): The labeled image before filtering.
            img_rgb (npt.NDArray[np.int_]): The RGB image corresponding to the labeled image.
            px2mm_display (float, optional): Conversion factor from pixels to millimeters for display. Defaults to 90.0.
            resample (float, optional): Image resampling factor. Defaults to 0.5.
        """
        self.filter_param_dict_1: dict[str, float | str]
        self.img_rgb: npt.NDArray[np.int_] | None = img_rgb
        self.labels_before_filtering: npt.NDArray[np.int_] | None = labels_before_filtering
        self.labels_after_filtering: npt.NDArray[np.int_]
        self.labels_for_calculations: npt.NDArray[np.int_]

        self.px2mm_display: float = px2mm_display
        self.resample: float = resample
        self.real_px2mm = self.px2mm_display * self.resample
        self.mm2px = 1 / self.real_px2mm
        logging.info(f"Real px2mm: {self.real_px2mm} = ({self.px2mm_display}*{self.resample})")

        self.ellipses: list[tuple[tuple[float, float], tuple[int, int], float]]
        self.ellipses_on_image: npt.NDArray[np.int_]
        self.ellipses_properties: list[dict[str, Any]]

    def load_filter_params(self, filter_param_dict_1: dict[str, float | str]) -> None:
        """Load filtering parameters for circle detection.

        Args:
            filter_param_dict_1 (dict[str, float | str]): Dictionary containing filtering parameters
                such as max_eccentricity, min_solidity, min_circularity, and min_size.
        """
        self.filter_param_dict_1 = filter_param_dict_1

    @staticmethod
    def detect_background_label(labels: npt.NDArray[np.int_]) -> int:
        """Infer the background label from border pixels instead of assuming 1.

        Args:
            labels: 2D integer labeled array.

        Returns:
            int: The inferred background label value.
        """
        if labels.size == 0:
            return 1
        border = np.concatenate(
            [
                labels[0, :].ravel(),
                labels[-1, :].ravel(),
                labels[:, 0].ravel(),
                labels[:, -1].ravel(),
            ]
        )
        values, counts = np.unique(border, return_counts=True)
        return int(values[int(np.argmax(counts))])

    def filter_labels_properties(self) -> npt.NDArray[np.int_]:
        """Filters out regions (circles) from the labeled image based on their properties.

        This function filters regions based on:
        - Geometric properties (eccentricity, solidity, circularity)
        - Size constraints (min/max area)
        - Boundary touching (regions touching image edges are removed)

        Returns:
            Updated labels array where regions not meeting the thresholds are removed.
        """
        labels = self.labels_before_filtering

        logging.basicConfig(level=logging.INFO)
        properties = measure.regionprops(labels)
        new_labels = np.copy(labels) if labels is not None else np.array([])
        mm2px = self.mm2px

        # Get image dimensions for boundary detection
        if labels is not None:
            _img_height, _img_width = labels.shape
        else:
            raise ValueError("labels_before_filtering is None")

        bg_label = self.detect_background_label(labels)

        max_eccentricity = float(self.filter_param_dict_1["max_eccentricity"])
        min_solidity = float(self.filter_param_dict_1["min_solidity"])
        min_circularity = float(self.filter_param_dict_1["min_circularity"])
        min_size = float(self.filter_param_dict_1["min_size"])
        max_size = float(self.filter_param_dict_1["max_size"])

        # Back-calculate diameter thresholds (mm) to area thresholds (pixels)
        min_area_px = (np.pi * (min_size / 2) ** 2) / (mm2px**2)
        max_area_px = (np.pi * (max_size / 2) ** 2) / (mm2px**2)

        for prop in properties:
            if prop.label == bg_label:  # Ignore the background
                continue

            # Check all conditions and mark for removal if any fail
            should_remove = False

            # Priority 1: Size Filtering
            if not (min_area_px <= prop.area <= max_area_px):
                should_remove = True
                logging.debug(f"Filtered out by Size: {prop.area:.1f} px")

            # Priority 2: Shape Filtering (Standard)
            elif prop.eccentricity > max_eccentricity or prop.solidity < min_solidity:
                should_remove = True
                logging.debug(f"Filtered out by Shape: Ecc={prop.eccentricity:.2f}, Sol={prop.solidity:.2f}")

            # Priority 3: Shape Filtering (Circularity)
            else:
                # Circularity = 4 * pi * Area / Perimeter^2
                circularity = (4 * np.pi * prop.area) / (prop.perimeter**2) if prop.perimeter > 0 else 0
                if circularity < min_circularity:
                    should_remove = True
                    logging.debug(f"Filtered out by Circularity: {circularity:.2f}")

            if should_remove:
                new_labels[tuple(prop.coords.T)] = bg_label

        self.labels_after_filtering = new_labels

        return new_labels

    def fill_ellipse_labels(
        self,
    ) -> list[tuple[tuple[float, float], tuple[int, int], float]]:
        """Fill each ellipse label in labels_before_filtering and return a new label object.

        Returns:
            list[tuple]: A list of fitted ellipses (center, axes, angle).
        """
        # Store ellipses
        ellipses = []

        # Use regionprops for optimized component access
        properties = measure.regionprops(self.labels_after_filtering)
        bg_label = self.detect_background_label(self.labels_after_filtering)

        for prop in properties:
            if prop.label == 0 or prop.label == bg_label:
                continue

            # Create a small local mask for this bubble using its bounding box
            # This is significantly faster than a full-image mask
            min_r, min_c, max_r, max_c = prop.bbox

            # Add padding to ensure contour is closed correctly
            local_mask = np.zeros((max_r - min_r + 2, max_c - min_c + 2), dtype=np.uint8)

            # Efficiently map global coords to local mask
            local_coords = prop.coords - [min_r, min_c]
            local_mask[local_coords[:, 0] + 1, local_coords[:, 1] + 1] = 255

            contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    ellipse_width, ellipse_height = axes

                    # Only add valid ellipses (width and height > 0, finite values)
                    if (
                        ellipse_width > 0
                        and ellipse_height > 0
                        and np.isfinite(ellipse_width)
                        and np.isfinite(ellipse_height)
                        and np.isfinite(center[0])
                        and np.isfinite(center[1])
                        and np.isfinite(angle)
                    ):
                        # Offset the center back to global coordinates
                        global_center = (center[0] + min_c - 1, center[1] + min_r - 1)
                        ellipses.append((global_center, axes, angle))
                    else:
                        logging.debug(f"Skipping invalid ellipse with dimensions: {axes}")

        self.ellipses = ellipses  # type: ignore

        return self.ellipses  # type: ignore

    def overlay_ellipses_on_image(self, thickness: int = 5) -> npt.NDArray[np.int_]:
        """Overlay detected ellipses on the RGB image.

        Draws each detected ellipse on the RGB image with the specified thickness.
        Also creates a labeled image from the ellipses.

        Args:
            thickness (int, optional): Thickness of the ellipse outlines. Defaults to 20.

        Returns:
            npt.NDArray[np.int_]: The RGB image with ellipses overlaid.
        """
        if self.img_rgb is None:
            raise ValueError("img_rgb is not initialized")
        ellipse_image = self.img_rgb.copy()

        for ellipse in self.ellipses:
            # Validate ellipse parameters before drawing
            center, axes, angle = ellipse
            ellipse_width, ellipse_height = axes

            # Check if ellipse dimensions are valid (positive, finite values)
            if (
                ellipse_width > 0
                and ellipse_height > 0
                and thickness > 0
                and np.isfinite(ellipse_width)
                and np.isfinite(ellipse_height)
                and np.isfinite(center[0])
                and np.isfinite(center[1])
                and np.isfinite(angle)
            ):
                try:
                    cv2.ellipse(ellipse_image, ellipse, (0, 0, 255), thickness)  # type: ignore
                except cv2.error as e:
                    logging.warning(f"Failed to draw ellipse {ellipse}: {e}")
                    continue
            else:
                logging.warning(
                    f"Invalid ellipse dimensions: width={ellipse_width}, height={ellipse_height}, thickness={thickness}"
                )
                continue

        self.ellipses_on_image = ellipse_image

        self.create_labelled_image_from_ellipses()

        return ellipse_image

    def create_labelled_image_from_ellipses(self) -> npt.NDArray[np.int_]:
        """Creates a labelled image based on the ellipses fitted.

        The resulting image is a 2D array with the same height and width as self.img_rgb where:
          - Background pixels have a value of 1.
          - Each ellipse is filled with a unique label (starting from 2).

        Returns:
            A labelled image as a numpy array of type np.int_.
        """
        if self.img_rgb is None:
            raise ValueError("img_rgb is not initialized")
        img_height, img_width = self.img_rgb.shape[:2]

        # Initialize the labelled image with background label (1)
        labelled_img = np.ones((img_height, img_width), dtype=np.int_)

        current_label = 2  # Start labelling from 2
        for ellipse in self.ellipses:
            # Validate ellipse parameters before drawing
            _center, axes, _angle = ellipse
            ellipse_width, ellipse_height = axes

            # Skip invalid ellipses (width or height <= 0, inf, or nan)
            if (
                ellipse_width <= 0
                or ellipse_height <= 0
                or not np.isfinite(ellipse_width)
                or not np.isfinite(ellipse_height)
            ):
                continue

            # Create a mask for the current ellipse.
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            try:
                cv2.ellipse(mask, ellipse, color=255, thickness=-1)  # type: ignore
                # Assign the current label to all pixels inside the ellipse.
                labelled_img[mask == 255] = current_label
                current_label += 1
            except cv2.error as e:
                # Log the error and skip this ellipse
                print(f"Warning: Skipping invalid ellipse {ellipse}: {e}")
                continue

        return labelled_img

    def calculate_circle_properties(self) -> list[dict[str, Any]]:
        """Calculate geometric properties for each detected ellipse.

        Computes various properties for each ellipse including major and minor axis lengths,
        area, perimeter, eccentricity, and equivalent diameter. All measurements are converted
        to millimeters using the px2mm conversion factor.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each containing
                the properties of one ellipse.
        """
        ellipse_properties = []
        mm2px = self.mm2px

        for ellipse in self.ellipses:
            _center, axes, _angle = ellipse
            major_axis_length = max(axes) * mm2px
            minor_axis_length = min(axes) * mm2px
            # major_axis_length = max(axes)
            # minor_axis_length = min(axes)
            area = np.pi * (major_axis_length / 2) * (minor_axis_length / 2)
            perimeter = np.pi * (
                3 * (major_axis_length + minor_axis_length)
                - np.sqrt((3 * major_axis_length + minor_axis_length) * (major_axis_length + 3 * minor_axis_length))
            )
            eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            ellipse_properties.append(
                {
                    "major_axis_length": major_axis_length,
                    "minor_axis_length": minor_axis_length,
                    "equivalent_diameter": equivalent_diameter,
                    "area": area,
                    "perimeter": perimeter,
                    "eccentricity": eccentricity,
                }
            )

        self.ellipses_properties = ellipse_properties
        # print(ellipse_properties)
        return ellipse_properties


if __name__ == "__main__":
    input_labels_path = "../../tests/test_labels_watershed.npy"
    output_filled_labels_path = "../../tests/test_labels_after_fill.npy"
    output_filled_labels_img_path = "../../tests/test_labels_after_fill.JPG"

    img_path = "../../tests/test_image_rgb.JPG"
    img_rgb: npt.NDArray[np.int_] = cv2.imread(img_path)  # type: ignore

    # Load the input label object
    labels_before_filtering = np.load(input_labels_path)

    # Create an instance of CircleHandler
    circle_handler = EllipseHandler(labels_before_filtering, img_rgb)
    ellipse_image_path = "../../tests/test_image_rgb.JPG"
    ellipse_image: npt.NDArray[np.int_] = cv2.imread(ellipse_image_path)  # type: ignore
    circle_handler.img_rgb = ellipse_image  # type: ignore

    # Fill ellipse labels
    _ = circle_handler.fill_ellipse_labels()
    filled_labels = circle_handler.overlay_ellipses_on_image()
    circle_handler.calculate_circle_properties()
    # Save the filled labels
    np.save(output_filled_labels_path, filled_labels)
    cv2.imwrite(output_filled_labels_img_path, filled_labels)
    print(f"Filled labels saved to {output_filled_labels_path}")
