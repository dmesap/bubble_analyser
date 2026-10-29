"""Component handlers module for the Bubble Analyser application.

This module provides model classes and handlers for managing various components of the
application, including image processing, calibration, and data management. It serves as
the bridge between the GUI interface and the underlying processing functionality.

The module contains the following key components:
- WorkerThread: A thread class for handling background processing tasks
- InputFilesModel: Model for managing input image files and paths
- CalibrationModel: Model for managing calibration data and pixel-to-millimeter
    conversion
- ImageProcessingModel: Model for managing image processing operations and parameters

These components work together to provide a structured approach to image processing,
data management, and user interface interaction in the Bubble Analyser application.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy import typing as npt
from PySide6.QtCore import QEventLoop, QObject, QThread, Signal

from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskConfig, BubMaskDetector
from bubble_analyser.core.models import ImageState
from bubble_analyser.processing import (
    Config,
    EllipseAdjuster,
    FilterParamHandler,
    MethodsHandler,
)


class WorkerThread(QThread):
    """A worker thread class for handling batch image processing operations.

    This class extends QThread to perform image processing tasks in the background,
    preventing the GUI from freezing during lengthy operations. It provides progress
    updates and completion signals.

    Attributes:
        update_progress (Signal[int]): Signal emitted to update the progress bar.
        processing_done (Signal): Signal emitted when processing is complete.
        error_occurred (Signal[str]): Signal emitted when an error occurs during processing.
        model (ImageProcessingModel): The model containing image processing logic.
    """

    update_progress = Signal(int)
    processing_done = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        model: "ImageProcessingModel",
    ) -> None:
        """Initialize the worker thread with processing parameters.

        Args:
            model (ImageProcessingModel): The model containing image processing logic.
        """
        super().__init__()
        self.model: ImageProcessingModel = model

    def run(self) -> None:
        """Execute the batch processing operation.

        This method is called when the thread starts. It delegates the actual processing
        to the model's batch_process_images method.
        """
        try:
            self.model.batch_process_images(self)
        except Exception as e:
            # Log the error instead of showing a dialog from worker thread
            import logging
            import traceback

            error_details = traceback.format_exc()
            logging.error(f"Error in worker thread: {error_details}")

            # Emit error signal to be handled on the main thread
            self.error_occurred.emit(f"Processing error: {e!s}\n\nDetails:\n{error_details}")

    def update_progress_bar(self, value: int) -> None:
        """Emit a signal to update the progress bar in the GUI.

        Args:
            value (int): The current progress value to display.
        """
        self.update_progress.emit(value)

    def on_processing_done(self) -> None:
        """Emit a signal indicating that processing is complete.

        This method is called when all images have been processed.
        """
        self.processing_done.emit()


class Step1Worker(QThread):
    """A worker thread for handling the first step of image processing (segmentation).

    This class runs the segmentation process in a background thread and emits
    signals upon completion or error.
    """

    finished = Signal(object)  # Emits the processed image (numpy array)
    error = Signal(str)

    def __init__(self, model: "ImageProcessingModel", index: int) -> None:
        """Initialize the Step 1 worker.

        Args:
            model (ImageProcessingModel): The image processing model.
            index (int): The index of the image to process.
        """
        super().__init__()
        self.model = model
        self.index = index

    def run(self) -> None:
        """Execute the segmentation process."""
        try:
            img = self.model.step_1_main(self.index)
            self.finished.emit(img)
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.error.emit(f"Error in Step 1 processing: {e!s}\n\n{error_details}")


class Step2Worker(QThread):
    """A worker thread for handling the second step of image processing (filtering).

    This class runs the filtering process in a background thread and emits
    signals upon completion or error.
    """

    finished = Signal(object)  # Emits the processed image (numpy array)
    error = Signal(str)

    def __init__(self, model: "ImageProcessingModel", index: int) -> None:
        """Initialize the Step 2 worker.

        Args:
            model (ImageProcessingModel): The image processing model.
            index (int): The index of the image to process.
        """
        super().__init__()
        self.model = model
        self.index = index

    def run(self) -> None:
        """Execute the filtering process."""
        try:
            img = self.model.step_2_main(self.index)
            self.finished.emit(img)
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.error.emit(f"Error in Step 2 processing: {e!s}\n\n{error_details}")


class ImageProcessingModel(QObject):
    """A model class for managing image processing operations and parameters.

    This class handles the processing of images using various algorithms, maintains
    processing parameters, and manages the state of processed images. It interfaces
    with the MethodsHandler to access processing routines and stores the results of
    image processing operations.

    Attributes:
        algorithm (str): The currently selected processing algorithm name.
        params (Config): Configuration parameters for image processing.
        filter_param_dict (dict[str, float]): Dictionary of filtering parameters.
        px2mm (float): Conversion factor from pixels to millimeters.
        if_bknd (bool): Flag indicating whether a background image is being used.
        bknd_img_path (Path): Path to the background image file.
        img_path_list (list[Path]): List of paths to images to be processed.
        img_dict (dict[Path, ImageState]): Dictionary mapping image paths to their Image
            objects.
        adjuster (EllipseAdjuster): Tool for manual adjustment of detected ellipses.
        ellipses_properties (list[list[dict[str, float]]]): Properties of detected
            ellipses
        for all images. methods_handler (MethodsHandler): Handler for accessing
            processing methods.
        all_methods_n_params (dict): Dictionary of all available methods and their
            parameters.
    """

    def __init__(self, params: Config) -> None:
        """Initialize the ImageProcessingModel with the provided configuration.

        Args:
            params (Config): Configuration parameters for image processing.
        """
        super().__init__()

        self.algorithm: str = ""
        self.params_config: Config = params
        self.if_batched: bool = False
        self.if_finalise_analysis: bool = False

        self.filter_param_dict_1: dict[str, Any] = {}

        self.px2mm_display = 1.0
        self.if_bknd = False
        self.bubble_count: int = 0
        self.bknd_img_path: Path = cast(Path, None)

        self.img_path_list: list[Path] = []
        self.img_dict: dict[Path, ImageState] = {}

        self.adjuster: EllipseAdjuster
        self.ellipses_properties: list[list[dict[str, Any]]] = []

        # Determine base directory for weights
        if getattr(sys, "frozen", False):
            # If the application is run as a bundle (PyInstaller)
            base_dir = getattr(sys, "_MEIPASS", "")
        else:
            # If running in development mode
            # component_handlers.py is in bubble_analyser/gui/
            # We need to go up two levels to get to the project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.weights_path: str = os.path.join(base_dir, "bubble_analyser/weights/mask_rcnn_bubble.h5")
        self.confidence_threshold: float = 0.9
        self.target_width: int = 1000
        self.image_min_dim: int = 192 * 2
        self.image_max_dim: int = 384 * 2
        self.detector: BubMaskDetector | None = None

        logging.info("------------------------------Intializing Parameters------------------------------")
        self.methods_handler: MethodsHandler
        self.filter_param_handler: FilterParamHandler
        self.initialize_methods_handlers()
        self.initialize_filter_param_handler()

        # Defer the heavy CNN initialization so the GUI boots instantly
        from PySide6.QtCore import QTimer

        QTimer.singleShot(100, self.initialize_cnn_model)

    def initialize_methods_handlers(self) -> None:
        """Initialize the methods handler and retrieve available processing methods.

        This method creates a new MethodsHandler instance using the current
        configuration and retrieves the dictionary of available processing methods
        and their parameters.
        """
        self.methods_handler = MethodsHandler(self.params_config)
        self.all_methods_n_params = self.methods_handler.full_dict
        logging.info(f"All detected methods and their parameters: {self.all_methods_n_params}")

    def initialize_filter_param_handler(self) -> None:
        """Initialize the filter parameter handler and retrieve filtering parameters.

        This method creates a new FilterParamHandler instance using the current
        configuration parameters and retrieves the dictionary of needed filtering
        parameters. The filter parameters are used to control various aspects of
        the image processing pipeline such as thresholds, sizes, and other
        filtering criteria.

        The parameters are also printed to the console for debugging purposes.
        """
        self.filter_param_handler = FilterParamHandler(self.params_config.model_dump())
        self.filter_param_dict_1 = self.filter_param_handler.get_needed_params()
        logging.info(f"Basic filtering parameters: {self.filter_param_dict_1}")

    def initialize_cnn_model(self) -> None:
        import os

        # Initialize BubMask detector if not already done
        if self.detector is None and self.weights_path:
            # 1. Safety Check: Does the file actually exist?
            if not os.path.exists(self.weights_path):
                logging.warning(f"Weights file not found at: {self.weights_path}")
                logging.warning("CNN-based segmentation will be disabled.")
                self.detector = None
                return  # Exit gracefully, do not crash!

            # 2. File exists, proceed to load
            try:
                config = BubMaskConfig(
                    confidence_threshold=self.confidence_threshold,
                    image_min_dim=self.image_min_dim,
                    image_max_dim=self.image_max_dim,
                )
                self.detector = BubMaskDetector(self.weights_path, config)
                logging.info("BubMask detector initialized successfully")
            except Exception as e:
                # 3. Catch other errors (e.g. corrupt file) without crashing app
                logging.error(f"Failed to initialize BubMask detector: {e}")
                self.detector = None
                # removed 'raise' so the app continues running

    def confirm_folder_selection(self, folder_path_list: list[Path]) -> None:
        """Set the list of image paths to be processed.

        Args:
            folder_path_list (list[Path]): List of paths to images to be processed.
        """
        # Clear existing state to prevent race conditions during re-processing
        self.img_path_list = folder_path_list
        self.img_dict.clear()
        self.ellipses_properties.clear()
        self.if_batched = False
        self.if_finalise_analysis = False

    def get_bknd_img_path(self, bknd_img_path: Path) -> None:
        """Set the path to the background image.

        Args:
            bknd_img_path (Path): Path to the background image file.
        """
        self.bknd_img_path = Path(bknd_img_path)

    def reset_batch_state(self, force_all: bool = False) -> None:
        """Clear cached processed results.

        Args:
            force_all (bool): If True, clear caches for all images including fine-tuned ones.
        """
        self.if_batched = False
        self.if_finalise_analysis = False
        for state in self.img_dict.values():
            if force_all or not state.if_fine_tuned:
                state.ellipses_on_images = np.zeros((0, 0, 3), dtype=np.uint8)
                state.labels_on_img_before_filter = np.zeros((0, 0, 3), dtype=np.uint8)
                if force_all:
                    state.if_fine_tuned = False

    def update_px2mm_display(self, px2mm_display: float) -> None:
        """Update the pixel-to-millimeter conversion ratio.

        Args:
            px2mm_display (float): The new pixel-to-millimeter conversion ratio for display.
        """
        self.px2mm_display = px2mm_display

    def preview_processed_image(self, index: int) -> tuple[bool, npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Retrieve the processed images for preview.

        Args:
            index (int): Index of the image in the image list.

        Returns:
            tuple[bool, npt.NDArray[np.int_], npt.NDArray[np.int_]]: A tuple containing:
                - Boolean indicating if the image exists and has been processed
                - The image with labels before filtering
                - The image with ellipses overlaid after filtering
        """
        name = self.img_path_list[index]
        if_img = False
        img_before_filter = cast(npt.NDArray[np.int_], None)
        img_after_filter = cast(npt.NDArray[np.int_], None)
        if name in self.img_dict:
            img_before_filter = self.img_dict[name].labels_on_img_before_filter

            try:
                img_after_filter = self.img_dict[name].ellipses_on_images
            except AttributeError as e:
                print(e)
                img_after_filter = cast(npt.NDArray[np.int_], None)
                return (
                    False,
                    cast(npt.NDArray[np.int_], None),
                    cast(npt.NDArray[np.int_], None),
                )
            if_img = True

            self.img_dict[name].set_fine_tuned()

        return if_img, img_before_filter, img_after_filter

    def load_filter_params(self, dict_params_1: dict[str, float | str]) -> None:
        """Load filtering parameters into the model.

        Args:
            dict_params_1 (dict[str, float]): Dictionary containing filtering parameters.
        """
        self.filter_param_dict_1 = dict_params_1

        # Sync with the actual handler used by services
        self.filter_param_handler.update_params_1(dict_params_1)

    def initialize_image(self, name: Path) -> None:
        """Initialize an ImageState object for processing if it doesn't already exist.

        Args:
            name (Path): The path of the image to initialize.
        """
        if name not in self.img_dict:
            self.img_dict[name] = ImageState(
                raw_img_path=name,
                px2mm_display=self.px2mm_display,
                bknd_img_path=self.bknd_img_path if self.if_bknd else None,
            )

    def step_1_main(self, index: int) -> npt.NDArray[np.int_]:
        """Execute the first step of image processing (pre-filtering).

        Args:
            index (int): Index of the image to process.

        Returns:
            npt.NDArray[np.int_]: The processed image with labels before filtering.
        """
        name = self.img_path_list[index]
        self.initialize_image(name)

        from bubble_analyser.core.services import AnalysisService

        service = AnalysisService(self.params_config)
        service.algorithm = self.algorithm
        service.px2mm_display = self.px2mm_display
        service.bknd_img_path = self.bknd_img_path
        service.set_detector(self.detector)
        service.setup_methods(self.methods_handler, self.filter_param_handler)

        # Use the service to process the image state
        result = service.process_image(name, skip_filtering=True)

        if result.success and result.labels_on_img_before_filter is not None:
            # Sync back to our state cache for UI consistency
            state = self.img_dict[name]
            state.labels_on_img_before_filter = result.labels_on_img_before_filter

            if result.labels_before_filter is not None:
                state.labels_before_filter = result.labels_before_filter
            if result.img_grey is not None:
                state.img_grey = result.img_grey
            if result.img_rgb is not None:
                state.img_rgb = result.img_rgb
            if result.img_grey_morph_eroded is not None:
                state.img_grey_morph_eroded = result.img_grey_morph_eroded

            return result.labels_on_img_before_filter

        return np.zeros((0, 0), dtype=np.int_)

    def step_2_main(self, index: int) -> npt.NDArray[np.int_]:
        """Execute the second step of image processing (filtering&ellipse detection).

        Args:
            index (int): Index of the image to process.

        Returns:
            npt.NDArray[np.int_]: The processed image with detected ellipses overlaid.
        """
        name = self.img_path_list[index]
        self.initialize_image(name)
        from bubble_analyser.core.services import AnalysisService

        service = AnalysisService(self.params_config)
        service.algorithm = self.algorithm
        service.px2mm_display = self.px2mm_display
        service.bknd_img_path = self.bknd_img_path
        service.set_detector(self.detector)
        service.setup_methods(self.methods_handler, self.filter_param_handler)

        result = service.process_image(name, existing_state=self.img_dict[name])

        if result.success and result.ellipses_on_images is not None:
            # Sync back to our state cache
            state = self.img_dict[name]
            state.ellipses_on_images = result.ellipses_on_images
            state.ellipses_properties = result.ellipses_properties if result.ellipses_properties else []
            # We don't have ellipses objects here yet in Result, but for UI preview it's enough
            return result.ellipses_on_images

        return np.zeros((0, 0), dtype=np.int_)

    def refresh_ellipses_properties(self) -> list[list[dict[str, Any]]]:
        """Synchronize self.ellipses_properties from all loaded images in self.img_path_list.

        Ensures that any manual fine-tuning or newly processed image states are
        reflected in the master ellipses_properties list in the order of img_path_list.

        Returns:
            list[list[dict[str, Any]]]: The updated master list of ellipse properties.
        """
        properties: list[list[dict[str, Any]]] = []
        for index, path in enumerate(self.img_path_list):
            state = self.img_dict.get(path)
            if state is not None and (state.if_fine_tuned or len(state.ellipses_properties) > 0):
                properties.append(state.ellipses_properties)
            elif index < len(self.ellipses_properties):
                properties.append(self.ellipses_properties[index])
            else:
                properties.append([])
        self.ellipses_properties = properties
        self.bubble_count = sum(len(props) for props in self.ellipses_properties)
        return self.ellipses_properties

    def get_or_render_annotated_image(self, name: Path) -> npt.NDArray[np.uint8] | None:
        """Get or dynamically render the annotated image with ellipses overlaid.

        Args:
            name (Path): The path of the image to retrieve or render.

        Returns:
            npt.NDArray[np.uint8] | None: The RGB image array with overlaid ellipses, or None if unavailable.
        """
        state = self.img_dict.get(name)
        if state is None:
            return None

        # Return cached overlay if valid
        if state.ellipses_on_images is not None and state.ellipses_on_images.size > 0:
            return np.asarray(state.ellipses_on_images, dtype=np.uint8)

        # Reload img_rgb if cleared from memory
        if state.img_rgb is None or state.img_rgb.size == 0:
            from bubble_analyser.processing.image_preprocess import image_preprocess

            try:
                _, state.img_rgb = image_preprocess(state.raw_img_path, state.resample or 0.5)
            except Exception as e:
                logging.error(f"Failed to load RGB image for {name}: {e}")
                return None

        # If ellipses exist, render them onto the RGB image
        if state.ellipses and state.img_rgb is not None and state.img_rgb.size > 0:
            from bubble_analyser.processing.circle_handler import EllipseHandler as CircleHandler

            labels = (
                state.labels_before_filter.copy()
                if (state.labels_before_filter is not None and state.labels_before_filter.size > 0)
                else None
            )
            handler = CircleHandler(
                labels,
                state.img_rgb.copy(),
                self.px2mm_display,
                resample=state.resample or 0.5,
            )
            handler.ellipses = state.ellipses
            state.ellipses_on_images = handler.overlay_ellipses_on_image()
            return np.asarray(state.ellipses_on_images, dtype=np.uint8)

        if state.img_rgb is not None and state.img_rgb.size > 0:
            return np.asarray(state.img_rgb, dtype=np.uint8)

        return None

    def ellipse_manual_adjustment(self, index: int) -> npt.NDArray[np.int_]:
        """Launch the ellipse adjustment tool for manual fine-tuning of ellipses.

        This method creates an EllipseAdjuster instance for the specified image,
        displays the adjustment interface, and waits for the user to complete the
        adjustments.

        Args:
            index (int): Index of the image to adjust.

        Returns:
            npt.NDArray[np.int_]: The updated image with adjusted ellipses overlaid.
        """
        logging.info("Ellipse handler triggered.")
        name = self.img_path_list[index]
        image = self.img_dict[name]

        if image.img_rgb is None or image.img_rgb.size == 0:
            from bubble_analyser.processing.image_preprocess import image_preprocess

            _, image.img_rgb = image_preprocess(image.raw_img_path, image.resample or 0.5)

        self.adjuster = EllipseAdjuster(image.ellipses, image.img_rgb)

        loop = QEventLoop()

        def on_finished() -> None:
            self.handle_ellipse_adjustment_finished(image)
            loop.quit()

        self.adjuster.finished.connect(on_finished)
        self.adjuster.show()

        loop.exec()
        logging.info("Ellipse handler finished.")
        return image.ellipses_on_images

    def label_image_fine_tuned(self, image: ImageState) -> None:
        image.set_fine_tuned()

    def handle_ellipse_adjustment_finished(self, image: ImageState) -> None:
        """Process the results of manual ellipse adjustment.

        This method is called when the user completes the manual adjustment process.
        It updates the image's ellipses with the adjusted ones and regenerates
        the overlay.

        Args:
            image (ImageState): The image state object containing the ellipses to update.
        """
        image.ellipses = self.adjuster.ellipses
        image.set_fine_tuned()

        if image.img_rgb is None or image.img_rgb.size == 0:
            from bubble_analyser.processing.image_preprocess import image_preprocess

            _, image.img_rgb = image_preprocess(image.raw_img_path, image.resample or 0.5)

        # Regenerate overlays since ellipses changed
        from bubble_analyser.processing.circle_handler import EllipseHandler as CircleHandler

        labels = (
            image.labels_before_filter.copy()
            if (image.labels_before_filter is not None and image.labels_before_filter.size > 0)
            else None
        )
        handler = CircleHandler(
            labels,
            image.img_rgb.copy(),
            self.px2mm_display,
            resample=image.resample or 0.5,
        )
        handler.ellipses = image.ellipses
        image.ellipses_on_images = handler.overlay_ellipses_on_image()
        image.labelled_ellipses_mask = handler.create_labelled_image_from_ellipses()
        image.ellipses_properties = handler.calculate_circle_properties()
        for ellipse in image.ellipses_properties:
            ellipse["filename"] = image.raw_img_path.name

        # Immediately update the master properties list so math/histogram/export stay in sync
        try:
            index = self.img_path_list.index(image.raw_img_path)
            while len(self.ellipses_properties) <= index:
                self.ellipses_properties.append([])
            self.ellipses_properties[index] = image.ellipses_properties
        except ValueError:
            pass
        self.refresh_ellipses_properties()

    def batch_process_images(
        self,
        worker_thread: WorkerThread,
    ) -> None:
        """Process all images in the image list using the current parameters.

        This method iterates through all images in the image list, applies the current
        processing parameters. It updates the progress through the provided worker thread.

        Args:
            worker_thread (WorkerThread): Thread object for progress reporting.
        """
        self.bubble_count = 0
        self.ellipses_properties = []
        logging.info("------------------------------Batch Process Started------------------------------")

        from bubble_analyser.core.services import AnalysisService

        service = AnalysisService(self.params_config)
        service.algorithm = self.algorithm
        service.px2mm_display = self.px2mm_display
        service.bknd_img_path = self.bknd_img_path
        service.set_detector(self.detector)
        service.setup_methods(self.methods_handler, self.filter_param_handler)

        # Process every image in the list
        for index, name in enumerate(self.img_path_list):
            logging.info(f"***Processing image {index + 1}/{len(self.img_path_list)}: {name}***")

            self.initialize_image(name)
            state = self.img_dict[name]

            if self.if_finalise_analysis:
                # If we are finalising, we only need to ensure properties are in our master list
                if not state.ellipses_properties:
                    # If properties are missing, calculate them
                    result = service.process_image(name)
                    state.ellipses_properties = (
                        result.ellipses_properties if result.ellipses_properties is not None else []
                    )

                self.ellipses_properties.append(state.ellipses_properties)

            else:
                if not state.if_fine_tuned:
                    # Only process if results aren't already cached
                    if state.ellipses_on_images is None or not state.ellipses_on_images.any():
                        result = service.process_image(name)
                        if result.success:
                            # Sync state
                            if result.labels_on_img_before_filter is not None:
                                state.labels_on_img_before_filter = result.labels_on_img_before_filter
                            if result.ellipses_on_images is not None:
                                state.ellipses_on_images = result.ellipses_on_images
                            if result.img_rgb is not None:
                                state.img_rgb = result.img_rgb
                            if result.img_grey_morph_eroded is not None:
                                state.img_grey_morph_eroded = result.img_grey_morph_eroded
                            if result.labelled_ellipses_mask is not None:
                                state.labelled_ellipses_mask = result.labelled_ellipses_mask
                            if result.ellipses is not None:
                                state.ellipses = result.ellipses

                            state.ellipses_properties = result.ellipses_properties if result.ellipses_properties else []
                            self.ellipses_properties.append(state.ellipses_properties)
                        else:
                            self.ellipses_properties.append([])
                    else:
                        logging.info(f"Skipping already processed image: {name}")
                        self.ellipses_properties.append(state.ellipses_properties)

                else:
                    logging.info(f"This image has been fine tuned: {name}, no need to process again.")
                    # Recalculate properties if needed or just use existing
                    self.ellipses_properties.append(state.ellipses_properties)

            # MEMORY OPTIMIZATION: Clear large hidden numpy arrays from state to prevent Out-Of-Memory errors
            # on large batches. We keep the display arrays (img_rgb, ellipses_on_images) so previews still work.
            if getattr(self, "img_path", None) != name:
                state.clear_memory()

            worker_thread.update_progress_bar(index + 1)

            import gc

            if index % 10 == 0:
                gc.collect()

        worker_thread.on_processing_done()

        self.if_batched = True  # Make finalise analysis true
        self.refresh_ellipses_properties()

    def save_processed_images(
        self, img: npt.NDArray[np.int_], img_name: Path, save_path: Path, if_mt: bool = False
    ) -> None:
        """Save the processed image with detected ellipses to disk.

        Args:
            img (npt.NDArray[np.int_]): The processed image array to save.
            img_name (Path): Original image path used to generate the output filename.
            save_path (Path): Directory where the image should be saved.
            if_mt (bool): Whether multi-threading is being used.
        """
        file_name = os.path.basename(img_name)
        new_name = os.path.join(save_path, file_name)
        logging.info(f"Processed image with ellipses saving to: {new_name}")
        try:
            img_to_save = img * 255 if if_mt else img
            img_uint8 = np.asarray(img_to_save, dtype=np.uint8)
            if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
                bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            else:
                bgr = img_uint8
            success, enc = cv2.imencode(".png", bgr)
            if success:
                enc.tofile(new_name)
                logging.info("saved")
            else:
                cv2.imwrite(new_name, bgr)
                logging.info("saved")
        except Exception as e:
            logging.error(f"Error saving processed image {new_name}: {e}")

    def save_labelled_masks(self, img: npt.NDArray[np.int_], img_name: Path, save_path: Path) -> None:
        """Save the labelled mask image to disk.

        This method saves a mask image where each detected ellipse is labeled with a
        unique identifier.
        The output filename is the original filename with '_mask.png' appended.

        Args:
            img (npt.NDArray[np.int_]): The mask image array to save.
            img_name (Path): Original image path used to generate the output filename.
            save_path (Path): Directory where the mask should be saved.
        """
        file_name = os.path.basename(img_name)
        new_name = os.path.join(save_path, f"{file_name}_mask.png")
        logging.info(f"Labelled mask saving to: {new_name}")
        try:
            cv2.imwrite(new_name, img)
            logging.info("saved")
        except Exception as e:
            logging.info(e)
