"""Event handlers module for the Bubble Analyser application.

This module provides a comprehensive set of event handlers and controllers for managing
user interactions within the GUI. It handles various aspects of the application's
functionality across different tabs and components.

Key components include:
- FolderTabHandler: Manages folder selection and image list management
- CalibrationTabHandler: Handles calibration processes and pixel-to-mm conversion
- ImageProcessingTabHandler: Controls image processing operations and parameter adjustments
- ExportSettingsHandler: Manages export settings and file saving operations
- TomlFileHandler: Handles configuration file loading and validation

Each handler is designed to manage specific aspects of the user interface while
maintaining separation of concerns and providing clear interaction patterns between
the GUI and the underlying processing logic.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd  # Add this import
import toml as tomllib  # type: ignore
from numpy import typing as npt
from pydantic import ValidationError
from PySide6.QtCore import QEventLoop, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QTableWidgetItem,
    QTreeView,
    QVBoxLayout,
)

from bubble_analyser.core.models import CalibrationModel, InputFilesModel
from bubble_analyser.gui import (
    ImageProcessingModel,
    Step1Worker,
    Step2Worker,
    WorkerThread,
)
from bubble_analyser.gui.export_dialog import ExportFormatDialog
from bubble_analyser.gui.gui import MainWindow, ProcessingDialog
from bubble_analyser.processing import Config, cv2_to_qpixmap
from bubble_analyser.processing.dataset_export import FrameAnnotation, export_dataset


class WeightsDownloadThread(QThread):
    """Thread to download ML weights without blocking the GUI."""

    progress_changed = Signal(int)
    finished_download = Signal(bool, str)

    def run(self) -> None:
        """Execute the weight download process."""
        try:
            import requests

            try:
                import bubble_analyser.weights.loader as weights_loader

                url = getattr(weights_loader, "WEIGHTS_URL")
            except (ImportError, AttributeError):
                url = "https://github.com/ImperialCollegeLondon/bubble_analyser/releases/download/v0.3.0/mask_rcnn_bubble.h5"

            if getattr(sys, "frozen", False):
                base_dir = getattr(sys, "_MEIPASS", "")
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            weights_dir = os.path.join(base_dir, "bubble_analyser", "weights")
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = os.path.join(weights_dir, "mask_rcnn_bubble.h5")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(weights_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            self.progress_changed.emit(int((downloaded / total_size) * 100))

            self.finished_download.emit(True, "")
        except Exception as e:
            self.finished_download.emit(False, str(e))


class ExportSettingsHandler(QDialog):
    """A dialog for configuring export settings for processed images.

    This class provides a user interface for selecting and confirming the directory
    where processed images will be saved.

    Attributes:
        save_path (Path): The directory path where processed images will be saved.
        default_path_edit (QLineEdit): Text field displaying the current save path.
        confirm_button (QPushButton): Button to confirm the selected path.
    """

    def __init__(self, parent=None) -> None:  # type: ignore
        """Initialize the export settings dialog.

        Args:
            parent: The parent widget. Defaults to None.
            params (Config, optional): Configuration parameters containing the default save path. Defaults to None.
        """
        super().__init__(parent)

        self.setWindowTitle("Export Settings")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout(self)

        self.save_path: Path = cast(Path, "Select Export Path")
        self.if_save_path: bool = False

        # Default path for results saving
        self.default_path_edit = QLineEdit()
        self.default_path_edit.setPlaceholderText(str(self.save_path))  # type: ignore
        select_folder_button = QPushButton("Select Folder")
        select_folder_button.clicked.connect(self.select_folder)

        # Confirm folder selection
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.default_path_edit)
        path_layout.addWidget(select_folder_button)

        path_Vlayout = QVBoxLayout()
        path_Vlayout.addLayout(path_layout)
        path_Vlayout.addWidget(self.confirm_button)

        layout.addLayout(path_Vlayout)

    def accept(self) -> None:
        """Handle the confirmation of the selected save path.

        Validates the path before accepting the dialog. If the path is invalid,
        the dialog remains open.

        Returns:
            None
        """
        if self.check_if_path_valid() is False:
            return None
        super().accept()
        self.save_path = cast(Path, self.default_path_edit.text())
        logging.info(
            f"Export path for final graph, csv datafile, and processed\
                images (optional) set as: {self.save_path}"
        )
        self.if_save_path = True

    def select_folder(self) -> None:
        """Open a file dialog to select a folder for saving processed images.

        Updates the text field with the selected folder path.
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", "", QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )
        if folder_path:
            self.default_path_edit.setText(folder_path)

    def check_if_path_valid(self) -> bool:
        """Check if the selected path is valid and accessible.

        Returns:
            bool: True if the path is valid, False otherwise.
        """
        path_text = cast(Path, self.default_path_edit.text())

        # Check if the text field is empty
        if not path_text:
            QMessageBox.warning(self, "Error in Export Path setting: ", "No path selected")
            return False

        try:
            # Convert to Path object and resolve it
            path_obj = Path(path_text).resolve()

            # Check if the path actually exists
            if not path_obj.exists():
                QMessageBox.warning(self, "Error in Export Path setting: ", f"Path does not exist: {path_obj}")
                return False

            # Check if it's a directory (not a file)
            if not path_obj.is_dir():
                QMessageBox.warning(self, "Error in Export Path setting: ", f"Path is not a directory: {path_obj}")
                return False

            # Check if the directory is writable
            if not os.access(path_obj, os.W_OK):
                QMessageBox.warning(self, "Error in Export Path setting: ", f"Directory is not writable: {path_obj}")
                return False

            return True

        except (OSError, ValueError) as e:
            error_str = f"Invalid path: {e!s}"
            QMessageBox.warning(self, "Error in Export Path setting: ", error_str)
            return False


class TomlFileHandler:
    """A handler class for loading and validating TOML configuration files.

    This class is responsible for reading configuration parameters from a TOML file,
    validating them against the Config schema, and providing access to the parsed parameters.

    Attributes:
        file_path (Path): Path to the TOML configuration file.
        params (Config): Validated configuration parameters loaded from the file.
        gui: Reference to the GUI instance for displaying warnings.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the TOML file handler with the specified file path.

        Args:
            file_path (Path): Path to the TOML configuration file to load.
        """
        self.file_path: Path = file_path
        self.params: Config
        self.load_toml()

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance for displaying warnings.

        This method should be called after the GUI has been initialized to enable
        the handler to interact with GUI components and display warning messages.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def load_toml(self) -> None:
        """Load and validate the TOML configuration file.

        Attempts to parse the TOML file and validate its contents against the Config schema.
        If validation fails, an error message is displayed.
        """
        try:
            self.params = Config(**tomllib.load(self.file_path))
        except ValidationError as e:
            error_str = str(e)
            print(error_str)
            self._show_warning("Error in Config File Setting", error_str)

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)


class FolderTabHandler:
    """Handler for the folder selection tab in the GUI.

    This class manages the selection and confirmation of folders containing image files
    for processing, and handles the preview of selected images.

    Attributes:
        model (InputFilesModel): Model for managing input image files and paths.
        image_path (Path): Default path for raw images from configuration.
        gui: Reference to the main GUI instance.
    """

    def __init__(self, model: InputFilesModel, params: Config) -> None:
        """Initialize the folder tab handler with model and configuration parameters.

        Args:
            model (InputFilesModel): Model for managing input image files.
            params (Config): Configuration parameters containing default paths.
        """
        self.model: InputFilesModel = model
        self.image_path: Path = params.raw_img_path

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance.

        This method stores a reference to the main GUI instance and prepares the handler
        to interact with GUI components for displaying and exporting results.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def select_folder(self) -> None:
        """Handle the folder selection process.

        Opens a file dialog for the user to select a folder containing images.
        The dialog shows both folders and image files to help with selection.
        If a folder has already been confirmed, displays a warning instead.
        """
        if self.model.sample_images_confirmed:
            self._show_warning("Selection Locked", "You have already confirmed the folder selection.")
            return

        # Create a custom file dialog that shows both folders and image files
        file_dialog = QFileDialog(self.gui, "Select Folder Containing Images")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        file_dialog.setNameFilter(
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.PNG *.JPG *.JPEG *.BMP *.TIF *.TIFF)"
        )

        # Show files as well as folders
        file_view = file_dialog.findChild(QListView, "listView")
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Also set this for the tree view
        tree_view = file_dialog.findChild(QTreeView)
        if tree_view:
            tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Initialize folder_path to None
        folder_path = None

        # Execute the dialog
        if file_dialog.exec():
            folder_path = file_dialog.selectedFiles()[0]

        if folder_path:
            self._update_folder_path(folder_path)
            self._populate_image_list(folder_path)
            logging.info(f"Raw image path set as: {folder_path}")
        else:
            logging.info("User cancelled folder selection")

    def _update_folder_path(self, folder_path: str) -> None:
        """Update the model and GUI with the selected folder path.

        Args:
            folder_path (str): The path to the selected folder.
        """
        self.model.folder_path = cast(Path, folder_path)
        self.gui.folder_path_edit.setText(folder_path)

    def _populate_image_list(self, folder_path: str) -> None:
        """Populate the image list widget with images from the selected folder.

        Args:
            folder_path (str): The path to the folder containing images.
        """
        images, _ = self.model.get_image_list(folder_path)
        self.gui.image_list.clear()
        self.gui.image_list.addItems(images)

    def confirm_folder_selection(self) -> None:
        """Confirm the selected folder and proceed to the next tab.

        Updates the model with the selected folder path, populates the image list,
        and switches to the calibration tab if a folder has been selected.
        """
        if self.model.sample_images_confirmed:
            self._show_warning("Selection Locked", "You have already confirmed the folder selection.")
            return

        folder_path = self.gui.folder_path_edit.text()
        if folder_path:
            self._update_folder_path(folder_path)
            self._populate_image_list(folder_path)
            self.model.confirm_folder_selection(folder_path)
            self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.calibration_tab))
            logging.info("Raw image path confirmed and locked.")
            logging.info("******************************Calibration session******************************")

    def preview_image_folder_tab(self) -> None:
        """Display a preview of the selected image in the folder tab.

        Loads the selected image from the list and displays it in the preview area,
        maintaining the aspect ratio.
        """
        selected_image = self.gui.image_list.currentItem().text()
        folder_path = self.gui.folder_path_edit.text()
        image_path = folder_path + "/" + selected_image
        pixmap = QPixmap(image_path)

        self.gui.image_preview.set_preview_pixmap(pixmap)

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)


class CalibrationTabHandler:
    """Handler for the calibration tab in the GUI.

    This class manages the calibration process, including selection and processing of
    ruler images for pixel-to-millimeter conversion and optional background image correction.

    Attributes:
        calibration_model (CalibrationModel): Model for managing calibration data.
        img_resample (float): Resampling factor for image processing.
        px_img_path (Path): Default path for the ruler calibration image.
        gui: Reference to the main GUI instance.
    """

    def __init__(self, calibration_model: CalibrationModel, params: Config) -> None:
        """Initialize the calibration tab handler.

        Args:
            calibration_model (CalibrationModel): Model for managing calibration data.
            params (Config): Configuration parameters containing default values.
        """
        self.calibration_model: CalibrationModel = calibration_model
        self.px_img_path: Path = params.ruler_img_path

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance.

        This method stores a reference to the main GUI instance and prepares the handler
        to interact with GUI components for displaying and exporting results.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def select_ruler_button(self) -> None:
        """Handle the process of resolution calibration with ruler image selection and pixel-to-mm conversion."""
        # First select the image
        image_selected = self.select_px_mm_image()

        # Only proceed to calculate ratio if an image was actually selected
        if image_selected:
            self.get_px2mm_ratio()
        else:
            logging.info("Skipping px2mm ratio calculation as no image was selected")

    def select_px_mm_image(self) -> bool:
        """Handle the selection of a ruler image for pixel-to-millimeter calibration.

        Opens a file dialog for selecting a ruler image and updates the preview in the GUI.
        If calibration has already been confirmed, displays a warning instead.

        Returns:
            bool: True if an image was selected, False otherwise.
        """
        if self.calibration_model.calibration_confirmed:
            self._show_warning(
                "Selection Locked",
                "You have already confirmed the ruler image selection.",
            )
            return False

        # Use getOpenFileName with explicit options for better Windows compatibility
        image_path, _ = QFileDialog.getOpenFileName(
            self.gui,
            "Select Ruler Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.PNG *.JPG *.JPEG *.BMP *.TIF *.TIFF)",
            options=QFileDialog.Option.ReadOnly,
        )

        if image_path:
            self.gui.pixel_img_name.setText(image_path)
            pixmap = QPixmap(image_path)
            self.gui.pixel_img_preview.set_preview_pixmap(pixmap)
            return True
        else:
            logging.info("User cancelled ruler image selection")
            return False

    def get_px2mm_ratio(self) -> None:
        """Calculate the pixel-to-millimeter ratio from the selected ruler image.

        If calibration has already been confirmed or no image is selected, displays
        appropriate warning messages.
        """
        if self.calibration_model.calibration_confirmed:
            self._show_warning("Selection Locked", "You have already confirmed the pixel-to-mm ratio.")
            return

        img_path: Path = cast(Path, self.gui.pixel_img_name.text())
        if os.path.exists(img_path):
            px2mm, img_drawed_line = self.calibration_model.get_px2mm_ratio(pixel_img_path=img_path, gui=self.gui)

            if px2mm <= 0:
                self.gui.statusBar().showMessage("Calibration cancelled: no line was drawn.", 5000)
                return

            px2mm_display = px2mm

            self.gui.manual_px_mm_input.setText(f"{px2mm_display:.3f}")

            pixmap = cv2_to_qpixmap(img_drawed_line)
            self.gui.pixel_img_preview.set_preview_pixmap(pixmap)
            logging.info(f"Pixel to millimeter ratio detected as: {px2mm:.3f}")
        else:
            self.gui.statusBar().showMessage("Image file does not exist or not selected.", 5000)

    def select_bg_corr_image(self) -> None:
        """Handle the selection of a background correction image.

        Opens a file dialog for selecting a background image and updates the preview in the GUI.
        If calibration has already been confirmed, displays a warning instead.
        """
        if self.calibration_model.calibration_confirmed:
            self._show_warning(
                "Selection Locked",
                "You have already confirmed the background image selection.",
            )
            return

        # Use getOpenFileName with explicit options for better Windows compatibility
        image_path, _ = QFileDialog.getOpenFileName(
            self.gui,
            "Select Background Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.PNG *.JPG *.JPEG *.BMP *.TIF *.TIFF)",
            options=QFileDialog.Option.ReadOnly,
        )

        if image_path:
            self.calibration_model.bknd_img_path = Path(image_path)
            self.gui.bg_corr_image_name.setText(image_path)
            pixmap = QPixmap(image_path)
            self.gui.bg_corr_image_preview.set_preview_pixmap(pixmap)
            self.calibration_model.if_bknd = True
            logging.info(f"Background image path set as: {image_path}")
        else:
            logging.info("User cancelled background image selection")

    def confirm_calibration(self) -> None:
        """Confirm the calibration settings and proceed to the next tab.

        Updates the calibration model with the final values and switches to the
        image processing tab. If calibration has already been confirmed, displays
        a warning instead. The first image of the list will be previewed in the
        image processing tab.
        """
        if self.calibration_model.calibration_confirmed:
            self.gui.manual_px_mm_input.setText(f"{self.calibration_model.px2mm:.3f}")
            self._show_warning("Selection Locked", "You have already confirmed the calibration.")
            return

        self.calibration_model.bknd_img_path = Path(self.gui.bg_corr_image_name.text())
        self.calibration_model.px2mm_display = float(self.gui.manual_px_mm_input.text())
        # self.calibration_model.px2mm = self.calibration_model.px2mm_display * self.img_resample
        self.calibration_model.px2mm = self.calibration_model.px2mm_display

        self.calibration_model.confirm_calibration()

        self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.image_processing_tab))
        self.preview_image_intialize()
        logging.info(f"Pixel to millimeter ratio locked as: {self.calibration_model.px2mm_display:.3f}")
        # logging.info(f"Real pixel to millimeter ratio locked as: {self.calibration_model.px2mm_display:.3f}")
        logging.info(f"Background image path locked as: {self.calibration_model.bknd_img_path}")
        logging.info("Calibration confirmed and locked.")
        logging.info("******************************Parameter Adjustment Session******************************")

    def preview_image_intialize(self) -> None:
        """Display a preview of the first image in the image processing tab.

        Loads the selected image from the list and displays it in the preview area,
        maintaining the aspect ratio.
        """
        self.gui.image_list.setCurrentRow(0)
        selected_image = self.gui.image_list.currentItem().text()
        folder_path = self.gui.folder_path_edit.text()
        image_path = folder_path + "/" + selected_image
        pixmap = QPixmap(image_path)

        self.gui.sample_image_preview.set_preview_pixmap(pixmap)

        # Update the image info display for the initial image
        if hasattr(self.gui, "sample_image_info_label"):
            filename = selected_image
            current_index = 1  # First image
            total_images = self.gui.image_list.count()
            info_text = f"Image {current_index} of {total_images}: {filename}"
            self.gui.sample_image_info_label.setText(info_text)

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)


class ImageProcessingTabHandler(QThread):
    """Handler for the image processing tab in the GUI.

    This class manages the image processing operations, including algorithm selection,
    parameter configuration, and batch processing. It extends QThread to handle
    long-running processing tasks without blocking the GUI.

    Attributes:
        batch_processing_done (Signal): Signal emitted when batch processing completes.
        model (ImageProcessingModel): Model for managing image processing operations.
        params (Config): Configuration parameters for processing.
        current_index (int): Index of the currently selected image.
        algorithm_list (list[str]): List of available processing algorithms.
        export_handler (ExportSettingsHandler): Handler for export settings.
        if_save_processed_images (bool): Flag indicating whether to save processed images.
        temp_param_dict (dict[str, int | float]): Temporary storage for processing parameters.
        temp_filter_param_dict (dict[str, float]): Temporary storage for filter parameters.
        save_path (Path): Path where processed images will be saved.
    """

    batch_processing_done = Signal()
    check_for_export_path = Signal()

    def __init__(self, image_processing_model: ImageProcessingModel, params: Config) -> None:
        """Initialize the image processing tab handler.

        Args:
            image_processing_model (ImageProcessingModel): Model for managing image processing.
            params (Config): Configuration parameters for processing.
        """
        super().__init__()
        self.model: ImageProcessingModel = image_processing_model
        self.params: Config = params
        self.params_checker: Config = self.params.model_copy()
        self.current_index: int = 0
        self.algorithm_list: list[str] = []
        self.export_handler: ExportSettingsHandler
        self.if_save_processed_images = False

        self.save_path: Path = cast(Path, None)

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance.

        This method stores a reference to the main GUI instance and prepares the handler
        to interact with GUI components for displaying and exporting results.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui

    def pass_filter_params(self) -> None:
        """Pass filter parameters to the processing model.

        Args:
            filter_param_dict (dict[str, int | float]): Dictionary of filter parameters.
        """
        self.model.load_filter_params(self.filter_param_dict_1)

    def preview_image(self) -> None:
        """Display a preview of the currently selected image.

        Loads the selected image from the list and displays it in the preview area,
        maintaining the aspect ratio.
        """
        selected_image = self.gui.image_list.currentItem().text()
        folder_path = self.gui.folder_path_edit.text()
        image_path = folder_path + "/" + selected_image
        pixmap = QPixmap(image_path)

        self.gui.sample_image_preview.set_preview_pixmap(pixmap)

        # Update the filename and index display
        self.update_image_info_display()

    def check_params(self, name: str, value: int | float | str) -> bool:
        """Validate a parameter value against the configuration schema.

        A new Config instance is being created so the validation error can be refreshed every time.

        Args:
            name (str): The name of the parameter to validate.
            value (int | float): The value to validate.

        Returns:
            bool: True if the parameter is valid, False otherwise.
        """
        new_checker: Config = self.params_checker.model_copy()

        logging.debug(f"Checking parameter: {name} {value}")
        if name == "element_size":
            try:
                new_checker.element_size = cast(int, value)
            except ValidationError as e:
                self._show_warning("Invalid Element Size", str(e))
                return False

        if name == "connectivity":
            try:
                new_checker.connectivity = cast(int, value)
            except ValidationError as e:
                self._show_warning("Invalid Connectivity", str(e))
                return False

        # if name == "if_gaussianblur":
        #     try:
        #         new_checker.if_gaussianblur = cast(str, value)
        #     except ValidationError as e:
        #         self._show_warning("Invalid if_gaussianblur value", str(e))
        #         return False

        if name == "ksize":
            try:
                new_checker.ksize = cast(int, value)
            except ValidationError as e:
                self._show_warning("Invalid ksize value", str(e))
                return False

        if name == "target_width":
            try:
                new_checker.target_width = cast(int, value)
            except ValidationError as e:
                self._show_warning("Invalid Resample Factor", str(e))
                return False

        if name == "max_thresh":
            try:
                new_checker.max_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Threshold value", str(e))
                return False

        if name == "min_thresh":
            try:
                new_checker.min_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Threshold value", str(e))
                return False

        if name == "step_size":
            try:
                new_checker.step_size = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Step Size", str(e))
                return False

        if name == "high_thresh":
            try:
                new_checker.high_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Threshold value", str(e))
                return False

        if name == "mid_thresh":
            try:
                new_checker.mid_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Threshold value", str(e))
                return False

        if name == "low_thresh":
            try:
                new_checker.low_thresh = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Thresh value", str(e))
                return False

        if name == "max_eccentricity":
            try:
                new_checker.max_eccentricity = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Max Eccentricity", str(e))
                return False

        if name == "min_solidity":
            try:
                new_checker.min_solidity = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Solidity", str(e))
                return False

        if name == "min_circularity":
            try:
                new_checker.min_circularity = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Circularity", str(e))
                return False

        if name == "min_size":
            try:
                new_checker.min_size = cast(float, value)
            except ValidationError as e:
                self._show_warning("Invalid Min Size", str(e))
                return False

        return True

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)

    def update_sample_image(self, direction: str) -> None:
        """Navigate between images in the image list.

        Args:
            direction (str): The direction to navigate, either "prev" or "next".
        """
        current_row = self.gui.image_list.currentRow()

        if direction == "prev":
            if current_row > 0:
                current_row -= 1
                self.gui.image_list.setCurrentRow(current_row)
        elif direction == "next":
            if current_row < self.gui.image_list.count() - 1:
                current_row += 1
                self.gui.image_list.setCurrentRow(current_row)

        self.current_index = current_row
        logging.info(f"(event_handlers/update_sample_img)Current image index: {self.current_index}")

        self.preview_image()
        self.update_preview_procsd_img_button()

    def update_image_info_display(self) -> None:
        """Update the display of filename and index information under the sample image preview.

        Shows the current image filename and its index in the format:
        "Image X of Y: filename.ext"
        """
        if self.gui.image_list.count() == 0:
            self.gui.sample_image_info_label.setText("No images loaded")
            return

        current_item = self.gui.image_list.currentItem()
        if current_item is None:  # type: ignore[unreachable]
            self.gui.sample_image_info_label.setText("No image selected")
            return

        filename = current_item.text()
        current_index = self.gui.image_list.currentRow() + 1  # 1-based index for display
        total_images = self.gui.image_list.count()

        info_text = f"Image {current_index} of {total_images}: {filename}"
        self.gui.sample_image_info_label.setText(info_text)

    def update_preview_procsd_img_button(self) -> None:
        """Check if the current image are being processed with segmentation and filtering.

        If the current image has been processed, the preview processed image button is enabled.
        """
        if_img, _img_before_filter, _img_after_filter = self.model.preview_processed_image(self.current_index)
        if if_img:
            logging.info("Preview processed image enabled for current image.")
            self.gui.preview_processed_images_button.setEnabled(True)

            # Below is only for cnn version
            self.preview_processed_images()

        else:
            self.gui.preview_processed_images_button.setEnabled(False)

    def preview_processed_images(self) -> None:
        """Preview the processed images for the current image.

        Retrieves the processed images from the model and displays them in the preview areas.
        If the image hasn't been processed yet, displays a warning.
        """
        if_img, img_before_filter, img_after_filter = self.model.preview_processed_image(self.current_index)

        if if_img:
            self.update_label_before_filtering(img_before_filter)
            self.update_process_image_preview(img_after_filter)

        else:
            self._show_warning("Image Not Found", "Image has not been fully processed yet.")

    # -------Second Column Functions-------------------------
    def initialize_algorithm_combo(self) -> None:
        """Initialize the algorithm combo box."""
        logging.info("Initializing algorithm combo box...")
        self.gui.algorithm_combo.clear()

        for algorithm, params in self.model.all_methods_n_params.items():
            # Create a lower-case version of the name for checking
            algo_name = algorithm.lower()

            # Skip the "NEW method" as requested by the user
            if algo_name == "new method":
                continue

            logging.info(f"Initialize algorithm: {algorithm}")
            self.algorithm_list.append(algorithm)

        self.gui.algorithm_combo.addItems(self.algorithm_list)

        # Handle case where list might be empty (unlikely given non-ML methods exist)
        if self.algorithm_list:
            self.previous_algorithm = self.algorithm_list[0]
            self.update_model_algorithm(self.algorithm_list[0])

            # Update description safely
            try:
                description = getattr(
                    self.model.methods_handler.all_classes[self.algorithm_list[0]],
                    "description",
                    "No description available.",
                )
                self.gui.algorithm_description_label.setText(description)
                self.gui.algorithm_description_label.setToolTip(description)
                self.gui.algorithm_combo.setToolTip("Select the segmentation algorithm to use for bubble detection.")
            except Exception as e:
                logging.error(f"Error updating description: {e}")
                self.gui.algorithm_description_label.setText("No description available.")

        logging.info("Algorithm combo box initialized.")

    def load_parameter_table_1(self, algorithm: str) -> None:
        """Load the parameter table with values for the selected algorithm.

        Args:
            algorithm (str): The name of the algorithm whose parameters should be loaded.
        """
        # pass the current algorithm text in the gui to the model
        # This function only triggered by first initialization and algorithm change

        self.current_algorithm = algorithm
        logging.info(f"Current choosing algorithm: {self.current_algorithm}")

        self.gui.param_sandbox1.clear()

        # Retrieve descriptions dynamically from the algorithm instance
        instance = self.model.methods_handler.all_classes.get(self.current_algorithm)

        if self.current_algorithm == "BubMask (Deep Learning)":
            self._ensure_bubmask_params_only()
        else:
            # Fallback: imported projects often store no parameters for the
            # non-BubMask methods (YOLO Segmentation, Iterative Watershed,
            # Peak-Based, ...), which left their parameter table empty. Seed it
            # from the method's own defaults so its attributes are always visible
            # and editable.
            stored = self.model.all_methods_n_params.get(self.current_algorithm)
            if not stored and instance is not None and hasattr(instance, "get_needed_params"):
                self.model.all_methods_n_params[self.current_algorithm] = dict(instance.get_needed_params())

        # Retrieve descriptions dynamically from the algorithm instance
        algo_descriptions = {}
        if instance and hasattr(instance, "get_param_descriptions"):
            algo_descriptions = instance.get_param_descriptions()

        for algorithm_name, params in self.model.all_methods_n_params.items():
            if algorithm_name == self.current_algorithm:
                self.gui.param_sandbox1.setRowCount(len(params))

                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    # Create name item and set tooltip
                    name_item = QTableWidgetItem(name)
                    description = algo_descriptions.get(name, "")
                    if description:
                        name_item.setToolTip(description)
                    self.gui.param_sandbox1.setItem(row, 0, name_item)

                    # Special handling for if_gaussianblur parameter
                    if name == "if_gaussianblur":
                        # Create a dropdown with True/False options
                        combo_box = QComboBox()
                        combo_box.addItems(["True", "False"])
                        if description:
                            combo_box.setToolTip(description)

                        # Set the current value
                        current_value = str(value)
                        if current_value in ["True", "False"]:
                            combo_box.setCurrentText(str(current_value))
                        else:
                            # Default to False if value is not boolean
                            combo_box.setCurrentText("False")

                        # Set the combo box as the cell widget
                        self.gui.param_sandbox1.setCellWidget(row, 1, combo_box)

                    elif name == "element_size":
                        combo_box = QComboBox()
                        combo_box.addItems(["0", "3", "5"])
                        if description:
                            combo_box.setToolTip(description)

                        # Set the current value
                        current_value = str(value)
                        if current_value in ["0", "3", "5"]:
                            combo_box.setCurrentText(current_value)
                        else:
                            # Default to False if value is not boolean
                            combo_box.setCurrentText("3")

                        # Set the combo box as the cell widget
                        self.gui.param_sandbox1.setCellWidget(row, 1, combo_box)

                    elif name == "connectivity":
                        combo_box = QComboBox()
                        combo_box.addItems(["4", "8"])
                        if description:
                            combo_box.setToolTip(description)

                        # Set the current value
                        current_value = str(value)
                        if current_value in ["4", "8"]:
                            combo_box.setCurrentText(current_value)
                        else:
                            # Default to False if value is not boolean
                            combo_box.setCurrentText("8")

                        # Set the combo box as the cell widget
                        self.gui.param_sandbox1.setCellWidget(row, 1, combo_box)

                    elif name == "ksize":
                        combo_box = QComboBox()
                        combo_box.addItems(["1", "3", "5", "7"])
                        if description:
                            combo_box.setToolTip(description)

                        # Set the current value
                        current_value = str(value)
                        if current_value in ["1", "3", "5", "7"]:
                            combo_box.setCurrentText(current_value)
                        else:
                            # Default to False if value is not boolean
                            combo_box.setCurrentText("3")

                        # Set the combo box as the cell widget
                        self.gui.param_sandbox1.setCellWidget(row, 1, combo_box)

                    elif name == "device" and self.current_algorithm == "YOLO Segmentation":
                        combo_box = QComboBox()
                        combo_box.addItems(["CPU", "GPU"])
                        combo_box.setToolTip(
                            "CPU: runs on the processor and is always compatible.\n"
                            "GPU: uses NVIDIA CUDA through PyTorch when available."
                        )

                        current_value = str(value).strip().lower()
                        if current_value in ["gpu", "cuda", "0"]:
                            combo_box.setCurrentText("GPU")
                        else:
                            combo_box.setCurrentText("CPU")

                        self.gui.param_sandbox1.setCellWidget(row, 1, combo_box)

                    else:
                        # Regular text item for other parameters
                        value_item = QTableWidgetItem(str(value))
                        if description:
                            value_item.setToolTip(description)
                        self.gui.param_sandbox1.setItem(row, 1, value_item)

                break

    def _ensure_bubmask_params_only(self) -> None:
        """Keep only the parameters that BubMask actually uses.

        Some imported projects can carry watershed parameters in the BubMask
        parameter dictionary. Those extra values confuse the GUI and can be
        saved back into the project. This guard makes the BubMask table explicit
        and stable.
        """
        algorithm = "BubMask (Deep Learning)"
        current = self.model.all_methods_n_params.get(algorithm, {})
        instance = self.model.methods_handler.all_classes.get(algorithm)
        default_params = instance.get_needed_params() if instance and hasattr(instance, "get_needed_params") else {}

        wanted_order = [
            "weights_path",
            "confidence_threshold",
            "resample",
            "image_min_dim",
            "image_max_dim",
            "max_bubbles",
            "nms_threshold",
        ]
        cleaned = {}
        for name in wanted_order:
            if name in current:
                cleaned[name] = current[name]
            elif name in default_params:
                cleaned[name] = default_params[name]

        # Fallbacks mirror the BubMaskWatershed defaults so the two never drift apart.
        cleaned["confidence_threshold"] = float(cleaned.get("confidence_threshold", 0.9))
        cleaned["resample"] = float(cleaned.get("resample", 1.0))
        cleaned["image_min_dim"] = int(float(cleaned.get("image_min_dim", 512)))
        cleaned["image_max_dim"] = int(float(cleaned.get("image_max_dim", 1024)))
        cleaned["max_bubbles"] = int(float(cleaned.get("max_bubbles", 500)))
        cleaned["nms_threshold"] = float(cleaned.get("nms_threshold", 0.3))

        self.model.all_methods_n_params[algorithm] = cleaned
        self.model.methods_handler.full_dict[algorithm] = cleaned

    def handle_algorithm_change(self, new_algorithm: str) -> None:
        """Handle a change in the selected algorithm.

        Updates the parameters in the model, reloads the parameter table,
        and updates the model with the new algorithm.

        Args:
            new_algorithm (str): The name of the newly selected algorithm.
        """
        algo_name = new_algorithm.lower()
        is_ml_method = (
            "cnn" in algo_name or "rcnn" in algo_name or "bubmask" in algo_name or "deep learning" in algo_name
        )

        auto_run_after = False

        if is_ml_method:
            from bubble_analyser.weights.loader import get_weights_path

            weights_path, _ = get_weights_path(download_if_missing=False)

            if weights_path is None:
                reply = QMessageBox.question(
                    self.gui,
                    "Download Required",
                    (
                        f"The '{new_algorithm}' method requires a deep learning weights "
                        f"file (mask_rcnn_bubble.h5) which was not found locally.\n\n"
                        f"Do you want to download it now? (This may take a few minutes)."
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    progress = QProgressDialog(
                        "Downloading weights file...\nThis may take a few minutes.",
                        "Cancel",
                        0,
                        100,
                        self.gui,
                    )
                    progress.setWindowTitle("Downloading...")
                    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
                    progress.setMinimumDuration(0)
                    progress.setAutoClose(False)
                    progress.setAutoReset(False)
                    progress.setValue(0)
                    progress.show()

                    loop = QEventLoop()
                    self._download_worker = WeightsDownloadThread()
                    download_success = [False]
                    download_completed = [False]

                    def on_finished(success: bool, msg: str) -> None:
                        download_completed[0] = True
                        download_success[0] = success
                        if not success:
                            self._show_warning("Download Failed", f"Failed to download weights: {msg}")
                        loop.quit()

                    def on_cancel() -> None:
                        if download_completed[0]:
                            return
                        self._download_worker.terminate()
                        self._download_worker.wait()
                        download_success[0] = False
                        loop.quit()

                    self._download_worker.progress_changed.connect(progress.setValue)
                    self._download_worker.finished_download.connect(on_finished)
                    progress.canceled.connect(on_cancel)
                    self._download_worker.start()
                    loop.exec()
                    progress.close()

                    if not download_success[0]:
                        self.gui.algorithm_combo.blockSignals(True)
                        self.gui.algorithm_combo.setCurrentText(
                            getattr(self, "previous_algorithm", self.algorithm_list[0])
                        )
                        self.gui.algorithm_combo.blockSignals(False)
                        return

                    auto_run_after = True
                else:
                    # User declined to download, revert to previous selection
                    self.gui.algorithm_combo.blockSignals(True)
                    self.gui.algorithm_combo.setCurrentText(getattr(self, "previous_algorithm", self.algorithm_list[0]))
                    self.gui.algorithm_combo.blockSignals(False)
                    return

        self.previous_algorithm = new_algorithm

        # Update the algorithm in the model
        self.update_segment_parameters()

        # Reload the param table in the GUI
        self.load_parameter_table_1(new_algorithm)

        # Update params in the model
        self.update_model_algorithm(new_algorithm)

        # Update description
        try:
            description = getattr(
                self.model.methods_handler.all_classes[new_algorithm], "description", "No description available."
            )
            self.gui.algorithm_description_label.setText(description)
        except Exception as e:
            logging.error(f"Error updating description: {e}")
            self.gui.algorithm_description_label.setText("No description available.")

        if auto_run_after:
            self.confirm_parameter_before_filtering()

    def update_model_algorithm(self, algorithm: str) -> None:
        """Update the algorithm in the processing model.

        Args:
            algorithm (str): The name of the algorithm to set in the model.
        """
        self.model.algorithm = algorithm

    def confirm_parameter_before_filtering(self) -> None:
        """Confirm the parameters for the first step of image processing.

        Validates all parameters against the configuration schema before
        proceeding with the first processing step.
        """
        logging.info("-----------------------------------------------------------------------------------------")
        logging.info("-------------------------------Running Step 1: Segmentation------------------------------")
        logging.info("-----------------------------------------------------------------------------------------")
        self.update_segment_parameters()
        if not self.validate_selected_segmentation_method():
            return

        # Validate the parameters
        logging.info("------------------------------Validating Segment Parameters------------------------------")
        # Update the model parameters
        for algorithm_name, params in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    if_valid = self.check_params(name, value)
                    if not if_valid:
                        return

        # self.pass_segment_params(self.model.segment_param_dict)
        self._process_step_1()

    def convert_value(self, text: str) -> int | float | str:
        """Convert a string to the appropriate numeric type.

        Args:
            text (str): The string to convert.

        Returns:
            int | float | str: The converted value, as an integer if the value is an integer,
                               as a float if the value is a floating-point number,
                               or as a string if the value cannot be converted to a number.
        """
        try:
            value = float(text)
            # Return an int if the number is integer
            if value.is_integer():
                logging.debug(f"{value} determined as integer.")
                return int(value)
            else:
                logging.debug(f"{value} determined as float.")
                return value
            # return int(value) if value.is_integer() else value
        except ValueError:
            # Fallback if not a number
            return text

    def extract_parameters_from_table(self, table_widget) -> dict[str, int | float]:  # type: ignore
        """Extract parameters from a table widget.

        Args:
            table_widget: The table widget containing parameter names and values.

        Returns:
            dict[str, int|float]: A dictionary mapping parameter names to their values.
        """
        params = {}
        row_count = table_widget.rowCount()
        for row in range(row_count):
            name_item = table_widget.item(row, 0)
            value_item = table_widget.item(row, 1)
            if name_item and value_item:
                params[name_item.text()] = self.convert_value(value_item.text())
            elif name_item:
                # Parameters edited through a dropdown (e.g. YOLO "device") live in a cell widget, not an item.
                cell_widget = table_widget.cellWidget(row, 1)
                if isinstance(cell_widget, QComboBox):
                    params[name_item.text()] = self.convert_value(cell_widget.currentText())
        return params  # type: ignore

    def update_segment_parameters(self) -> bool:
        """Update the segment parameters in the model from the GUI table.

        Extracts parameters from the table widget and updates the corresponding
        parameters in the model for the currently selected algorithm.

        Returns:
            bool: True if the parameters were successfully updated.
        """
        # Update the params in the model
        # Extract parameters from the table
        logging.info("------------------------------Updating Parameters------------------------------")
        params = self.extract_parameters_from_table(self.gui.param_sandbox1)
        # Update the model's dictionary for the selected algorithm
        for algorithm_name, params_in_dict in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                if algorithm_name == "BubMask (Deep Learning)":
                    allowed = {
                        "weights_path",
                        "confidence_threshold",
                        "resample",
                        "image_min_dim",
                        "image_max_dim",
                        "max_bubbles",
                        "nms_threshold",
                    }
                    params = {name: value for name, value in params.items() if name in allowed}
                    params_in_dict.clear()
                for name, value in params.items():
                    logging.debug(f"Updating {name} to {value}")
                    params_in_dict[name] = value

        if self.model.algorithm == "BubMask (Deep Learning)":
            self._ensure_bubmask_params_only()

        return True

    def validate_selected_segmentation_method(self) -> bool:
        """Validate requirements that are specific to the selected segmentation method."""
        if self.model.algorithm != "YOLO Segmentation":
            return True

        params = self.model.all_methods_n_params.get(self.model.algorithm, {})
        weights_path = Path(str(params.get("weights_path", ""))).expanduser()
        selected_device = str(params.get("device", "CPU")).strip().lower()
        missing_items: list[str] = []

        if not weights_path.is_file():
            missing_items.append(
                "YOLO weights were not found. Set 'weights_path' to a valid YOLO segmentation .pt file.\n"
                f"Current path:\n{weights_path}"
            )
        elif weights_path.suffix.lower() != ".pt":
            missing_items.append(
                f"The selected YOLO weights file should be a .pt segmentation model.\nCurrent file:\n{weights_path}"
            )

        if importlib.util.find_spec("ultralytics") is None:
            missing_items.append(
                "The Python package 'ultralytics' is not installed in the Bubble Analyzer environment."
            )

        if selected_device in ["gpu", "cuda", "0"]:
            from bubble_analyser.methods.yolo_method import describe_cuda_problem

            cuda_problem = describe_cuda_problem()
            if cuda_problem is not None:
                params["device"] = "CPU"  # type: ignore[assignment]  # YOLO params also hold strings
                self.set_yolo_device_combo_text("CPU")
                logging.warning(f"GPU unavailable for YOLO: {cuda_problem}")
                QMessageBox.warning(
                    self.gui,
                    "GPU Not Available",
                    f"GPU was selected, but it cannot be used:\n\n{cuda_problem}\n\n"
                    "Bubble Analyzer will continue with CPU for this run.\n\n"
                    "CPU is slower, but it is the safest compatible option.",
                )

        if not missing_items:
            return True

        QMessageBox.warning(
            self.gui,
            "YOLO Segmentation Not Ready",
            "YOLO Segmentation needs a trained YOLO segmentation model before it can run.\n\n"
            + "\n\n".join(missing_items)
            + "\n\nUse BubMask for now, or install Ultralytics and set weights_path to your trained YOLO .pt model.",
        )
        return False

    def set_yolo_device_combo_text(self, text: str) -> None:
        """Update the YOLO device combo box in the parameter table, when visible."""
        row_count = self.gui.param_sandbox1.rowCount()
        for row in range(row_count):
            name_item = self.gui.param_sandbox1.item(row, 0)
            if name_item is None or name_item.text() != "device":
                continue
            cell_widget = self.gui.param_sandbox1.cellWidget(row, 1)
            if isinstance(cell_widget, QComboBox):
                cell_widget.setCurrentText(text)
            return

    def _process_step_1(self) -> None:
        """Execute the first step of image processing.

        Calls the model's step_1_main method to process the current image
        and updates the preview with the results.
        """
        logging.info("------------------------------Processing Started------------------------------")

        # Determine the correct message for the dialog
        msg = "Processing Step 1..."
        if "Deep Learning" in self.model.algorithm:
            instance = self.model.methods_handler.all_classes.get(self.model.algorithm)
            # Check if it has a detector attribute that is not yet loaded
            if instance and hasattr(instance, "detector"):
                detector = getattr(instance, "detector")
                if detector is None or (hasattr(detector, "is_loaded") and not detector.is_loaded):
                    msg = "Loading Machine Learning engine...\n(First run: ~20-60 seconds)"

        # Create and show processing dialog
        self.processing_dialog = ProcessingDialog(self.gui, msg)
        self.processing_dialog.show()

        # Force the UI to paint the dialog before starting the heavy work
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        # Create and start worker thread
        self.step_1_worker = Step1Worker(self.model, self.current_index)
        self.step_1_worker.finished.connect(self.on_step_1_finished)
        self.step_1_worker.error.connect(self.on_worker_error)
        self.step_1_worker.start()

    def on_step_1_finished(self, img: npt.NDArray[np.int_]) -> None:
        """Handle completion of step 1 processing.

        Args:
            img (npt.NDArray[np.int_]): The processed image.
        """
        # Close dialogs
        if hasattr(self, "processing_dialog"):
            self.processing_dialog.close()

        self.update_label_before_filtering(img)
        logging.info("Step 1 processing completed.")

    def update_label_before_filtering(self, img: npt.NDArray[np.int_]) -> None:
        """Update the preview of the image after the first processing step.

        Args:
            img (npt.NDArray[np.int_]): The processed image to display.
        """
        self.gui.label_before_filtering.axes.clear()
        if img is not None and img.size > 0:
            self.gui.label_before_filtering.axes.imshow(img)
        self.gui.label_before_filtering.draw()

    # -------Third Column Functions: Filtering-------------------------
    def initialize_parameter_table_2(self) -> None:
        """Initialize the filtering parameters table with default values.

        Populates the filtering parameters table with the current values from
        the temporary filter parameter dictionary. Size parameters are shown first
        with units.
        """
        logging.info("Initializing parameter table 2...")
        self.filter_param_dict_1 = self.model.filter_param_dict_1

        # Retrieve descriptions dynamically from the filter param handler
        filter_descriptions = {}
        if hasattr(self.model.filter_param_handler, "get_param_descriptions"):
            filter_descriptions = self.model.filter_param_handler.get_param_descriptions()

        # Define priority order for display
        priority_params = ["min_size", "max_size"]
        other_params = [k for k in self.filter_param_dict_1.keys() if k not in priority_params]
        ordered_params = priority_params + other_params

        self.gui.param_sandbox2.setRowCount(len(ordered_params))

        for row, property in enumerate(ordered_params):
            value = self.filter_param_dict_1[property]
            display_name = property

            # Add unit labels for size parameters
            if property in ["min_size", "max_size"]:
                display_name = f"{property} (mm)"

            logging.info(f"Filter param name: {property}, value: {value}")

            # Create table items and set tooltips
            name_item = QTableWidgetItem(display_name)
            value_item = QTableWidgetItem(str(value))

            # Apply tooltip from decentralized descriptions
            description = filter_descriptions.get(property, "")
            if description:
                name_item.setToolTip(description)
                value_item.setToolTip(description)

            self.gui.param_sandbox2.setItem(row, 0, name_item)
            self.gui.param_sandbox2.setItem(row, 1, value_item)

        logging.info("Parameter table 2 initialized.")

    def confirm_parameter_for_filtering(self) -> None:
        """Confirm the filtering parameters and apply them to the current image.

        Validates all filtering parameters against the configuration schema before
        proceeding with the second processing step.
        """
        logging.info("-----------------------------------------------------------------------------------------")
        logging.info("--------------------------------Running Step 2: Filtering--------------------------------")
        logging.info("-----------------------------------------------------------------------------------------")

        logging.info("------------------------------Updating Filtering Parameters------------------------------")
        self.store_filter_params()

        logging.info("------------------------------Validating Filter Parameters------------------------------")
        for name, value in self.filter_param_dict_1.items():
            if_valid = self.check_params(name, value)
            if not if_valid:
                return

        self.pass_filter_params()
        self._process_step_2()

    def store_filter_params(self) -> None:
        """Store the filtering parameters from the GUI table into the filter parameter dictionary.

        Updates the filter parameter dictionary with the new values from the table widget.
        """
        logging.info("Storing filter parameters...")

        for row in range(self.gui.param_sandbox2.rowCount()):
            name_item = self.gui.param_sandbox2.item(row, 0)
            value_item = self.gui.param_sandbox2.item(row, 1)
            if name_item and value_item:
                # Strip units from label for internal mapping (e.g., 'min_size (mm)' -> 'min_size')
                param_name = name_item.text().split(" (")[0].strip()
                param_value = value_item.text()

                # Handle numeric parameter
                float_value = float(param_value)
                self.filter_param_dict_1[param_name] = float_value

                logging.info(f"Updating {param_name} to {param_value}")

    def _process_step_2(self) -> None:
        """Execute the second step of image processing (filtering).

        Calls the model's step_2_main method to apply filtering to the current image
        and updates the preview with the results.
        """
        # Create and show processing dialog
        self.processing_dialog = ProcessingDialog(self.gui, "Processing Step 2...")
        self.processing_dialog.show()

        # Create and start worker thread
        self.step_2_worker = Step2Worker(self.model, self.current_index)
        self.step_2_worker.finished.connect(self.on_step_2_finished)
        self.step_2_worker.error.connect(self.on_worker_error)
        self.step_2_worker.start()

    def on_step_2_finished(self, img: npt.NDArray[np.int_]) -> None:
        """Handle completion of step 2 processing.

        Args:
            img (npt.NDArray[np.int_]): The processed image.
        """
        # Close dialog
        if hasattr(self, "processing_dialog"):
            self.processing_dialog.close()

        self.update_process_image_preview(img)
        logging.info("Step 2 processing completed.")

    # -------Third Column Functions: Manual Ellipse Adjustment-------------------------
    def ellipse_manual_adjustment(self) -> None:
        """Launch the ellipse adjustment tool for manual fine-tuning of detected ellipses.

        Opens a separate window with tools for manually adjusting the detected ellipses,
        and updates the preview with the adjusted ellipses when complete.
        """
        from PySide6.QtCore import QEventLoop

        from bubble_analyser.processing import EllipseAdjuster

        logging.info("Ellipse handler triggered.")
        name = self.model.img_path_list[self.current_index]
        image = self.model.img_dict[name]
        self.model.adjuster = EllipseAdjuster(image.ellipses, image.img_rgb)

        loop = QEventLoop()

        def on_finished() -> None:
            self.model.handle_ellipse_adjustment_finished(image)
            self.update_process_image_preview(image.ellipses_on_images)
            loop.quit()

        self.model.adjuster.finished.connect(on_finished)
        self.model.adjuster.show()

        loop.exec()
        logging.info("Ellipse handler finished.")

    def update_process_image_preview(self, img: npt.NDArray[np.int_]) -> None:
        """Update the preview of the processed image after filtering or adjustment.

        Args:
            img (npt.NDArray[np.int_]): The processed image to display.
        """
        self.gui.processed_image_preview.axes.clear()
        if img is not None and img.size > 0:
            self.gui.processed_image_preview.axes.imshow(img)
        self.gui.processed_image_preview.draw()

    # -------Third Column Functions: Batch Processing----------------------------
    def ask_if_batch(self) -> None:
        """Function to handle the batch processing of all images in the folder.

        Creates and displays a confirmation dialog with options for saving processed images.
        If confirmed, initiates the batch processing operation.
        """
        if self.model.if_batched:
            warn_dialog = QMessageBox(self.gui)
            warn_dialog.setWindowTitle("Warning")
            warn_dialog.setText(
                "You have already run batch processing. Re-running will discard all past segmentations, "
                "including manual adjustments. Are you sure you want to continue?"
            )
            warn_dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            warn_dialog.setDefaultButton(QMessageBox.StandardButton.No)
            if warn_dialog.exec() != QMessageBox.StandardButton.Yes:
                logging.info("Batch re-processing canceled by user.")
                return

            # Wipe all caches (including fine-tuned ones) so everything re-runs
            self.model.reset_batch_state(force_all=True)

        # Simplified: no more save-images checkbox here.
        # All saving is now done via the Results tab at the end of the workflow.
        confirm_dialog = self.create_confirm_dialog(
            "Batch Processing Confirmation", "The parameters will be applied to all the images. Confirm to process."
        )

        response = confirm_dialog.exec()

        if response == QMessageBox.StandardButton.Ok:
            self.batch_process_images()  # Direct call, no more redundant checks
        else:
            logging.info("Batch processing canceled.")

    def finalise_analysis(self) -> None:
        """Finalise the analysis and transition to the results visualization."""
        if not self.model.if_batched:
            QMessageBox.information(
                self.gui, "Information", "Please finish at least one time of batch processing first."
            )
            return

        confirm_dialog = self.create_confirm_dialog(
            "Finalise Analysis Confirmation",
            "The analysis will be finalised. \n "
            "Make sure you are satisfied with the segmentation of all images. Confirm to finalise.",
        )
        self.create_save_images_checkbox(confirm_dialog)

        response = confirm_dialog.exec()

        if response == QMessageBox.StandardButton.Ok:
            self.model.if_finalise_analysis = True
            self.model.refresh_ellipses_properties()

            if self.if_save_processed_images:
                save_dir = None
                if hasattr(self, "export_handler") and self.export_handler and self.export_handler.if_save_path:
                    save_dir = str(self.export_handler.save_path)
                if not save_dir or not os.path.exists(save_dir):
                    save_dir = QFileDialog.getExistingDirectory(
                        self.gui, "Select Folder to Save Annotated Images"
                    )
                if save_dir:
                    self.save_annotated_images_to_directory(save_dir)

            # Transition directly to results
            self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.results_tab))
            self.batch_processing_done.emit()  # Trigger histogram generation
            logging.info("Analysis finalised. Switched to results tab.")
        else:
            logging.info("Analysis finalisation canceled.")

    def save_annotated_images_to_directory(self, folder_path: str) -> tuple[int, int]:
        """Save annotated images with ellipses overlaid to the specified directory.

        Args:
            folder_path: Directory path where images should be saved.

        Returns:
            tuple[int, int]: (success_count, error_count)
        """
        success_count = 0
        error_count = 0
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        for name in self.model.img_path_list:
            annotated_img = self.model.get_or_render_annotated_image(name)
            if annotated_img is not None and annotated_img.size > 0:
                base_name = os.path.splitext(os.path.basename(name))[0]
                export_path = os.path.join(folder_path, f"{base_name}_annotated.png")
                try:
                    img_uint8 = np.asarray(annotated_img, dtype=np.uint8)
                    if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
                        bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    else:
                        bgr = img_uint8
                    success, enc = cv2.imencode(".png", bgr)
                    if success:
                        enc.tofile(export_path)
                        success_count += 1
                    else:
                        if cv2.imwrite(export_path, bgr):
                            success_count += 1
                        else:
                            error_count += 1
                except Exception as e:
                    logging.error(f"Failed to save annotated image {name}: {e}")
                    error_count += 1
            else:
                error_count += 1

        logging.info(f"Saved {success_count} annotated images to {folder_path} (Errors: {error_count})")
        return success_count, error_count

    def create_confirm_dialog(self, title: str, text: str) -> QMessageBox:
        """Create a confirmation dialog for batch processing.

        Returns:
            QMessageBox: The configured confirmation dialog.
        """
        confirm_dialog = QMessageBox(self.gui)
        confirm_dialog.setWindowTitle(title)
        confirm_dialog.setText(text)
        confirm_dialog.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        return confirm_dialog

    def create_save_images_checkbox(self, dialog: QMessageBox) -> QCheckBox:
        """Create a checkbox for enabling image saving during batch processing.

        Args:
            dialog (QMessageBox): The dialog to add the checkbox to.

        Returns:
            QCheckBox: The configured checkbox widget.
        """
        save_images_checkbox = QCheckBox("Save processed images with ellipses")
        save_images_checkbox.setChecked(self.if_save_processed_images)
        save_images_checkbox.stateChanged.connect(
            lambda: self.update_if_save_processed_images(save_images_checkbox.isChecked())
        )
        dialog.setCheckBox(save_images_checkbox)
        return save_images_checkbox

    def update_if_save_processed_images(self, state: bool) -> None:
        """Update the flag indicating whether to save processed images.

        Args:
            state (bool): The new state of the save images flag.
        """
        self.if_save_processed_images = state

    def batch_process_images(self) -> None:
        """Execute batch processing on all images in the folder.

        Validates all parameters, initializes the progress window, and starts
        the worker thread to process all images.
        """
        logging.info("-----------------------------------------------------------------------------------------")
        logging.info("--------------------------------Running Batch Processing--------------------------------")
        logging.info("-----------------------------------------------------------------------------------------")
        logging.info("------------------------------Validating Segment Parameters------------------------------")
        # Update the model parameters
        for algorithm_name, params in self.model.all_methods_n_params.items():
            if algorithm_name == self.model.algorithm:
                for (
                    row,
                    (name, value),
                ) in enumerate(params.items()):
                    logging.info(f"Checking segment params before batch processing {name} {value}")
                    if_valid = self.check_params(name, value)
                    if not if_valid:
                        return

        logging.info("------------------------------Validating Filter Parameters------------------------------")
        self.store_filter_params()
        self.update_segment_parameters()
        if not self.validate_selected_segmentation_method():
            return
        self.pass_filter_params()
        self.show_progress_window(len(self.model.img_path_list))

        self.worker_thread = WorkerThread(self.model)
        self.worker_thread.update_progress.connect(self.update_progress_bar)
        self.worker_thread.processing_done.connect(self.on_processing_done)
        self.worker_thread.error_occurred.connect(self.on_worker_error)
        self.worker_thread.start()

    def show_progress_window(self, num_images: int) -> None:
        """Create and show a progress window with a loading bar.

        Args:
            num_images (int): The total number of images to process, used to set the progress bar range.
        """
        self.progress_dialog = QDialog(self.gui)
        self.progress_dialog.setWindowTitle("Batch Processing")
        self.progress_dialog.setFixedSize(400, 150)

        # Professional flags: Title only, no close button for stability
        self.progress_dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint)

        layout = QVBoxLayout(self.progress_dialog)

        # Add a status label
        self.status_label = QLabel("Processing images...", self.progress_dialog)
        if "Deep Learning" in self.model.algorithm:
            from bubble_analyser.methods.bubmask_method import BubMaskWatershed

            instance = self.model.methods_handler.all_classes.get(self.model.algorithm)
            if isinstance(instance, BubMaskWatershed) and (
                instance.detector is None or not instance.detector.is_loaded
            ):
                self.status_label.setText("Loading Machine Learning engine...\n(First run: ~20-60 seconds)")

        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self.progress_dialog)
        self.progress_bar.setRange(0, num_images)
        self.progress_bar.setValue(0)

        layout.addWidget(self.progress_bar)

        self.progress_dialog.setLayout(layout)
        self.progress_dialog.show()

    def update_progress_bar(self, value: int) -> None:
        """Update the progress bar value.

        Args:
            value (int): The new progress value to display.
        """
        if hasattr(self, "status_label"):
            self.status_label.setText(f"Processing images... ({value}/{self.progress_bar.maximum()})")
        self.progress_bar.setValue(value)
        logging.info(f"Updating progress bar: {value}")

    def on_processing_done(self) -> None:
        """Handle the completion of image processing.

        Closes the progress dialog, emits the batch_processing_done signal,
        and switches to the results tab.
        """
        # Close the progress dialog
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.close()

        # Thread cleanup to prevent QThread destruction errors
        if hasattr(self, "worker_thread") and self.worker_thread:
            self.worker_thread.wait()
            self.worker_thread.deleteLater()

        self.preview_processed_images()
        self.batch_processing_done.emit()
        logging.info("Batch processing completed.")

        # Switch to the final tab
        if self.model.if_finalise_analysis:
            self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.results_tab))
            logging.info("******************************Result Session******************************")

    def on_worker_error(self, error_message: str) -> None:
        """Handle errors that occur in the worker thread.

        This method is called on the main thread when the worker thread encounters an error.

        Args:
            error_message (str): The error message and details from the worker thread.
        """
        # Close the progress dialog if it's open
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.close()

        # Show error dialog on main thread
        QMessageBox.critical(
            self.gui, "Processing Error", "An error occurred during batch processing:\n\n" + error_message
        )

        logging.error(f"Worker thread error handled on main thread: {error_message}")


class ResultsTabHandler(QThread):
    """Handler for the results tab in the GUI.

    This class manages the display and export of processing results, including
    histogram generation and descriptive statistics calculation.

    Attributes:
        params (Config): Configuration parameters for results display.
        save_path (Path): Path where results will be saved.
        ellipses_properties (list[list[dict[str, float]]]): Properties of detected ellipses for all images.
        export_handler (ExportSettingsHandler): Handler for export settings.
        gui: Reference to the main GUI instance.
    """

    check_for_export_path = Signal()

    def __init__(self, model: ImageProcessingModel, params: Config) -> None:
        """Initialize the results tab handler.

        Args:
            model (ImageProcessingModel): The model containing image processing results.
            params (Config): Configuration parameters containing default values.
        """
        super().__init__()

        # Properties to be exported
        self.model = model
        self.ellipses_properties: list[list[dict[str, float | str]]]
        self.px2mm = 0.0
        self.px2mm_display = 0.0
        self.params = params
        self.bknd_img_path = cast(Path, None)

        # Parameters handling export settings
        self.export_handler: ExportSettingsHandler
        self.if_dinf_displayed: bool = False
        self.save_path = params.save_path

    def load_gui(self, gui: MainWindow) -> None:
        """Load a reference to the GUI instance.

        This method stores a reference to the main GUI instance and prepares the handler
        to interact with GUI components for displaying and exporting results.

        Args:
            gui: The main GUI instance.
        """
        self.gui = gui
        # results tab
        self.gui.histogram_by.currentIndexChanged.connect(self.generate_histogram)
        self.gui.pdf_checkbox.stateChanged.connect(self.generate_histogram)
        self.gui.cdf_checkbox.stateChanged.connect(self.generate_histogram)

        self.gui.bins_spinbox.valueChanged.connect(self.generate_histogram)

        self.gui.min_x_axis_input.textChanged.connect(self.generate_histogram)
        self.gui.max_x_axis_input.textChanged.connect(self.generate_histogram)

        self.gui.legend_position_combobox.currentIndexChanged.connect(self.generate_histogram)
        self.gui.d32_checkbox.stateChanged.connect(self.generate_histogram)
        self.gui.dmean_checkbox.stateChanged.connect(self.generate_histogram)
        self.gui.dxy_checkbox.stateChanged.connect(self.generate_histogram)
        self.gui.dxy_x_input.textChanged.connect(self.generate_histogram)
        self.gui.dxy_y_input.textChanged.connect(self.generate_histogram)
        self.gui.save_button.clicked.connect(self.save_results)
        self.gui.export_images_button.clicked.connect(self.export_annotated_images)
        self.gui.export_ml_button.clicked.connect(self.export_ml_data)

    def load_ellipse_properties(
        self,
        properties: list[list[dict[str, float | str]]],
        algorithm: str,
        all_methods_n_params: dict[str, dict[str, float | int]],
        param_dict_1: dict[str, float | str],
        px2mm_display: float,
        bknd_img_path: Path = cast(Path, None),
    ) -> None:
        """Load the properties of detected ellipses for display and analysis.

        Args:
            properties (list[list[dict[str, float]]]): Properties of detected ellipses for all images.
            algorithm (str): The name of the algorithm used for detection.
            all_methods_n_params (dict[str, dict[str, float | int]]): Dictionary containing
                parameters for all algorithms.
            param_dict_1 (dict[str, float | str]): Dictionary containing parameters for the
                algorithm.
            px2mm_display (float): The pixel-to-millimeter conversion ratio for display purposes.
            bknd_img_path (Path): Path to the background image file. Defaults to None.
        """
        self.ellipses_properties = properties
        self.algorithm = algorithm
        self.all_methods_n_params = all_methods_n_params
        self.param_dict_1 = param_dict_1
        self.px2mm_display = px2mm_display
        self.bknd_img_path = bknd_img_path

    def generate_histogram(self) -> None:
        """Generate and display a histogram of bubble sizes.

        Creates a histogram showing the distribution of equivalent diameters of detected bubbles.
        Optionally displays PDF, CDF, and characteristic diameters (d32, dmean, dxy) based on
        user selections. Updates the plot with appropriate labels and legend.

        Supports two histogram types:
        - Count: Shows the count of bubbles for each diameter range
        - Volume: Shows the volume of bubbles (using diameter^3) for each diameter range
        """
        num_bins = self.gui.bins_spinbox.value()
        show_pdf = self.gui.pdf_checkbox.isChecked()
        show_cdf = self.gui.cdf_checkbox.isChecked()
        show_d32 = self.gui.d32_checkbox.isChecked()
        show_dmean = self.gui.dmean_checkbox.isChecked()
        show_dxy = self.gui.dxy_checkbox.isChecked()
        histogram_type = self.gui.histogram_by.currentText()

        logging.info(f"Histogram type: {histogram_type}")

        try:
            equivalent_diameters_array = self.get_equivalent_diameters_list()
        except AttributeError as e:
            error = str(e)
            self._show_warning(
                "Error in Histogram Generation",
                f"{error}, please process the images first.",
            )
            return

        try:
            x_min = float(self.gui.min_x_axis_input.text())
        except ValueError:
            x_min = float(np.min(equivalent_diameters_array)) if len(equivalent_diameters_array) > 0 else 0.0

        try:
            x_max = float(self.gui.max_x_axis_input.text())
        except ValueError:
            x_max = float(np.max(equivalent_diameters_array)) if len(equivalent_diameters_array) > 0 else 5.0

        # Clear current graph
        self.gui.histogram_canvas.axes.set_xlabel("")
        self.gui.histogram_canvas.axes.set_ylabel("")
        self.gui.histogram_canvas.axes.clear()

        try:
            if self.gui.histogram_canvas.axes2:
                self.gui.histogram_canvas.axes2.clear()
                self.gui.histogram_canvas.axes2.set_ylabel("")
                self.gui.histogram_canvas.axes2.set_yticklabels([])
                self.gui.histogram_canvas.axes2.set_yticks([])
                del self.gui.histogram_canvas.axes2

        except AttributeError:
            pass

        # Plot histogram based on selected type
        if histogram_type == "Volume":
            # Calculate volumes (diameter^3) for each bubble
            volumes = equivalent_diameters_array**3
            counts, bins, _ = self.gui.histogram_canvas.axes.hist(
                equivalent_diameters_array,
                bins=num_bins,
                range=(x_min, x_max),
                weights=volumes,  # Use volumes as weights
            )
            # Set graph labels for volume histogram
            self.gui.histogram_canvas.axes.set_xlabel("Equivalent diameter [mm]")
            self.gui.histogram_canvas.axes.set_ylabel("Volume [mm³]")
        else:  # Count histogram (default)
            counts, bins, _patches = self.gui.histogram_canvas.axes.hist(
                equivalent_diameters_array, bins=num_bins, range=(x_min, x_max)
            )
            # Set graph labels for count histogram
            self.gui.histogram_canvas.axes.set_xlabel("Equivalent diameter [mm]")
            self.gui.histogram_canvas.axes.set_ylabel("Count [#]")

        dxy_x_power: int = int(self.gui.dxy_x_input.text())
        dxy_y_power: int = int(self.gui.dxy_y_input.text())

        # Calculate descriptive sizes based on the selected basis
        d32, d_mean, dxy = self.calculate_descriptive_sizes(
            equivalent_diameters_array, dxy_x_power, dxy_y_power, basis=histogram_type
        )

        # Calculate total bubble count
        total_bubbles = len(equivalent_diameters_array)

        if not self.if_dinf_displayed:
            logging.info(
                f"Basis: {histogram_type}, d32: {d32}, Mean: {d_mean}, "
                f"d{dxy_x_power}{dxy_y_power}: {dxy}, Total bubbles: {total_bubbles}"
            )
            self.if_dinf_displayed = True

        # Update descriptive size label with scientific formatting and bubble count
        subscripts = {
            "0": "₀",
            "1": "₁",
            "2": "₂",
            "3": "₃",
            "4": "₄",
            "5": "₅",
            "6": "₆",
            "7": "₇",
            "8": "₈",
            "9": "₉",
        }
        x_sub = "".join(subscripts.get(c, c) for c in str(dxy_x_power))
        y_sub = "".join(subscripts.get(c, c) for c in str(dxy_y_power))
        dxy_label = f"d{x_sub}{y_sub}"

        basis_label = "Volume-Weighted" if histogram_type == "Volume" else "Number-Weighted"
        mean_label = "d₄₃ (Volume Mean)" if histogram_type == "Volume" else "d₁₀ (Arithmetic)"

        desc_text = (
            f"<b>Results ({basis_label}):</b><br>"
            f"d₃₂ (Sauter) = {d32:.2f} mm<br>"
            f"{mean_label} = {d_mean:.2f} mm<br>"
            f"{dxy_label} = {dxy:.2f} mm<br><br>"
            f"<b>Total Bubbles Found: {total_bubbles}</b>"
        )
        self.gui.descriptive_size_label.setText(desc_text)

        # Optionally add CDF and PDF
        if show_pdf or show_cdf:
            self.gui.histogram_canvas.axes2 = self.gui.histogram_canvas.axes.twinx()
            self.gui.histogram_canvas.axes2.set_ylabel("Probability [%]")

            # For volume histogram, we need to normalize differently
            if histogram_type == "Volume":
                total = np.sum(counts)
                if show_cdf:
                    cdf = np.cumsum(counts) / total * 100
                    self.gui.histogram_canvas.axes2.plot(bins[:-1], cdf, "r-", marker="o", label="CDF")
                if show_pdf:
                    pdf = counts / total * 100
                    self.gui.histogram_canvas.axes2.plot(bins[:-1], pdf, "b-", marker="o", label="PDF")
            else:  # Count histogram
                if show_cdf:
                    cdf = np.cumsum(counts) / np.sum(counts) * 100
                    self.gui.histogram_canvas.axes2.plot(bins[:-1], cdf, "r-", marker="o", label="CDF")
                if show_pdf:
                    pdf = counts / np.sum(counts) * 100
                    self.gui.histogram_canvas.axes2.plot(bins[:-1], pdf, "b-", marker="o", label="PDF")

        # Add characteristic diameter lines
        if show_d32:
            self.gui.histogram_canvas.axes.axvline(x=d32, color="r", linestyle="-", label="d32")

        if show_dmean:
            self.gui.histogram_canvas.axes.axvline(x=d_mean, color="g", linestyle="--", label="dmean")

        if show_dxy:
            self.gui.histogram_canvas.axes.axvline(
                x=dxy, color="b", linestyle="--", label=f"d{dxy_x_power}{dxy_y_power}"
            )

        # Apply Legend Options
        legend_position = self.gui.legend_position_combobox.currentText()

        legend_location_map = {
            "North East": "upper right",
            "North West": "upper left",
            "South East": "lower right",
            "South West": "lower left",
        }

        logging.info(f"Legend_position: {legend_position}")

        # Add legend to the graph
        if show_cdf or show_pdf or show_d32 or show_dmean or show_dxy:
            lines1, labels1 = self.gui.histogram_canvas.axes.get_legend_handles_labels()
            if show_cdf or show_pdf:
                lines2, labels2 = self.gui.histogram_canvas.axes2.get_legend_handles_labels()
                self.gui.histogram_canvas.axes.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc=legend_location_map.get(legend_position, "upper right"),
                )

        # Apply x-axis limits
        self.gui.histogram_canvas.axes.set_xlim(x_min, x_max)

        # Redraw the canvas
        self.gui.histogram_canvas.draw()

        self.current_d32 = d32
        self.current_dmean = d_mean
        self.current_dxy = dxy
        self.current_d_x_power = dxy_x_power
        self.current_d_y_power = dxy_y_power
        return

    def calculate_descriptive_sizes(
        self,
        equivalent_diameters: npt.NDArray[np.float64],
        dxy_x_power: float,
        dxy_y_power: float,
        basis: str = "Count",
    ) -> tuple[float, float, float]:
        """Calculate characteristic diameters based on the selected basis.

        Args:
            equivalent_diameters: Array of equivalent diameters.
            dxy_x_power: Power for x component in d_xy.
            dxy_y_power: Power for y component in d_xy.
            basis: "Count" for number-weighted or "Volume" for volume-weighted.

        Returns:
            tuple: (d32, d_mean, dxy)
        """
        # d32 (Sauter Mean) is physically the same regardless of basis (Sum(d^3)/Sum(d^2))
        d32: float = np.sum(equivalent_diameters**3) / np.sum(equivalent_diameters**2)

        if basis == "Volume":
            # Volume-weighted Mean (d43)
            d_mean = np.sum(equivalent_diameters**4) / np.sum(equivalent_diameters**3)
            # Volume-weighted d_xy: Shift moments by 3 orders (volume weight is d^3)
            dxy = np.sum(equivalent_diameters ** (dxy_x_power + 3)) / np.sum(equivalent_diameters ** (dxy_y_power + 3))
        else:
            # Number-weighted Mean (d10)
            d_mean = float(np.mean(equivalent_diameters))
            # Number-weighted d_xy
            dxy = np.sum(equivalent_diameters**dxy_x_power) / np.sum(equivalent_diameters**dxy_y_power)

        return d32, d_mean, dxy

    def get_equivalent_diameters_list(self) -> npt.NDArray[np.float64]:
        """Extract equivalent diameters from all detected ellipses.

        Returns:
            npt.NDArray[np.float64]: Array of equivalent diameters from all processed images.
        """
        if hasattr(self, "model") and self.model is not None:
            self.ellipses_properties = self.model.refresh_ellipses_properties()

        equivalent_diameters = []
        for image in self.ellipses_properties:
            for ellipse in image:
                equivalent_diameters.append(ellipse["equivalent_diameter"])

        equivalent_diameters_array = np.array(equivalent_diameters)
        return equivalent_diameters_array

    def save_results(self) -> None:
        """Saves histogram and data using a standard Save As dialog."""
        from PySide6.QtWidgets import QFileDialog

        # 1. Ask user where to save
        default_name = f"Bubble_Analyser_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path, _ = QFileDialog.getSaveFileName(
            self.gui,
            "Save Analysis Results",
            str(Path.home() / default_name),
            "Excel Files (*.xlsx)",
        )

        if not file_path:
            return

        # Ensure .xlsx extension
        base_path = Path(file_path)
        if base_path.suffix.lower() != ".xlsx":
            base_path = base_path.with_suffix(".xlsx")

        excel_path = str(base_path)
        graph_path = str(base_path.with_suffix(".png"))

        try:
            # 2. Save graph and combined data
            self.save_graph(graph_path)
            self.save_combined_data(excel_path)

            QMessageBox.information(
                self.gui,
                "Results Saved",
                "Results have been saved successfully:\n\n"
                f"Excel: {os.path.basename(excel_path)}\n"
                f"Graph: {os.path.basename(graph_path)}",
            )
            logging.info(f"Results saved to {os.path.dirname(excel_path)}")
        except Exception as e:
            logging.error(f"Error during save: {e}")
            self._show_warning("Save Error", f"An error occurred while saving: {e}")

    def export_annotated_images(self) -> None:
        """Export final annotated images (right preview) for all processed images."""
        folder_path = QFileDialog.getExistingDirectory(self.gui, "Select Folder to Export Annotated Images")
        if not folder_path:
            return

        success_count = 0
        error_count = 0
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        for name in self.model.img_path_list:
            annotated_img = self.model.get_or_render_annotated_image(name)
            if annotated_img is not None and annotated_img.size > 0:
                base_name = os.path.splitext(os.path.basename(name))[0]
                export_path = os.path.join(folder_path, f"{base_name}_annotated.png")
                try:
                    img_uint8 = np.asarray(annotated_img, dtype=np.uint8)
                    if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
                        bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                    else:
                        bgr = img_uint8
                    success, enc = cv2.imencode(".png", bgr)
                    if success:
                        enc.tofile(export_path)
                        success_count += 1
                    else:
                        if cv2.imwrite(export_path, bgr):
                            success_count += 1
                        else:
                            error_count += 1
                except Exception as e:
                    logging.error(f"Failed to export {name}: {e}")
                    error_count += 1
            else:
                error_count += 1

        QMessageBox.information(
            self.gui,
            "Export Complete",
            f"Successfully exported {success_count} images to {folder_path}."
            + (f"\nFailed: {error_count}" if error_count > 0 else ""),
        )

    def export_ml_data(self) -> None:
        """Export the corrected segmentations as a dataset, in the format the user picks."""
        dialog = ExportFormatDialog(self.gui)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        dataset_format = dialog.selected_format()

        root_path = QFileDialog.getExistingDirectory(self.gui, f"Select Folder for the {dataset_format.label} Export")
        if not root_path:
            return

        processed_paths = [
            name
            for name in self.model.img_path_list
            if (state := self.model.img_dict.get(name)) is not None
            and (len(state.ellipses) > 0 or state.labelled_ellipses_mask.size > 0)
        ]
        frames: list[FrameAnnotation] = []
        for name in processed_paths:
            state = self.model.img_dict.get(name)
            if state is None:
                continue
            height, width = state.img_rgb.shape[:2]
            frames.append(
                FrameAnnotation(
                    image_path=Path(name),
                    ellipses=state.ellipses,
                    polygons=getattr(state, "ellipse_polygons", ()),
                    annotation_shape=(height, width),
                    labelled_mask=state.labelled_ellipses_mask,
                )
            )

        try:
            summary = export_dataset(Path(root_path), frames, dataset_format.key)
        except ValueError as error:
            QMessageBox.warning(self.gui, "Nothing to Export", str(error))
            return
        except Exception as error:
            logging.exception("Dataset export failed")
            QMessageBox.critical(self.gui, "Export Failed", str(error))
            return

        unprocessed = len(self.model.img_path_list) - len(processed_paths)
        lines = [
            f"Exported {summary.frames_exported} images and {summary.instances_exported} bubbles "
            f"as {dataset_format.label} to {root_path}.",
            f"Skipped {unprocessed} unprocessed images.",
        ]
        if summary.frames_without_bubbles:
            lines.append(f"Skipped {summary.frames_without_bubbles} processed images without bubbles.")
        if summary.failed:
            lines.append(f"Failed: {len(summary.failed)} ({summary.failed[0][0]}: {summary.failed[0][1]})")
        QMessageBox.information(self.gui, "Export Complete", "\n".join(lines))

    def save_graph(self, export_path: str) -> None:
        """Saves the current histogram to the selected folder."""
        self.gui.histogram_canvas.fig.savefig(export_path)

    def save_combined_data(self, export_path: str) -> None:
        """Save both ellipse data and config data to a single Excel file with multiple sheets."""
        if hasattr(self, "model") and self.model is not None:
            self.ellipses_properties = self.model.refresh_ellipses_properties()

        # Prepare ellipse data
        ellipse_headers = [
            "major_axis_length(px)",
            "minor_axis_length(px)",
            "equivalent_diameter(px)",
            "area(px2)",
            "perimeter(px)",
            "eccentricity",
            "image name",
        ]

        ellipse_rows = []
        number_of_bubbles = 0
        number_of_photos = 0
        for image in self.ellipses_properties:
            number_of_photos += 1
            for circle in image:
                number_of_bubbles += 1
                ellipse_rows.append(
                    [
                        circle["major_axis_length"],
                        circle["minor_axis_length"],
                        circle["equivalent_diameter"],
                        circle["area"],
                        circle["perimeter"],
                        circle["eccentricity"],
                        circle.get("filename", ""),
                    ]
                )

        # Create ellipse DataFrame
        ellipse_df = pd.DataFrame(data=ellipse_rows, columns=ellipse_headers)  # type: ignore

        # Prepare config data
        config_data = []

        # Add segmentation parameters
        config_data.append(["Section", "Parameter", "Value"])
        # config_data.append(["Calibration", "px/mm(raw)", self.px2mm])
        config_data.append(["Calibration", "px/mm(display)", str(self.px2mm_display)])
        config_data.append(["Segmentation", "Algorithm", self.algorithm])

        for algorithm_name, params in self.all_methods_n_params.items():
            if algorithm_name == self.algorithm:
                for key, value in params.items():
                    config_data.append(["Segmentation", key, str(value)])

        # Add filtering parameters
        for key, value in self.param_dict_1.items():  # type: ignore
            config_data.append(["Filtering", key, str(value)])

        # Add timestamp
        config_data.append(["Metadata", "Generated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        # Create config DataFrame
        config_df = pd.DataFrame(config_data[1:], columns=config_data[0])  # Skip header row

        summary_headers = ["Properties", "Value"]

        summary_data = [
            ["d32", self.current_d32],
            ["d_mean", self.current_dmean],
            [f"d{self.current_d_x_power}{self.current_d_y_power}", self.current_dxy],
            ["number of bubbles", number_of_bubbles],
            ["number of photos", number_of_photos],
        ]

        summary_df = pd.DataFrame(data=summary_data, columns=summary_headers)  # type: ignore

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
            ellipse_df.to_excel(writer, sheet_name="Ellipse_Data", index=False)
            config_df.to_excel(writer, sheet_name="Configuration", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        logging.info(f"Combined data saved to {export_path}")

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)


class MainHandler:
    """Main controller class that coordinates all handlers and GUI components.

    This class serves as the central coordinator for the application, initializing and
    managing all handler classes, models, and the GUI. It connects signals between
    components and provides methods to handle user interactions across different tabs.

    Attributes:
        toml_file_path (Path): Path to the configuration TOML file.
        toml_handler (TomlFileHandler): Handler for loading and validating TOML configuration.
        input_file_model (InputFilesModel): Model for managing input image files.
        folder_tab_handler (FolderTabHandler): Handler for the folder selection tab.
        calibration_model (CalibrationModel): Model for managing calibration data.
        calibration_tab_handler (CalibrationTabHandler): Handler for the calibration tab.
        image_processing_model (ImageProcessingModel): Model for image processing operations.
        image_processing_tab_handler (ImageProcessingTabHandler): Handler for the image processing tab.
        results_tab_handler (ResultsTabHandler): Handler for the results tab.
        app (QApplication): The main Qt application instance.
        gui (MainWindow): The main GUI window instance.
        export_handler (ExportSettingsHandler): Handler for export settings.
    """

    def __init__(self) -> None:
        """Initialize the MainHandler with all necessary components and handlers.

        Sets up all models, handlers, and GUI components, and establishes connections
        between them to enable proper application functionality.
        """
        # First try to find config.toml relative to the executable when packaged
        import os
        import sys

        self.gui: MainWindow

        # Set up logging to capture terminal output
        # self.setup_logging()

        # Get the base directory for the application
        if getattr(sys, "frozen", False):
            # If the application is run as a bundle (PyInstaller)
            base_dir = os.path.dirname(sys.executable)
            # Try to find config.toml in the same directory as the executable
            config_path = os.path.join(base_dir, "config.toml")
            if os.path.exists(config_path):
                self.toml_file_path = Path(config_path)
            else:
                # Fall back to the bundled path
                self.toml_file_path = Path(__file__).parent.parent / "config.toml"
        else:
            # If running in development mode
            self.toml_file_path = Path(__file__).parent.parent / "config.toml"

        logging.info(f"Using TOML file at: {self.toml_file_path}")

        self.initialize_gui()
        self.initialize_handlers()
        self.initialize_handlers_signals()
        self.initialize_new_export_settings()
        self.load_export_settings()

        self.load_gui_for_handlers()
        self.connect_gui_and_handlers()

    def initialize_handlers(self) -> None:
        """Initialize all handler classes and models used by the application.

        Creates instances of all necessary handlers and models with appropriate
        configuration parameters from the TOML file.
        """
        logging.info("Initializing Handlers...")
        self.toml_handler = TomlFileHandler(self.toml_file_path)

        self.input_file_model = InputFilesModel()
        self.folder_tab_handler = FolderTabHandler(self.input_file_model, params=self.toml_handler.params)

        self.calibration_model = CalibrationModel()
        self.calibration_tab_handler = CalibrationTabHandler(self.calibration_model, params=self.toml_handler.params)

        self.image_processing_model = ImageProcessingModel(params=self.toml_handler.params)
        self.image_processing_tab_handler = ImageProcessingTabHandler(
            self.image_processing_model, params=self.toml_handler.params
        )

        self.results_tab_handler = ResultsTabHandler(model=self.image_processing_model, params=self.toml_handler.params)

    def initialize_handlers_signals(self) -> None:
        """Connect signals between handlers to enable communication.

        Establishes signal-slot connections between different handlers to enable
        proper event propagation and response to user actions.
        """
        self.image_processing_tab_handler.batch_processing_done.connect(self.start_generate_histogram)

    def initialize_gui(self) -> None:
        """Initialize the main GUI application and window, and display it.

        Creates the QApplication instance and the main window for the application.
        """
        from bubble_analyser.gui import MainWindow

        # Only configure logging if it hasn't been configured by __main__.py
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)

        logging.info("Initializing GUI...")

        # Apply macOS-specific fixes
        if sys.platform == "darwin":
            # Set environment variable to fix Qt rendering issues on macOS
            os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
            # Disable native menu bar on macOS to prevent crashes
            os.environ.setdefault("QT_MAC_NO_NATIVE_MENUBAR", "1")

        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        # Set application properties for better macOS compatibility
        self.app.setApplicationName("Bubble Analyser")
        self.app.setApplicationVersion("1.0")
        self.app.setOrganizationName("Bubble Analyser")

        self.gui = MainWindow()
        self.gui.show()

    def gui_exiting(self) -> None:
        """Handle the exit of the GUI application.

        Ensures that the application exits gracefully when the main window is closed.
        """
        if self.app is not None:
            sys.exit(self.app.exec())

    def load_gui_for_handlers(self) -> None:
        """Load GUI references into all handlers.

        Provides each handler with a reference to the main GUI instance to enable
        direct interaction with GUI components. This method must be called after
        the GUI has been initialized and before handlers start interacting with
        GUI components.
        """
        logging.info("Connecting GUI and Handlers...")
        self.folder_tab_handler.load_gui(self.gui)
        self.calibration_tab_handler.load_gui(self.gui)
        self.image_processing_tab_handler.load_gui(self.gui)
        self.results_tab_handler.load_gui(self.gui)

    def connect_gui_and_handlers(self) -> None:
        """Connect GUI components to their respective handlers.

        Sets up signal-slot connections between GUI components and their respective
        handlers to enable proper event handling and interaction.
        """
        # menubar
        self.gui.export_setting_action.triggered.connect(self.menubar_open_export_settings_dialog)
        self.gui.restart_action.triggered.connect(self.menubar_ask_if_restart)
        self.gui.restart_button.clicked.connect(self.menubar_ask_if_restart)

        # folder tab
        self.gui.folder_path_edit.setText(str(self.folder_tab_handler.image_path))
        self.gui.select_folder_button.clicked.connect(self.tab1_select_folder)
        self.gui.confirm_folder_button.clicked.connect(self.tab1_confirm_folder_selection)
        self.gui.image_list.clicked.connect(self.folder_tab_handler.preview_image_folder_tab)

        # calibration tab
        self.gui.pixel_img_name.setText(str(self.calibration_tab_handler.px_img_path))
        self.gui.pixel_img_select_button.clicked.connect(self.tab2_select_ruler_button)
        self.gui.bg_corr_select_button.clicked.connect(self.tab2_select_bg_corr_image)
        self.gui.confirm_px_mm_button.clicked.connect(
            self.tab2_confirm_calibration
        )  # Connect confirm button to the handler

        # image processing tab
        # column 1
        self.gui.prev_button.clicked.connect(lambda: self.tab3_update_sample_image("prev"))
        self.gui.next_button.clicked.connect(lambda: self.tab3_update_sample_image("next"))
        self.gui.preview_processed_images_button.clicked.connect(self.tab3_preview_processed_images)
        # column 2
        self.image_processing_tab_handler.initialize_algorithm_combo()
        self.gui.algorithm_combo.currentTextChanged.connect(
            lambda: self.tab3_handle_algorithm_change(self.gui.algorithm_combo.currentText())
        )
        self.tab3_load_parameter_table_1(self.gui.algorithm_combo.currentText())
        self.gui.preview_button1.clicked.connect(self.tab3_confirm_parameter_before_filtering)
        # column 3
        self.tab3_initialize_parameter_table_2()
        self.gui.manual_adjustment_button.clicked.connect(self.tab3_ellipse_manual_adjustment)
        self.gui.preview_button2.clicked.connect(self.tab3_confirm_parameter_for_filtering)
        self.gui.batch_process_button.clicked.connect(self.tab3_ask_if_batch)
        self.gui.finalise_analysis_button.clicked.connect(self.tab3_finalise_analysis)

    def disconnect_gui_and_handlers(self) -> None:
        """Disconnect GUI components from their respective handlers.

        Removes signal-slot connections between GUI components and their respective
        handlers to prevent further event handling and interaction.
        """
        # menubar
        self.gui.export_setting_action.triggered.disconnect(self.menubar_open_export_settings_dialog)
        self.gui.restart_action.triggered.disconnect(self.menubar_ask_if_restart)
        self.gui.restart_button.clicked.disconnect(self.menubar_ask_if_restart)

        # folder tab
        self.gui.folder_path_edit.setText(str(self.folder_tab_handler.image_path))
        self.gui.select_folder_button.clicked.disconnect(self.tab1_select_folder)
        self.gui.confirm_folder_button.clicked.disconnect(self.tab1_confirm_folder_selection)
        self.gui.image_list.clicked.disconnect(self.folder_tab_handler.preview_image_folder_tab)

        # calibration tab
        self.gui.pixel_img_name.setText(str(self.calibration_tab_handler.px_img_path))
        self.gui.pixel_img_select_button.clicked.disconnect(self.tab2_select_ruler_button)
        self.gui.bg_corr_select_button.clicked.disconnect(self.tab2_select_bg_corr_image)
        self.gui.confirm_px_mm_button.clicked.disconnect(
            self.tab2_confirm_calibration
        )  # Connect confirm button to the handler

        # image processing tab
        # column 1
        self.gui.prev_button.clicked.disconnect()
        self.gui.next_button.clicked.disconnect()
        self.gui.preview_processed_images_button.clicked.disconnect(self.tab3_preview_processed_images)
        # column 2
        self.image_processing_tab_handler.initialize_algorithm_combo()
        self.gui.algorithm_combo.currentTextChanged.disconnect()
        self.tab3_load_parameter_table_1(self.gui.algorithm_combo.currentText())
        self.gui.preview_button1.clicked.disconnect(self.tab3_confirm_parameter_before_filtering)
        # column 3
        self.tab3_initialize_parameter_table_2()
        self.gui.manual_adjustment_button.clicked.disconnect(self.tab3_ellipse_manual_adjustment)
        self.gui.preview_button2.clicked.disconnect(self.tab3_confirm_parameter_for_filtering)
        self.gui.batch_process_button.clicked.disconnect(self.tab3_ask_if_batch)
        self.gui.finalise_analysis_button.clicked.disconnect(self.tab3_finalise_analysis)

        # results tab
        self.gui.pdf_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.cdf_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.bins_spinbox.valueChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.min_x_axis_input.textChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.max_x_axis_input.textChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.legend_position_combobox.currentIndexChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.d32_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.dmean_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.dxy_checkbox.stateChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.dxy_x_input.textChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.dxy_y_input.textChanged.disconnect(self.results_tab_handler.generate_histogram)
        self.gui.save_button.clicked.disconnect()
        self.gui.export_images_button.clicked.disconnect()
        self.gui.export_ml_button.clicked.disconnect()

        self.image_processing_tab_handler.batch_processing_done.disconnect(self.start_generate_histogram)

    def disconnect_handlers_signals(self) -> None:
        """Disconnect signals between handlers to prevent further event handling.

        Removes signal-slot connections between handlers to prevent further event
        handling and interaction.
        """
        self.image_processing_tab_handler.batch_processing_done.disconnect()

    def clear_all_gui_contents(self) -> None:
        """Clear all contents from the GUI components.

        Clears all contents from all GUI components to reset the application state.
        """
        self.gui.image_list.clear()

        self.gui.pixel_img_preview.clear()
        self.gui.bg_corr_image_preview.clear()
        self.gui.manual_px_mm_input.clear()

        self.gui.sample_image_preview.clear()
        self.gui.label_before_filtering.axes.clear()
        self.gui.label_before_filtering.draw()
        self.gui.processed_image_preview.axes.clear()
        self.gui.processed_image_preview.draw()

        self.gui.histogram_canvas.axes.clear()
        self.gui.histogram_canvas.draw()

    def restart(self) -> None:
        """Restart the application.

        Resets the application state and restarts the GUI.
        """
        logging.info("##############################Restarting Mission...##############################")
        self.disconnect_gui_and_handlers()
        self.disconnect_handlers_signals()
        self.clear_all_gui_contents()

        self.initialize_handlers()
        self.initialize_handlers_signals()
        self.load_export_settings()
        self.load_gui_for_handlers()
        self.connect_gui_and_handlers()

        self.gui.tabs.setCurrentIndex(self.gui.tabs.indexOf(self.gui.folder_tab))

        logging.info("Application restarted, a new mission initialzed.")
        logging.info("##############################New mission started##############################")

    def load_export_settings(self) -> None:
        """Initialize and configure the export settings handler.

        Creates the export settings handler and provides it to relevant tab handlers
        that need access to export functionality.
        """
        logging.info("Connecting Export Settings with handlers...")
        self.image_processing_tab_handler.export_handler = self.export_handler
        self.results_tab_handler.export_handler = self.export_handler

    def initialize_new_export_settings(self) -> None:
        """Initialize and configure the export settings handler."""
        logging.info("Initializing Export Settings...")
        self.export_handler = ExportSettingsHandler(parent=self.gui)

    def menubar_open_export_settings_dialog(self) -> None:
        """Open the export settings dialog from the menu bar.

        Displays a dialog allowing the user to configure export settings for
        processed images and results.
        """
        self.export_handler.exec()

    def menubar_ask_if_restart(self) -> None:
        """Show a confirmation dialog for restarting the application.

        Displays a message box asking the user if they want to restart the application,
        warning that unsaved progress will be lost.
        """
        restart = QMessageBox.question(
            self.gui,
            "Restart",
            "Do you want to start a new mission? \n Current unsaved progress will be lost",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if restart == QMessageBox.StandardButton.Yes:
            self.restart()

    def _show_warning(self, title: str, message: str) -> None:
        """Display a warning message box to the user.

        Args:
            title (str): The title of the warning dialog.
            message (str): The detailed warning message to display.
        """
        QMessageBox.warning(self.gui, title, message)

    def tab1_select_folder(self) -> None:
        """Handle folder selection in the first tab.

        Delegates the folder selection process to the folder tab handler.
        """
        self.folder_tab_handler.select_folder()

    def tab1_confirm_folder_selection(self) -> None:
        """Confirm the folder selection in the first tab.

        Confirms the selected folder and loads the image path list into the
        image processing model.
        """
        self.folder_tab_handler.confirm_folder_selection()
        # load image path list to Processing Hanlder
        self.image_processing_model.confirm_folder_selection(self.input_file_model.image_list_full_path_in_path)

    def tab2_get_px2mm_ratio(self) -> None:
        """Calculate the pixel-to-millimeter ratio in the calibration tab.

        Delegates the pixel-to-millimeter calculation to the calibration tab handler.
        """
        self.calibration_tab_handler.get_px2mm_ratio()

    def tab2_select_ruler_button(self) -> None:
        """Select a ruler image for pixel-to-millimeter calibration.

        Delegates the ruler image selection to the calibration tab handler.
        """
        self.calibration_tab_handler.select_ruler_button()

    def tab2_select_bg_corr_image(self) -> None:
        """Select a background correction image in the calibration tab.

        Delegates the background image selection to the calibration tab handler.
        """
        self.calibration_tab_handler.select_bg_corr_image()

    def tab2_confirm_calibration(self) -> None:
        """Confirm the calibration settings and update models.

        Confirms the calibration settings and updates the image processing model
        with the calibration data, including pixel-to-millimeter ratio and
        background image if selected.
        """
        self.calibration_tab_handler.confirm_calibration()
        self.image_processing_model.update_px2mm_display(self.calibration_model.px2mm_display)

        if self.calibration_model.if_bknd:
            self.image_processing_model.if_bknd = self.calibration_model.if_bknd
            self.image_processing_model.get_bknd_img_path(self.calibration_model.bknd_img_path)

    def tab3_load_parameter_table_1(self, algorithm: str) -> None:
        """Load parameters for the selected algorithm in the first parameter table.

        Args:
            algorithm (str): The name of the selected algorithm.
        """
        self.image_processing_tab_handler.load_parameter_table_1(algorithm)

    def tab3_initialize_parameter_table_2(self) -> None:
        """Initialize the second parameter table in the image processing tab.

        Prepares the table for displaying filtering parameters.
        """
        self.image_processing_tab_handler.initialize_parameter_table_2()

    def tab3_update_sample_image(self, status: str) -> None:
        """Update the sample image display based on processing status.

        Args:
            status (str): The current processing status to determine which image to display.
        """
        self.image_processing_tab_handler.update_sample_image(status)

    def tab3_preview_processed_images(self) -> None:
        """Preview the processed images in the image processing tab.

        Delegates the image preview functionality to the image processing tab handler.
        """
        self.image_processing_tab_handler.preview_processed_images()

    def tab3_handle_algorithm_change(self, algorithm: str) -> None:
        """Handle a change in the selected algorithm in the image processing tab.

        Delegates the algorithm change handling to the image processing tab handler,
        which updates parameters and UI elements accordingly.

        Args:
            algorithm (str): The name of the newly selected algorithm.
        """
        self.image_processing_tab_handler.handle_algorithm_change(algorithm)

    def tab3_confirm_parameter_before_filtering(self) -> None:
        """Confirm the parameters for the first step of image processing.

        Delegates the parameter confirmation to the image processing tab handler,
        which validates all parameters against the configuration schema before
        proceeding with the first processing step.
        """
        self.image_processing_tab_handler.confirm_parameter_before_filtering()

    def tab3_confirm_parameter_for_filtering(self) -> None:
        """Confirm the filtering parameters and apply them to the current image.

        Delegates the filtering parameter confirmation to the image processing tab handler,
        which validates all filtering parameters against the configuration schema before
        proceeding with the second processing step.
        """
        self.image_processing_tab_handler.confirm_parameter_for_filtering()

    def tab3_ellipse_manual_adjustment(self) -> None:
        """Enable manual adjustment of detected ellipses in the image processing tab.

        Delegates the ellipse manual adjustment functionality to the image processing
        tab handler, allowing users to modify automatically detected ellipses.
        """
        self.image_processing_tab_handler.ellipse_manual_adjustment()

    def tab3_ask_if_batch(self) -> None:
        """Ask the user if they want to perform batch processing on all images.

        Delegates the batch processing confirmation dialog to the image processing
        tab handler, which prompts the user and initiates batch processing if confirmed.
        """
        self.image_processing_tab_handler.ask_if_batch()

    def tab3_finalise_analysis(self) -> None:
        """Finalise the analysis and save the results.

        Delegates the finalise analysis functionality to the image processing tab handler,
        which saves the results to the specified export path and updates the UI accordingly.
        """
        self.image_processing_tab_handler.finalise_analysis()

    def start_generate_histogram(self) -> None:
        """Generate histograms based on the processed ellipse properties.

        Loads the ellipse properties from the image processing model into the
        results tab handler and triggers histogram generation to visualize
        the distribution of bubble sizes and other properties.
        """
        self.image_processing_model.refresh_ellipses_properties()
        self.results_tab_handler.load_ellipse_properties(
            properties=self.image_processing_model.ellipses_properties,
            algorithm=self.image_processing_model.algorithm,
            all_methods_n_params=self.image_processing_model.all_methods_n_params,
            param_dict_1=self.image_processing_model.filter_param_dict_1,
            px2mm_display=self.calibration_model.px2mm_display,
            bknd_img_path=self.calibration_model.bknd_img_path,
        )
        self.results_tab_handler.generate_histogram()

    # def setup_logging(self) -> None:
    #     """Set up logging to capture terminal output."""
    #     # Use user's home directory or temporary directory for logs
    #     import tempfile

    #     # Try to use the user's application data directory first
    #     user_home = Path.home()
    #     app_data_dir = user_home / "Library" / "Application Support" / "BubbleAnalyser"

    #     try:
    #         app_data_dir.mkdir(parents=True, exist_ok=True)
    #         logs_dir = app_data_dir / "logs"
    #         logs_dir.mkdir(exist_ok=True)
    #     except OSError:
    #         # Fallback to temp directory if we can't write to app data dir
    #         logs_dir = Path(tempfile.gettempdir()) / "BubbleAnalyser" / "logs"
    #         logs_dir.mkdir(parents=True, exist_ok=True)

    #     log_file = logs_dir / "bubble_analyser.log"

    #     # Configure logging
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format="%(asctime)s - %(levelname)s - %(message)s",
    #         handlers=[
    #             logging.FileHandler(log_file),
    #             logging.StreamHandler(),
    #         ],
    #     )

    #     logging.info("Logging initialized")
    #     logging.info("******************************Folder Selection Session******************************")


if __name__ == "__main__":
    main_handler = MainHandler()
