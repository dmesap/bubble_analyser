"""GUI module for the Bubble Analyser application.

This module implements the graphical user interface for the Bubble Analyser application,
providing a user-friendly interface for analyzing bubble images. It defines the main
window and all UI components including tabs for folder selection, calibration, image
processing, and results visualization.

The module contains the following key components:

- MplCanvas: A class for embedding Matplotlib figures in the PySide6 application
- MainWindow: The main application window with multiple tabs for different functionalities

The GUI is structured around a tab-based interface with separate sections for:

1. Folder selection - for choosing input image folders
2. Calibration - for pixel-to-mm calibration and background image selection
3. Image processing - for bubble detection, filtering, and manual adjustment
4. Results - for visualizing and exporting histogram data

The module works closely with event handlers defined in the event_handlers module
to connect UI interactions with the underlying processing functionality.
"""

# import bubble_analyser.gui.event_handlers as hd
# Try to change orders
from typing import cast

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bubble_analyser.widgets.zoom_view import ZoomablePreviewLabel


class SpinnerWidget(QWidget):
    """A custom widget that displays a rotating spinner animation.

    This widget draws a rotating arc to indicate an indeterminate process.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the spinner widget.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(50)  # Update every 50ms
        self.setFixedSize(100, 100)

    def rotate(self) -> None:
        """Update the rotation angle and trigger a repaint."""
        self.angle = (self.angle + 10) % 360
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the spinner.

        Args:
            event: The paint event.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Center point
        center_x = width / 2
        center_y = height / 2

        # Radius
        radius = min(width, height) / 2 - 10

        # Draw background circle (optional, faint)
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(6)
        painter.setPen(pen)
        painter.drawEllipse(int(center_x - radius), int(center_y - radius), int(radius * 2), int(radius * 2))

        # Draw rotating arc
        pen.setColor(QColor(66, 135, 245))  # Blue color
        painter.setPen(pen)

        # Draw a 90 degree arc
        span_angle = 90 * 16  # Angles are in 1/16th of a degree
        start_angle = -self.angle * 16

        painter.drawArc(
            int(center_x - radius), int(center_y - radius), int(radius * 2), int(radius * 2), start_angle, span_angle
        )


class ProcessingDialog(QDialog):
    """A modal dialog that displays a processing spinner.

    This dialog is used to block user interaction while a background task is running.
    """

    def __init__(self, parent: QWidget | None = None, title: str = "Processing...") -> None:
        """Initialize the processing dialog.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
            title (str, optional): The title of the dialog. Defaults to "Processing...".
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(350, 230)  # Reverted height, kept width for visibility

        # Remove the close button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout(self)

        self.spinner = SpinnerWidget(self)
        layout.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel(title, self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        # Time elapsed label
        self.time_label = QLabel("Time elapsed: 0s", self)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        # Timer for updating elapsed time
        self.elapsed_time = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Update every second

    def update_time(self) -> None:
        """Update the elapsed time label."""
        self.elapsed_time += 1
        self.time_label.setText(f"Time elapsed: {self.elapsed_time}s")


class MplCanvas(FigureCanvas):
    """A class for creating a Matplotlib figure within a PySide6 application.

    Attributes:
        fig: The Matplotlib figure.
        axes: The axes of the figure.
    """

    def __init__(self, parent: QMainWindow, width: float = 5, height: float = 4, dpi: float = 100) -> None:
        """The constructor for MplCanvas.

        Parameters:
            parent: The parent widget.
            width: The width of the figure in inches.
            height: The height of the figure in inches.
            dpi: The dots per inch of the figure.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):
    """The main window of the Bubble Analyser application.

    This class represents the main window of the application, containing multiple tabs
    for different functionalities including folder selection, calibration, image
    processing, and results visualization. It manages the overall layout and
    interaction between different components of the application.

    Attributes:
        main_handler: The main event handler for the application.
        folder_tab_handler: Handler for folder selection tab operations.
        calibration_tab_handler: Handler for calibration tab operations.
        image_processing_tab_handler: Handler for image processing tab operations.
        results_tab_handler: Handler for results tab operations.
        bknd_img_exist: Flag indicating if background image exists.
        calibration_confirmed: Flag indicating if calibration is confirmed.
    """

    def __init__(self) -> None:
        """Initialize the main window of the Bubble Analyser application.

        Sets up the main window with all its components including the menu bar
        and different tabs for folder selection, calibration, image processing,
        and results visualization.

        Args:
            main_handler: The main event handler that manages the application's logic.
        """
        super().__init__()
        self.bknd_img_exist = False
        self.calibration_confirmed = False

        self.setWindowTitle("Bubble Analyser v0.3.0")
        self.setGeometry(100, 100, 1200, 800)

        # Create a Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add Folder Tab
        self.folder_tab = QWidget()
        self.tabs.addTab(self.folder_tab, "Folder")
        self.sample_images_confirmed = False

        # Add Calibration Tab
        self.calibration_tab = QWidget()
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.bg_image_confirmed = False
        self.px_res_confirmed = False

        # Add Image Processing Tab
        self.current_algorithm: str = cast(str, None)
        self.image_processing_tab = QWidget()
        self.tabs.addTab(self.image_processing_tab, "Bubble detection and filtering")

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.setup_menu_bar()

        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Results")

        self.setup_folder_tab()
        self.setup_calibration_tab()
        self.setup_image_processing_tab()
        self.setup_results_tab()

    def setup_menu_bar(self) -> None:
        """Set up the application's menu bar.

        Creates and configures the menu bar with settings menu items including
        options to export settings and restart a new session.
        """
        setting_menu = QMenu("Settings")
        self.export_setting_action = QAction("Export settings", self)
        # self.export_setting_action.triggered.connect(self.main_handler.menubar_open_export_settings_dialog)
        setting_menu.addAction(self.export_setting_action)

        self.restart_action = QAction("Restart a new session", self)
        # self.restart_action.triggered.connect(self.main_handler.menubar_ask_if_restart)
        setting_menu.addAction(self.restart_action)

        setting_menu = self.menu_bar.addMenu(setting_menu)  # type: ignore

    def setup_folder_tab(self) -> None:
        """Set up the folder tab.

        This function sets up the folder tab, which contains the following components:

        1. A text box for user to input the folder path.
        2. A button to select the folder.
        3. A button to confirm the folder selection.
        4. A list of images in the selected folder.
        5. An image preview section to show the selected image.

        When the user selects an image from the list, the image will be previewed in
        the image preview section.
        """
        layout = QVBoxLayout(self.folder_tab)

        # Top Part: Folder Selection
        top_frame = QFrame()
        top_layout = QHBoxLayout(top_frame)
        self.folder_path_edit = QLineEdit()

        self.select_folder_button = QPushButton("Select Folder")
        self.confirm_folder_button = QPushButton("Confirm Folder")
        top_layout.addWidget(self.select_folder_button)
        top_layout.addWidget(self.folder_path_edit)
        top_layout.addWidget(self.confirm_folder_button)

        # Bottom Left: List of images in folder
        bottom_left_frame = QFrame()
        bottom_left_layout = QVBoxLayout(bottom_left_frame)
        self.image_list = QListWidget()
        bottom_left_layout.addWidget(self.image_list)

        # Bottom Right: Image Preview
        bottom_right_frame = QFrame()
        bottom_right_layout = QVBoxLayout(bottom_right_frame)
        self.image_preview = ZoomablePreviewLabel(viewer_title="Image preview")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setFixedSize(600, 600)  # Set a fixed size for the image preview
        bottom_right_layout.addWidget(self.image_preview)

        # Split the bottom part into two sections
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.addWidget(bottom_left_frame)
        bottom_layout.addWidget(bottom_right_frame)

        # Add top and bottom frames to the main layout
        layout.addWidget(top_frame, 1)
        layout.addWidget(bottom_frame, 6)

    def setup_calibration_tab(self) -> None:
        """Set up the calibration tab.

        Set up the calibration tab, which contains the following components:

        1. A text box for user to input the name of the image for pixel
           resolution calibration.
        2. A button to select the image for pixel resolution calibration.
        3. A label to preview the selected image.
        4. A text box for user to input the name of the background image.
        5. A button to select the background image.
        6. A label to preview the selected background image.
        7. A button to confirm the calibration and background image.

        When the user selects an image from the list, the image will be
        previewed in the image preview section. When the user clicks the
        "Confirm" button, the image will be processed and the pixel-to-mm
        ratio will be calculated and stored in the "px_mm" attribute of the
        MainWindow object. The background image will be stored in the
        "bknd_img" attribute of the MainWindow object. The tab will then be
        switched to the next tab.
        """
        layout = QGridLayout(self.calibration_tab)

        # Create top frame
        top_frame = QFrame()
        top_frame_layout = QHBoxLayout()
        top_frame.setLayout(top_frame_layout)

        # Pixel Resolution Calibration
        pixel_img_frame = QFrame()
        pixel_img_layout = QVBoxLayout(pixel_img_frame)
        self.pixel_img_name = QLineEdit()
        # self.pixel_img_name.setText(str(self.calibration_tab_handler.px_img_path))
        self.pixel_img_select_button = QPushButton("Select ruler image")
        # self.pixel_img_select_button.clicked.connect(self.main_handler.tab2_select_ruler_button)
        self.pixel_img_preview = ZoomablePreviewLabel(viewer_title="Ruler image")
        self.pixel_img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixel_img_preview.setFixedSize(400, 300)

        pixel_img_layout.addWidget(QLabel("Step 1: Pixel resolution calibration"))
        pixel_img_layout.addWidget(QLabel("Image name"))
        pixel_img_layout.addWidget(self.pixel_img_name)
        pixel_img_layout.addWidget(self.pixel_img_select_button)
        pixel_img_layout.addWidget(self.pixel_img_preview)
        pixel_img_layout.addStretch()

        # Background Correction Image
        bg_corr_frame = QFrame()
        bg_corr_layout = QVBoxLayout(bg_corr_frame)
        self.bg_corr_image_name = QLineEdit()
        self.bg_corr_image_name.setText("Choose your background image from local")
        # bg_corr_confirm_button = QPushButton("Confirm")
        # bg_corr_confirm_button.clicked.connect(self.confirm_bg_corr_image)
        self.bg_corr_select_button = QPushButton("Select background image (optional)")
        # self.bg_corr_select_button.clicked.connect(self.main_handler.tab2_select_bg_corr_image)
        self.bg_corr_image_preview = ZoomablePreviewLabel(viewer_title="Background image")
        self.bg_corr_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bg_corr_image_preview.setFixedSize(400, 300)

        bg_corr_layout.addWidget(QLabel("Step 2: Background correction image"))
        bg_corr_layout.addWidget(QLabel("Image name"))
        bg_corr_layout.addWidget(self.bg_corr_image_name)
        # bg_corr_layout.addWidget(bg_corr_confirm_button)
        bg_corr_layout.addWidget(self.bg_corr_select_button)
        bg_corr_layout.addWidget(self.bg_corr_image_preview)
        bg_corr_layout.addStretch()

        # Adding Pixel Resolution and Background Correction frames to top layout
        top_frame_layout.addWidget(pixel_img_frame)
        top_frame_layout.addWidget(bg_corr_frame)

        # Create bottom frame for manual calibration input
        bottom_frame = QFrame()
        bottom_frame_layout = QVBoxLayout()
        bottom_frame.setLayout(bottom_frame_layout)

        manualcalibration_frame = QFrame()
        manualcalibration_layout = QHBoxLayout(manualcalibration_frame)
        manualcalibration_layout.addWidget(QLabel("or:"))
        self.manual_px_mm_input = QLineEdit()
        self.manual_px_mm_input.setPlaceholderText("Calibrate manually")
        manualcalibration_layout.addWidget(self.manual_px_mm_input)
        manualcalibration_layout.addWidget(QLabel("px/mm"))

        self.confirm_px_mm_button = QPushButton("Confirm calibration and background image")
        # self.confirm_px_mm_button.clicked.connect(
        #     self.main_handler.tab2_confirm_calibration
        # )  # Connect confirm button to the handler

        bottom_frame_layout.addWidget(manualcalibration_frame)
        bottom_frame_layout.addWidget(self.confirm_px_mm_button)

        # Add frames to main layout
        layout.addWidget(top_frame, 0, 0, 1, 2)
        layout.addWidget(bottom_frame, 1, 0, 1, 2)

    def setup_image_processing_tab(self) -> None:
        """Set up the image processing tab, which contains the following components.

        Set up the image processing tab, which contains the following components:

        1. A box to select the image processing algorithm.
        2. A table to display and edit the processing parameters.
        3. A button to confirm the parameter settings and preview a sample image.
        4. A button to batch process the sample images.
        5. A section to display the sample image preview.
        6. A section to display the processed image preview, including before and after
        filtering.

        When the user selects an algorithm, the parameters will be loaded and displayed
        in the table. When the user confirms the parameter settings and previews a
        sample image, the sample image will be processed using the selected algorithm
        and displayed in the preview section.
        When the user clicks the "Batch process images" button, the algorithm will be
        applied to all the images in the selected folder and the results will be saved
        in a new folder.
        """
        layout = QGridLayout(self.image_processing_tab)

        # ----------- First Column: Sample Image Preview -----------

        first_column_frame = QFrame()
        first_column_layout = QVBoxLayout(first_column_frame)

        # Sample Image Preview Canvas
        self.sample_image_preview = ZoomablePreviewLabel("Sample image preview", viewer_title="Sample image")
        self.sample_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_image_preview.setFixedSize(400, 300)  # Adjust size as needed
        self.prev_button = QPushButton("< Prev. Img")
        self.next_button = QPushButton("Next Img >")
        # self.prev_button.clicked.connect(lambda: self.main_handler.tab3_update_sample_image("prev"))
        # self.next_button.clicked.connect(lambda: self.main_handler.tab3_update_sample_image("next"))
        self.preview_processed_images_button = QPushButton("Preview Processed Images")
        self.preview_processed_images_button.setEnabled(False)
        # self.preview_processed_images_button.clicked.connect(self.main_handler.tab3_preview_processed_images)

        first_column_layout.addWidget(QLabel("Select image and preview"))
        first_column_layout.addWidget(self.sample_image_preview)

        # Add filename and index display label
        self.sample_image_info_label = QLabel("No image selected")
        self.sample_image_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_image_info_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        first_column_layout.addWidget(self.sample_image_info_label)

        # Prev/Next Buttons
        first_column_buttons_layout = QHBoxLayout()
        first_column_buttons_layout.addWidget(self.prev_button)
        first_column_buttons_layout.addWidget(self.next_button)

        first_column_layout.addLayout(first_column_buttons_layout)
        first_column_layout.addWidget(self.preview_processed_images_button)

        # ----------- Second Column: Processed Image Before Filtering and Sandbox ----

        second_column_frame = QFrame()
        second_column_layout = QVBoxLayout(second_column_frame)

        # Processed Image Before Filtering Canvas
        self.label_before_filtering = MplCanvas(self, width=5, height=4, dpi=100)
        second_column_layout.addWidget(QLabel("Step 1: Bubble Segmentation"))
        second_column_layout.addWidget(self.label_before_filtering)

        algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_combo = QComboBox()

        # Parameter sandbox for img_resample_factor, threshold_value, element_size
        sandbox1_label = QLabel("Adjust parameters before filtering")

        second_column_layout.addWidget(algorithm_label)
        second_column_layout.addWidget(self.algorithm_combo)
        second_column_layout.addWidget(sandbox1_label)

        self.param_sandbox1 = QTableWidget(0, 2)
        self.param_sandbox1.setHorizontalHeaderLabels(["Parameter", "Value"])

        # Confirm button for this sandbox
        self.preview_button1 = QPushButton("Confirm parameter and preview")
        second_column_layout.addWidget(sandbox1_label)
        second_column_layout.addWidget(self.param_sandbox1)
        second_column_layout.addWidget(self.preview_button1)

        # Algorithm Description Label
        self.algorithm_description_label = QLabel("")
        self.algorithm_description_label.setWordWrap(True)
        self.algorithm_description_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.algorithm_description_label.setStyleSheet("color: #666; font-style: italic; margin-top: 10px;")
        second_column_layout.addWidget(self.algorithm_description_label)
        second_column_layout.addStretch()

        # ----------- Third Column: Processed Image After Filtering and Sandbox ------

        third_column_frame = QFrame()
        third_column_layout = QVBoxLayout(third_column_frame)

        # Processed Image After Filtering Canvas
        self.processed_image_preview = MplCanvas(self, width=5, height=4, dpi=100)
        third_column_layout.addWidget(QLabel("Step 2: Filtering and Ellipse Fitting"))
        third_column_layout.addWidget(self.processed_image_preview)

        # Parameter sandbox for max_eccentricity, min_solidity, min_size
        sandbox2_label = QLabel("Adjust parameters for filtering circles")
        self.param_sandbox2 = QTableWidget(0, 2)
        self.param_sandbox2.setHorizontalHeaderLabels(["Parameter", "Value"])

        # Confirm and Batch Process buttons for this sandbox
        self.preview_button2 = QPushButton("Confirm parameter and preview")
        self.manual_adjustment_button = QPushButton("Manual adjustment")
        # self.manual_adjustment_button.clicked.connect(self.main_handler.tab3_ellipse_manual_adjustment)
        self.batch_process_button = QPushButton("Batch process images")
        self.finalise_analysis_button = QPushButton("Finalise analysis")
        self.finalise_analysis_button.setStyleSheet("background-color: red; color: white;")
        # self.preview_button2.clicked.connect(self.main_handler.tab3_confirm_parameter_for_filtering)
        # self.batch_process_button.clicked.connect(self.main_handler.tab3_ask_if_batch)

        third_column_layout.addWidget(sandbox2_label)
        third_column_layout.addWidget(self.param_sandbox2)
        third_column_layout.addWidget(self.preview_button2)
        third_column_layout.addWidget(self.manual_adjustment_button)
        third_column_layout.addWidget(self.batch_process_button)
        third_column_layout.addWidget(self.finalise_analysis_button)

        # Add the columns to the main layout
        layout.addWidget(first_column_frame, 0, 0)
        layout.addWidget(second_column_frame, 0, 1)
        layout.addWidget(third_column_frame, 0, 2)

    def setup_results_tab(self) -> None:
        """Set up the results tab.

        This function sets up the results tab by creating the following widgets:

        1. A graph canvas for displaying the histogram.
        2. Controls for histogram options, including the type of histogram to
        generate (by number or volume), checkboxes for PDF and CDF, the number
        of bins, and the x-axis limits.
        3. A legend position.
        4. A descriptive size options section, which includes checkboxes for
        displaying d32, dmean, and dxy, as well as input boxes for x and y
        values for dxy.
        5. A save button that saves the graph and CSV data to a user-selected
        folder.

        This function creates the layout of the results tab, including the graph
        canvas, controls, and save button. The controls include the histogram type,
        PDF/CDF checkboxes, number of bins, x-axis limits, legend position and
        orientation dropdown, and descriptive size options. The save button is
        connected to the save_results slot, which saves the graph and CSV data to
        the user-selected folder.
        """
        layout = QGridLayout(self.results_tab)

        # Create canvas for displaying the graph
        self.histogram_canvas = MplCanvas(self, width=8, height=8, dpi=100)

        # Controls for histogram options
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)

        # Histogram type
        histogram_by_label = QLabel("Histogram by:")
        self.histogram_by = QComboBox()
        self.histogram_by.addItems(["Count", "Volume"])

        # Connect to auto-update
        self.options_label = QLabel("Histogram Options:")

        # PDF/CDF Checkboxes
        self.pdf_checkbox = QCheckBox("PDF")
        self.cdf_checkbox = QCheckBox("CDF")

        # Number of bins
        bins_label = QLabel("Number of bins:")
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setValue(15)
        self.bins_spinbox.setRange(1, 100)

        # X-axis limits
        x_axis_limits_label = QLabel("X-axis limits:")
        self.min_x_axis_input = QLineEdit("0.0")
        self.max_x_axis_input = QLineEdit("5.0")

        legend_label = QLabel("Legend settings:")
        legend_frame = QFrame()
        legend_layout = QGridLayout(legend_frame)

        # Legend position dropdown
        legend_layout.addWidget(QLabel("Position:"), 0, 0)
        self.legend_position_combobox = QComboBox()
        self.legend_position_combobox.addItems(["North East", "North West", "South East", "South West"])
        legend_layout.addWidget(self.legend_position_combobox, 0, 1)

        # Descriptive size options
        # Descriptive Size Checkboxes Section
        descriptive_frame = QFrame()
        descriptive_layout = QGridLayout(descriptive_frame)

        self.d32_checkbox = QCheckBox("d32")
        self.dmean_checkbox = QCheckBox("d mean")
        self.dxy_checkbox = QCheckBox("dxy")

        # Add input boxes for `x` and `y` values
        self.dxy_x_input = QLineEdit()
        self.dxy_x_input.setText("5")  # Set default value for x
        self.dxy_x_input.setMaximumWidth(40)

        self.dxy_y_input = QLineEdit()
        self.dxy_y_input.setText("4")  # Set default value for y
        self.dxy_y_input.setMaximumWidth(40)

        # Add elements to the descriptive layout
        descriptive_layout.addWidget(self.d32_checkbox, 0, 0)
        descriptive_layout.addWidget(self.dmean_checkbox, 1, 0)
        descriptive_layout.addWidget(self.dxy_checkbox, 2, 0)
        descriptive_layout.addWidget(QLabel("x"), 2, 1)
        descriptive_layout.addWidget(self.dxy_x_input, 2, 2)
        descriptive_layout.addWidget(QLabel("y"), 2, 3)
        descriptive_layout.addWidget(self.dxy_y_input, 2, 4)

        # Save button
        save_frame = QFrame()
        save_layout = QVBoxLayout(save_frame)

        # Save button
        self.save_button = QPushButton("Save Results (Excel + Graph)")

        # Export annotated images button
        self.export_images_button = QPushButton("Export Annotated Images")

        # Export the corrected segmentations as a dataset (format chosen in a dialog)
        self.export_ml_button = QPushButton("Export Dataset (choose format)")

        # Restart button (new addition)
        self.restart_button = QPushButton("Restart New Mission")
        self.restart_button.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")

        # Add folder selection and save button to the layout
        save_layout.addWidget(self.save_button)
        save_layout.addWidget(self.export_images_button)
        save_layout.addWidget(self.export_ml_button)
        save_layout.addWidget(self.restart_button)  # Add restart button under save button

        # Assemble controls layout
        controls_layout.addWidget(histogram_by_label)
        controls_layout.addWidget(self.histogram_by)
        controls_layout.addWidget(self.options_label)
        controls_layout.addWidget(self.pdf_checkbox)
        controls_layout.addWidget(self.cdf_checkbox)
        controls_layout.addWidget(bins_label)
        controls_layout.addWidget(self.bins_spinbox)
        controls_layout.addWidget(x_axis_limits_label)
        controls_layout.addWidget(self.min_x_axis_input)
        controls_layout.addWidget(self.max_x_axis_input)
        controls_layout.addWidget(legend_label)
        controls_layout.addWidget(legend_frame)
        controls_layout.addWidget(descriptive_frame)
        controls_layout.addWidget(save_frame)

        # Place graph and controls in the layout
        layout.addWidget(self.histogram_canvas, 0, 0)
        layout.addWidget(controls_frame, 0, 1)

        # Add a label to display the descriptive sizes
        self.descriptive_size_label = QLabel("")
        layout.addWidget(self.descriptive_size_label, 2, 0, 1, 2)
