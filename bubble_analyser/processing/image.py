"""Image processing module for the Bubble Analyser application.

This module provides classes for loading, processing, and analyzing images of bubbles.
It includes functionality for dynamic method loading, image preprocessing, segmentation,
filtering, and ellipse detection and measurement.
"""

import importlib.util
import inspect
import logging
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy import typing as npt

if TYPE_CHECKING:
    from bubble_analyser.methods.watershed_methods import IterativeWatershed, NormalWatershed

from bubble_analyser.processing.circle_handler import EllipseHandler as CircleHandler
from bubble_analyser.processing.config import Config
from bubble_analyser.processing.image_preprocess import image_preprocess


class MethodsHandler:
    """A handler class for dynamically loading and managing method modules.

    This class is responsible for loading Python modules from a specified folder, extracting classes from these modules,
    instantiating them with a given configuration, and aggregating their required parameters into a dictionary.

    Attributes:
        params_dict (dict): A dictionary representation of the configuration parameters obtained via the
            `model_dump()` method.
        folder_path (Path): The file system path to the folder containing the method modules.
        modules (dict[str, object]): A dictionary mapping module names to their corresponding loaded module objects.
        all_classes (dict[str, object]): A dictionary mapping instance names (from each module's class instance) to
            the instantiated objects.
        full_dict (dict[str, dict[str, float|int]]): A dictionary mapping instance names to their required parameters.
    """

    def __init__(self, params: Config) -> None:
        """Initialize a MethodsHandler instance with the provided configuration.

        This method sets up the internal parameters, loads the modules from the methods package,
        instantiates classes from those modules, and collects the necessary parameters from each instance.

        Args:
            params (Config): A configuration object that includes the necessary parameters. The configuration is
                converted to a dictionary using its `model_dump()` method.

        Side Effects:
            - Populates the `params_dict`, `modules`, `all_classes`, and `full_dict` attributes.
            - Loads modules and prints module and class information to the console.
        """
        self.params_dict = params.model_dump()

        self.modules: dict[str, types.ModuleType] = {}
        self.modules = self.load_modules_from_package()

        self.all_classes: dict[str, object] = {}

        self.full_dict: dict[str, dict[str, float | int]] = {}
        self._get_full_dict()

    def load_modules_from_package(self) -> dict[str, "types.ModuleType"]:
        """Load Python modules from the methods package using importlib.resources.

        Uses importlib.resources to access the methods package, identifies Python files,
        dynamically imports each file as a module, and stores the module objects in a dictionary
        keyed by the module name (derived from the file name).

        Returns:
            dict[str, object]: A dictionary where each key is a module name and each value is the corresponding module
                object.
        """
        modules = {}
        package_name = "bubble_analyser.methods"

        if getattr(sys, "frozen", False):
            logging.info("Running in frozen mode, loading modules explicitly")
            try:
                from bubble_analyser.methods import bubmask_method, peak_watershed, watershed_methods, yolo_method

                modules["watershed_methods"] = watershed_methods
                modules["bubmask_method"] = bubmask_method
                modules["peak_watershed"] = peak_watershed
                modules["yolo_method"] = yolo_method
            except ImportError as e:
                logging.error(f"Error loading modules in frozen mode: {e}")
            return modules

        try:
            logging.info(f"Loading modules from package: {package_name}")
            # For Python 3.9+
            if sys.version_info >= (3, 9):  # type: ignore # noqa: UP036
                from importlib.resources import files

                package_path = files(package_name)
                # Use a more cross-platform approach instead of glob()
                try:
                    # For directories

                    for file_path in package_path.iterdir():  # type: ignore
                        logging.info(f"Found file: {file_path}")

                        if file_path.is_file() and file_path.name.endswith(".py") and file_path.name != "__init__.py":
                            module_name = str(file_path).rsplit(".", 1)[0].split("/")[-1]
                            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                            if spec is not None:
                                logging.info(f"Loading module: {module_name} from {file_path}")
                                module = importlib.util.module_from_spec(spec)
                                if spec.loader is not None:
                                    spec.loader.exec_module(module)
                                    modules[module_name] = module

                except (TypeError, AttributeError):
                    # For MultiplexedPath or other path types that don't support iterdir
                    from importlib.resources import contents

                    for resource in contents(package_name):
                        if resource.endswith(".py") and resource != "__init__.py":
                            module_name = resource[:-3]  # Remove .py extension
                            module_path = f"{package_name}.{module_name}"
                            modules[module_name] = importlib.import_module(module_path)

        except Exception as e:
            print(f"Error loading modules: {e}")
            # Fallback to direct import of watershed_methods
            try:
                from ..methods import watershed_methods

                modules["watershed_methods"] = watershed_methods
            except Exception as inner_e:
                logging.info(f"Fallback import failed: {inner_e}")

        return modules

    def get_new_classes(self, module: object) -> dict[str, object]:  # type: ignore
        """Retrieve classes that are defined within the provided module.

        This method uses the `inspect` module to iterate over members of the module, filtering out only those that are
        classes and verifying that they were defined in the module itself (by comparing the class's `__module__`
        attribute to the module's name).

        Args:
            module (module): A Python module object from which to extract classes. If a falsy value is provided,
                             the method defaults to using the modules stored in the `self.modules` attribute.

        Returns:
            dict[str, object]: A dictionary mapping class names to their corresponding class objects defined in the
                module.
        """
        if not module:
            module = self.modules

        new_classes = {}
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if the class is defined in the current module by comparing module names
            if isinstance(module, type(importlib)):
                module_name = module.__spec__.name  # type: ignore
                if hasattr(obj, "__module__") and obj.__module__ == module_name:
                    new_classes[name] = obj
        return new_classes  # type: ignore

    def _get_full_dict(self) -> None:
        """Instantiate classes from loaded modules and build dictionaries of instances and their parameters.

        Iterates through each module loaded in `self.modules`, retrieves the classes defined in the module,
        instantiates each class using the configuration dictionary (`self.params_dict`), and then:
          - Stores the instance in `self.all_classes` using its `name` attribute as the key.
          - Retrieves the necessary parameters via the instance's `get_needed_params()` method and stores
            them in `self.full_dict`.

        Side Effects:
            - Populates the `all_classes` and `full_dict` attributes with the instances and their parameters.
        """
        for module_name, module in self.modules.items():
            logging.debug(f"Module: {module_name}")
            new_classes = self.get_new_classes(module)
            for class_name, class_obj in new_classes.items():
                logging.debug(f"  Class: {class_name}")

                instance: IterativeWatershed | NormalWatershed = class_obj(self.params_dict)  # type: ignore
                self.all_classes[instance.name] = instance
                self.full_dict[instance.name] = instance.get_needed_params()  # type: ignore

        # Reorder dictionary to put "Default" method first if it exists
        if "Default" in self.full_dict:
            default_value = self.full_dict["Default"]
            del self.full_dict["Default"]
            self.full_dict = {"Default": default_value, **self.full_dict}

        logging.info("All methods achieved and their needed params:", self.full_dict)


class Image:
    """A class representing an image and its associated processing routines.

    This class encapsulates the functionality to preprocess an image,
    apply filtering and segmentation using various algorithms, and handle
    post-processing tasks such as label filtering and ellipse detection. It
    integrates with external processing methods via a provided methods handler,
    and supports optional background image processing.

    Attributes:
        filter_param_dict (dict[str, float]): Dictionary of filtering parameters (e.g., "max_eccentricity",
                                                "min_solidity", "min_size").
        px2mm (float): Conversion factor from pixels to millimeters.
        raw_img_path (Path): File path to the raw image.
        bknd_img_path (Path, optional): File path to the background image (if provided).
        bknd_img (npt.NDArray[np.int_], optional): Array representing the background image.
        img_rgb (npt.NDArray[np.int_]): RGB version of the processed image.
        img_grey (npt.NDArray[np.int_]): Grayscale version of the processed image.
        labels_on_img_before_filter (npt.NDArray[np.int_]): Processed image labels overlay before filtering.
        labels_before_filter (npt.NDArray[np.int_]): Labels extracted from the image before filtering.
        labels_after_filter (npt.NDArray[np.int_]): Labels data after the filtering process.
        ellipses (list[RotatedRect]): List of detected ellipse regions.
        ellipses_properties (list[dict[str, float]]): List containing properties of each detected ellipse.
        ellipses_on_images (npt.NDArray[np.int_]): Image data with ellipses overlaid.
        all_methods_n_params (dict[str, dict[str, float|int]]): Dictionary mapping method names to their parameters.
        methods_handler (MethodsHandler): Instance of MethodsHandler that provides processing methods.
        new_normal_watershed (NormalWatershed): Instance of the NormalWatershed processing method.
        new_iterative_watershed (IterativeWatershed): Instance of the IterativeWatershed processing method.
        new_circle_handler (CircleHandler): Instance of CircleHandler for ellipse processing.
        if_fine_tuned (bool): Flag indicating whether ellipses have been manually fine-tuned.
    """

    def __init__(
        self,
        px2mm_display: float,
        raw_img_path: Path,
        all_methods_n_params: dict[str, dict[str, float | int]],
        methods_handler: MethodsHandler,
        bknd_img_path: Path | None = None,
    ) -> None:
        """Initialize an Image instance with the specified parameters and processing handler.

        Sets up the image attributes including conversion factors, file paths, and initial
        processing parameters. If a background image path is provided, it sets the appropriate flag.
        Additionally, it stores the provided methods and parameters for subsequent processing.

        Args:
            px2mm_display (float): Conversion factor from pixels to millimeters for display purposes.
            raw_img_path (Path): File path to the raw image to be processed.
            all_methods_n_params (dict): Dictionary mapping algorithm names to their processing parameters.
            methods_handler (MethodsHandler): An instance of MethodsHandler to access processing routines.
            bknd_img_path (Path, optional): File path to the background image. Defaults to None.
        """
        self.filter_param_dict: dict[str, float | str]

        self.px2mm_display: float = px2mm_display
        self.resample = 0.5
        self.target_width: int = 800
        self.raw_img_path = raw_img_path
        logging.info(f"Initialzing image: {raw_img_path}")
        self.if_bknd_img: bool = False
        self.bknd_img_path: Path | None = bknd_img_path
        self.bknd_img: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], None)

        if bknd_img_path is not None:
            logging.info(f"bknd_img_path: {bknd_img_path}")
            self.if_bknd_img = True
            # self.bknd_img_path = bknd_img_path

        self.img_rgb: npt.NDArray[np.int_]
        self.img_grey: npt.NDArray[np.int_]
        self.img_grey_morph_eroded: npt.NDArray[np.int_]

        self.labels_on_img_before_filter: npt.NDArray[np.int_]
        self.labels_before_filter: npt.NDArray[np.int_]
        self.labels_after_filter: npt.NDArray[np.int_]
        self.labelled_ellipses_mask: npt.NDArray[np.int_]
        self.ellipses: list[tuple[tuple[float, float], tuple[int, int], float]] = []
        self.ellipses_properties: list[dict[str, float | str]]
        self.ellipses_on_images: npt.NDArray[np.int_]

        self.all_methods_n_params: dict[str, dict[str, float | int]] = all_methods_n_params

        self.methods_handler: MethodsHandler = methods_handler

        self.new_normal_watershed: NormalWatershed
        self.new_iterative_watershed: IterativeWatershed
        self.new_circle_handler: CircleHandler = None  # type: ignore
        self.if_fine_tuned: bool = False

    def load_filter_params(self, dict_params_1: dict[str, float | str]) -> None:
        """Load and update filtering parameters for the image.

        Args:
            dict_params_1 (dict[str, float]): Dictionary containing filtering parameters (e.g.,
                                              "max_eccentricity", "min_solidity", "min_size").

        Returns:
            None
        """
        self.filter_param_dict_1 = dict_params_1
        return

    def _img_preprocess(self, resample: float) -> None:
        """Preprocess the raw and background images by resampling and converting to grayscale and RGB formats.

        Uses an external function `image_preprocess` to generate both a grayscale and an RGB version of the raw image.
        If a background image path is provided, the background image is processed similarly.

        Args:
            resample (float): Resampling factor used during image preprocessing.

        Returns:
            None
        """
        # Get resized grey and RGB version of the target image
        self.img_grey, self.img_rgb = image_preprocess(self.raw_img_path, resample)
        if self.bknd_img_path is not None:
            self.bknd_img, _ = image_preprocess(self.bknd_img_path, resample)

        return

    def processing_image_before_filtering(self, algorithm: str, cnn_model: object) -> None:
        """Process the image using a specified algorithm prior to filtering.

        Iterates through the available methods and their parameters, and if a match is found for the given
        algorithm, preprocesses the image and invokes the corresponding processing routine to obtain initial label data.

        Args:
            algorithm (str): The name of the algorithm to use for processing the image.
            cnn_model (object): Pre-initialized CNN model.

        Returns:
            None
        """
        for algorithm_name, params in self.all_methods_n_params.items():
            if algorithm_name == algorithm:
                for (
                    name,
                    processing_instance,
                ) in self.methods_handler.all_classes.items():
                    if name == algorithm_name:
                        # self.target_width = cast(int, params["target_width"])
                        self.resample = params["resample"]  # type: ignore
                        self._img_preprocess(self.resample)
                        logging.info(f"resample used for preprocess: {self.resample}")
                        # processing_instance
                        if name == "BubMask (Deep Learning)":
                            processing_instance.initialize_processing(  # type: ignore
                                params=params,
                                img_grey=self.img_grey,
                                img_rgb=self.img_rgb,
                                if_bknd_img=self.if_bknd_img,
                                px2mm=self.px2mm_display,
                                bknd_img=self.bknd_img,
                                cnn_model=cnn_model,
                            )  # type: ignore
                        else:
                            processing_instance.initialize_processing(  # type: ignore
                                params=params,
                                img_grey=self.img_grey,
                                img_rgb=self.img_rgb,
                                if_bknd_img=self.if_bknd_img,
                                px2mm=self.px2mm_display,
                                bknd_img=self.bknd_img,
                            )  # type: ignore
                        self.labels_on_img_before_filter, self.labels_before_filter, self.img_grey_morph_eroded = (
                            processing_instance.get_results_img()  # type: ignore
                        )
                break

    def initialize_circle_handler(self) -> None:
        """Initialize the circle handler for ellipse detection and filtering.

        Creates a new instance of CircleHandler using copies of the pre-filtered labels and RGB image,
        and loads the current filtering parameters into the handler.

        Returns:
            None
        """
        labels_before_filter = self.labels_before_filter.copy()
        rgb_img = self.img_rgb.copy()

        if self.new_circle_handler is not None:
            del self.new_circle_handler

        self.new_circle_handler = CircleHandler(
            labels_before_filter, rgb_img, self.px2mm_display, resample=self.resample
        )
        self.new_circle_handler.load_filter_params(self.filter_param_dict_1)

    def labels_filtering(self) -> None:
        """Filter the image labels using the circle handler's filtering algorithm.

        Updates the `labels_after_filter` attribute with the results of the filtering process.

        Returns:
            None
        """
        self.labels_after_filter = self.new_circle_handler.filter_labels_properties()

    def fill_ellipses(self) -> None:
        """Detect and fill ellipse regions based on the filtered labels.

        Uses the circle handler to generate a list of ellipses that correspond to detected regions,
        and updates the `ellipses` attribute accordingly.

        Returns:
            None
        """
        self.ellipses = self.new_circle_handler.fill_ellipse_labels()

    def update_ellipses(self, ellipses: list[tuple[tuple[float, float], tuple[int, int], float]]) -> None:
        """Update the detected ellipses with a manually provided list of ellipses.

        This allows for manual fine-tuning of ellipse detection. The method updates both the instance's
        ellipse list and the circle handler's ellipse data, and sets a flag indicating that fine-tuning has occurred.

        Args:
            ellipses (list[RotatedRect]): A list of RotatedRect objects representing the updated ellipses.

        Returns:
            None
        """
        self.ellipses = ellipses
        self.new_circle_handler.ellipses = ellipses
        self.set_fine_tuned()

    def set_fine_tuned(self) -> None:
        self.if_fine_tuned = True

    def get_ellipse_properties(self) -> None:
        """Calculate and retrieve properties of the detected ellipses.

        Utilizes the circle handler's functionality to compute various properties (e.g., size, orientation)
        for each detected ellipse, and stores the results in the `ellipses_properties` attribute.

        Returns:
            None
        """
        self.ellipses_properties = self.new_circle_handler.calculate_circle_properties()  # type: ignore
        for ellipse in self.ellipses_properties:
            ellipse["filename"] = self.raw_img_path.name

    def overlay_ellipses_on_images(self) -> None:
        """Overlay the detected ellipses onto the RGB image.

        Uses the circle handler to superimpose the ellipse outlines on the image,
        and updates the `ellipses_on_images` attribute with the resultant image.

        Returns:
            None
        """
        self.ellipses_on_images = self.new_circle_handler.overlay_ellipses_on_image()

    def get_labelled_mask(self) -> None:
        """Create a labeled mask image from the detected ellipses.

        This method uses the circle handler to generate a mask where each detected ellipse
        is represented with a unique label. The result is stored in the labelled_ellipses_mask
        attribute for later use in visualization or analysis.

        Returns:
            None
        """
        self.labelled_ellipses_mask = self.new_circle_handler.create_labelled_image_from_ellipses()

    def filtering_processing(self) -> None:
        """Execute the complete filtering process on the image.

        This method runs the full sequence of steps necessary for filtering:
          1. Initializing the circle handler.
          2. Filtering labels.
          3. Detecting and filling ellipses.
          4. Overlaying ellipses on the image.
          5. Calculating ellipse properties.

        Returns:
            None
        """
        self.initialize_circle_handler()
        self.labels_filtering()
        self.fill_ellipses()
        self.overlay_ellipses_on_images()
        self.get_ellipse_properties()
        self.get_labelled_mask()
