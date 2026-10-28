from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from bubble_analyser.core.models import ImageState
from bubble_analyser.gui.component_handlers import ImageProcessingModel
from bubble_analyser.gui.event_handlers import (
    ImageProcessingTabHandler,
    ResultsTabHandler,
)
from bubble_analyser.gui.gui import MainWindow
from bubble_analyser.processing import Config


def test_manual_corrections_synced_to_master_properties(tmp_path: Path):
    _ = QApplication.instance() or QApplication([])
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    cv2.imwrite(str(img1), np.full((100, 100, 3), 200, dtype=np.uint8))
    cv2.imwrite(str(img2), np.full((100, 100, 3), 200, dtype=np.uint8))

    model = ImageProcessingModel(Config())
    model.img_path_list = [img1, img2]

    # Simulate batch processing having found 1 bubble in img1 and 1 in img2
    model.ellipses_properties = [
        [{"equivalent_diameter": 2.0, "filename": "img1.png"}],
        [{"equivalent_diameter": 4.0, "filename": "img2.png"}],
    ]

    state1 = ImageState(img1, px2mm_display=10.0)
    state1.img_rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    state1.ellipses = [((50.0, 50.0), (20.0, 20.0), 0.0)]
    state1.ellipses_properties = [{"equivalent_diameter": 2.0, "filename": "img1.png"}]
    model.img_dict[img1] = state1

    state2 = ImageState(img2, px2mm_display=10.0)
    state2.img_rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    state2.ellipses = [((50.0, 50.0), (40.0, 40.0), 0.0)]
    state2.ellipses_properties = [{"equivalent_diameter": 4.0, "filename": "img2.png"}]
    model.img_dict[img2] = state2

    # User manually fine-tunes img1: adds a second bubble, making 2 bubbles in img1
    model.adjuster = MagicMock()
    model.adjuster.ellipses = [
        ((30.0, 30.0), (10.0, 10.0), 0.0),
        ((70.0, 70.0), (15.0, 15.0), 0.0),
    ]
    model.handle_ellipse_adjustment_finished(state1)

    # Verify state1 is marked fine-tuned
    assert state1.if_fine_tuned is True
    assert len(state1.ellipses) == 2
    assert len(state1.ellipses_properties) == 2

    # Verify model.ellipses_properties updated
    assert len(model.ellipses_properties[0]) == 2
    assert len(model.ellipses_properties[1]) == 1
    assert model.bubble_count == 3

    # Verify ResultsTabHandler gets updated diameters
    results_handler = ResultsTabHandler(model, Config())
    gui = MainWindow()
    results_handler.load_gui(gui)
    results_handler.load_ellipse_properties(
        properties=model.ellipses_properties,
        algorithm="Watershed",
        all_methods_n_params={},
        param_dict_1={},
        px2mm_display=10.0,
    )

    diameters = results_handler.get_equivalent_diameters_list()
    assert len(diameters) == 3


def test_fine_tuned_zero_bubbles_retained(tmp_path: Path):
    """If user deletes all bubbles on an image, the zero-bubble state must be counted."""
    _ = QApplication.instance() or QApplication([])
    img1 = tmp_path / "img1.png"
    cv2.imwrite(str(img1), np.full((50, 50, 3), 150, dtype=np.uint8))

    model = ImageProcessingModel(Config())
    model.img_path_list = [img1]
    model.ellipses_properties = [[{"equivalent_diameter": 3.0, "filename": "img1.png"}]]

    state1 = ImageState(img1, px2mm_display=10.0)
    state1.img_rgb = np.full((50, 50, 3), 150, dtype=np.uint8)
    state1.ellipses = [((25.0, 25.0), (10.0, 10.0), 0.0)]
    model.img_dict[img1] = state1

    # User manually fine-tunes and deletes all ellipses
    model.adjuster = MagicMock()
    model.adjuster.ellipses = []
    model.handle_ellipse_adjustment_finished(state1)

    assert state1.if_fine_tuned is True
    assert len(state1.ellipses) == 0
    assert len(state1.ellipses_properties) == 0
    assert model.bubble_count == 0
    assert model.ellipses_properties == [[]]


def test_get_or_render_annotated_image(tmp_path: Path):
    _ = QApplication.instance() or QApplication([])
    img_path = tmp_path / "test_frame.png"
    cv2.imwrite(str(img_path), np.full((100, 100, 3), 255, dtype=np.uint8))

    model = ImageProcessingModel(Config())
    model.img_path_list = [img_path]

    state = ImageState(img_path, px2mm_display=10.0)
    state.img_rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
    state.ellipses = [((50.0, 50.0), (20.0, 20.0), 0.0)]
    # Intentionally leave ellipses_on_images empty
    state.ellipses_on_images = np.zeros((0, 0, 3), dtype=np.uint8)
    model.img_dict[img_path] = state

    # Calling get_or_render_annotated_image should render the ellipses
    rendered = model.get_or_render_annotated_image(img_path)
    assert rendered is not None
    assert rendered.shape == (100, 100, 3)
    assert rendered.dtype == np.uint8
    # Center should not be all 255 because of drawn ellipse overlay
    assert not np.all(rendered == 255)


def test_export_annotated_images_writes_files(monkeypatch, tmp_path: Path):
    _ = QApplication.instance() or QApplication([])
    img1 = tmp_path / "img1.png"
    cv2.imwrite(str(img1), np.full((60, 60, 3), 200, dtype=np.uint8))

    export_dir = tmp_path / "exported"
    export_dir.mkdir()

    model = ImageProcessingModel(Config())
    model.img_path_list = [img1]

    state1 = ImageState(img1, px2mm_display=10.0)
    state1.img_rgb = np.full((60, 60, 3), 200, dtype=np.uint8)
    state1.ellipses = [((30.0, 30.0), (10.0, 10.0), 0.0)]
    model.img_dict[img1] = state1

    results_handler = ResultsTabHandler(model, Config())
    gui = MainWindow()
    results_handler.load_gui(gui)

    monkeypatch.setattr(
        "bubble_analyser.gui.event_handlers.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(export_dir),
    )
    monkeypatch.setattr(
        "bubble_analyser.gui.event_handlers.QMessageBox.information",
        lambda *args, **kwargs: None,
    )

    results_handler.export_annotated_images()

    exported_files = list(export_dir.glob("*.png"))
    assert len(exported_files) == 1
    assert exported_files[0].name == "img1_annotated.png"


def test_finalise_analysis_checkbox_save_annotated(monkeypatch, tmp_path: Path):
    _ = QApplication.instance() or QApplication([])
    img1 = tmp_path / "frame1.png"
    cv2.imwrite(str(img1), np.full((80, 80, 3), 180, dtype=np.uint8))

    export_dir = tmp_path / "finalise_export"
    export_dir.mkdir()

    model = ImageProcessingModel(Config())
    model.img_path_list = [img1]
    model.if_batched = True

    state1 = ImageState(img1, px2mm_display=10.0)
    state1.img_rgb = np.full((80, 80, 3), 180, dtype=np.uint8)
    state1.ellipses = [((40.0, 40.0), (15.0, 15.0), 0.0)]
    model.img_dict[img1] = state1

    tab_handler = ImageProcessingTabHandler(model, Config())
    gui = MainWindow()
    tab_handler.load_gui(gui)

    # Simulate checking the save images checkbox
    tab_handler.if_save_processed_images = True

    # Mock dialog execution to accept
    class MockDialog:
        def __init__(self, *args, **kwargs):
            pass

        def setCheckBox(self, cb):
            pass

        def exec(self):
            return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(tab_handler, "create_confirm_dialog", lambda *args, **kwargs: MockDialog())
    monkeypatch.setattr(
        "bubble_analyser.gui.event_handlers.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(export_dir),
    )

    tab_handler.finalise_analysis()

    exported = list(export_dir.glob("*.png"))
    assert len(exported) == 1
    assert exported[0].name == "frame1_annotated.png"
