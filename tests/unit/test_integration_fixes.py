"""Regression tests for the fixes applied on top of the YOLO, BubMask and export pull requests."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication, QComboBox, QTableWidget, QTableWidgetItem

from bubble_analyser.cnn_methods.bubmask_wrapper import BubMaskConfig, BubMaskDetector
from bubble_analyser.gui.event_handlers import ImageProcessingTabHandler
from bubble_analyser.methods.yolo_method import YOLOSegmentation
from bubble_analyser.processing.dataset_export import FrameAnnotation, export_dataset, render_label_mask


def test_dropdown_values_are_read_from_the_parameter_table():
    QApplication.instance() or QApplication([])
    table = QTableWidget(2, 2)
    table.setItem(0, 0, QTableWidgetItem("confidence_threshold"))
    table.setItem(0, 1, QTableWidgetItem("0.25"))
    table.setItem(1, 0, QTableWidgetItem("device"))
    combo = QComboBox()
    combo.addItems(["CPU", "GPU"])
    combo.setCurrentText("GPU")
    table.setCellWidget(1, 1, combo)

    handler = SimpleNamespace(convert_value=lambda text: ImageProcessingTabHandler.convert_value(None, text))  # type: ignore[arg-type]
    params = ImageProcessingTabHandler.extract_parameters_from_table(handler, table)  # type: ignore[arg-type]

    assert params == {"confidence_threshold": 0.25, "device": "GPU"}


def test_yolo_weights_are_loaded_once_across_images(tmp_path: Path, monkeypatch):
    weights = tmp_path / "bubbles.pt"
    weights.write_bytes(b"fake")
    loads: list[str] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            loads.append(path)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    method = YOLOSegmentation({})
    grey = np.zeros((8, 8), dtype=np.uint8)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    params = {"weights_path": str(weights)}
    for _ in range(3):
        method.initialize_processing(params, grey, rgb, False, 1.0, cnn_model=object())

    assert len(loads) == 1


def test_changing_bubmask_image_size_drops_the_loaded_model():
    detector = BubMaskDetector("unused.h5", BubMaskConfig(image_min_dim=384, image_max_dim=768))
    detector.model = object()  # type: ignore[assignment]  # stands in for a loaded Mask R-CNN

    detector.update_config(image_min_dim=384, image_max_dim=768, max_bubbles=500)
    assert detector.is_loaded

    detector.update_config(image_min_dim=512, image_max_dim=1024)
    assert not detector.is_loaded
    assert detector.config.image_min_dim == 512


def test_npy_masks_are_rebuilt_when_batch_processing_freed_them(tmp_path: Path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"not read for masks")
    frame = FrameAnnotation(
        image_path=image,
        ellipses=[((20.0, 20.0), (16.0, 16.0), 0.0), ((50.0, 40.0), (12.0, 20.0), 30.0)],
        annotation_shape=(80, 100),
        labelled_mask=np.zeros((0, 0), dtype=np.int32),  # what ImageState.clear_memory() leaves behind
    )

    rebuilt = render_label_mask(frame)
    assert rebuilt.shape == (80, 100)
    assert rebuilt[20, 20] == 2
    assert rebuilt[40, 50] == 3
    assert rebuilt[0, 0] == 1

    summary = export_dataset(tmp_path / "out", [frame], "masks_npy")
    saved = np.load(tmp_path / "out" / "masks" / "frame_mask.npy")

    assert summary.frames_exported == 1
    assert summary.instances_exported == 2
    assert np.array_equal(saved, rebuilt)
