import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog

from bubble_analyser.core.models import ImageState
from bubble_analyser.gui import event_handlers
from bubble_analyser.gui.component_handlers import ImageProcessingModel
from bubble_analyser.gui.event_handlers import ResultsTabHandler
from bubble_analyser.gui.export_dialog import ExportFormatDialog
from bubble_analyser.gui.gui import MainWindow
from bubble_analyser.processing import Config
from bubble_analyser.processing.dataset_export import DATASET_FORMATS


def test_dialog_offers_every_format_and_describes_the_selection():
    _ = QApplication.instance() or QApplication([])
    dialog = ExportFormatDialog()

    assert [dialog.format_combo.itemData(i) for i in range(dialog.format_combo.count())] == [
        item.key for item in DATASET_FORMATS
    ]
    assert dialog.selected_format().key == "yolo_seg"
    assert "labels" in dialog.description_label.text()

    dialog.format_combo.setCurrentIndex(dialog.format_combo.findData("coco"))

    assert dialog.selected_format().key == "coco"
    assert "annotations.json" in dialog.description_label.text()
    dialog.close()


def test_export_button_writes_the_chosen_format(monkeypatch, tmp_path: Path):
    _ = QApplication.instance() or QApplication([])
    image_path = tmp_path / "frames" / "f0001.png"
    image_path.parent.mkdir()
    cv2.imwrite(str(image_path), np.full((200, 400, 3), 100, dtype=np.uint8))

    model = ImageProcessingModel(Config())
    model.img_path_list = [image_path]
    state = ImageState(image_path, px2mm_display=1.0)
    state.img_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    state.ellipses_on_images = np.zeros((100, 200, 3), dtype=np.uint8)
    state.ellipses_on_images[0, 0] = 1
    state.ellipses = [((50.0, 50.0), (20.0, 10.0), 0.0)]  # type: ignore[list-item]
    state.ellipse_polygons = [None]
    state.ellipses_properties = [{"filename": image_path.name}]
    model.img_dict[image_path] = state

    handler = ResultsTabHandler(model, Config())
    gui = MainWindow()
    handler.load_gui(gui)

    class ChooseCoco:
        def __init__(self, *args, **kwargs):
            self.format = next(item for item in DATASET_FORMATS if item.key == "coco")

        def exec(self):
            return QDialog.DialogCode.Accepted

        def selected_format(self):
            return self.format

    shown = []
    out = tmp_path / "export"
    out.mkdir()
    monkeypatch.setattr(event_handlers, "ExportFormatDialog", ChooseCoco)
    monkeypatch.setattr(event_handlers.QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
    monkeypatch.setattr(event_handlers.QMessageBox, "information", lambda *a, **k: shown.append(a[2]))

    handler.export_ml_data()

    document = json.loads((out / "annotations.json").read_text())
    assert document["images"][0]["width"] == 400
    assert document["annotations"][0]["bbox"] == [80.0, 90.0, 40.0, 20.0]
    assert (out / "images" / "f0001.png").is_file()
    assert "1 images and 1 bubbles" in shown[0]
    gui.close()


def test_cancelling_the_dialog_writes_nothing(monkeypatch, tmp_path: Path):
    _ = QApplication.instance() or QApplication([])
    handler = ResultsTabHandler(ImageProcessingModel(Config()), Config())
    gui = MainWindow()
    handler.load_gui(gui)

    class Cancelled:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

    asked_for_folder: list[bool] = []

    def pick_folder(*args, **kwargs) -> str:
        asked_for_folder.append(True)
        return ""

    monkeypatch.setattr(event_handlers, "ExportFormatDialog", Cancelled)
    monkeypatch.setattr(event_handlers.QFileDialog, "getExistingDirectory", pick_folder)

    handler.export_ml_data()

    assert not asked_for_folder
    gui.close()
