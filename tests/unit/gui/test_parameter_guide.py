"""The information button next to the method drop-down and the parameter guide window."""

from types import SimpleNamespace

from PySide6.QtWidgets import QApplication, QComboBox, QWidget

from bubble_analyser.gui.event_handlers import ImageProcessingTabHandler
from bubble_analyser.gui.gui import MainWindow
from bubble_analyser.gui.param_guide_dialog import ParameterGuideDialog
from bubble_analyser.methods.sam_method import SamSegmentation
from bubble_analyser.methods.watershed_methods import NormalWatershed


def test_information_button_sits_next_to_the_method_drop_down():
    _ = QApplication.instance() or QApplication([])
    window = MainWindow()

    button = window.algorithm_info_button
    assert not button.icon().isNull()
    assert "Parameter guide" in button.toolTip()
    assert button.parentWidget() is window.algorithm_combo.parentWidget()
    window.close()


def test_guide_window_follows_the_selected_method(monkeypatch):
    _ = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ParameterGuideDialog, "show", lambda self: None)  # keep the window off screen

    combo = QComboBox()
    combo.addItems(["Default", "SAM 2.1 (Segment Anything)"])
    gui = QWidget()
    gui.algorithm_combo = combo  # type: ignore[attr-defined]
    methods = {"Default": NormalWatershed({}), "SAM 2.1 (Segment Anything)": SamSegmentation({})}
    model = SimpleNamespace(
        methods_handler=SimpleNamespace(all_classes=methods),
        all_methods_n_params={name: method.get_needed_params() for name, method in methods.items()},
    )
    handler = SimpleNamespace(gui=gui, model=model, _parameter_guide_dialog=None)
    handler._update_parameter_guide = lambda dialog: ImageProcessingTabHandler._update_parameter_guide(
        handler,  # type: ignore[arg-type]
        dialog,
    )

    ImageProcessingTabHandler.show_parameter_guide(handler)  # type: ignore[arg-type]
    dialog = handler._parameter_guide_dialog
    assert dialog.windowTitle() == "Parameter guide: Default"
    assert "high_thresh" in dialog.browser.toPlainText()

    combo.setCurrentText("SAM 2.1 (Segment Anything)")
    monkeypatch.setattr(ParameterGuideDialog, "isVisible", lambda self: True)
    ImageProcessingTabHandler.refresh_parameter_guide(handler)  # type: ignore[arg-type]
    assert dialog.windowTitle() == "Parameter guide: SAM 2.1 (Segment Anything)"
    assert "points_per_side" in dialog.browser.toPlainText()
    assert "high_thresh" not in dialog.browser.toPlainText()
    dialog.close()
