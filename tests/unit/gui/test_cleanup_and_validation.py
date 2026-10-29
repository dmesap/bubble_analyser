from PySide6.QtWidgets import QApplication, QMessageBox

from bubble_analyser.gui.event_handlers import MainHandler
from bubble_analyser.gui.gui import MainWindow


def test_preview_image_folder_tab_none_selection():
    _ = QApplication.instance() or QApplication([])
    handler = MainHandler()
    handler.gui = MainWindow()
    handler.load_gui_for_handlers()
    handler.gui.image_list.clear()
    assert handler.gui.image_list.currentItem() is None

    # Should not raise AttributeError or crash when nothing is selected
    handler.folder_tab_handler.preview_image_folder_tab()


def test_confirm_calibration_validation_and_invalid_bg(monkeypatch):
    _ = QApplication.instance() or QApplication([])
    handler = MainHandler()
    handler.gui = MainWindow()
    handler.load_gui_for_handlers()

    warning_calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warning_calls.append(args))

    # 1. Empty string
    handler.gui.manual_px_mm_input.setText("")
    handler.calibration_tab_handler.confirm_calibration()
    assert len(warning_calls) == 1
    assert handler.calibration_model.calibration_confirmed is False

    # 2. Invalid text
    handler.gui.manual_px_mm_input.setText("not_a_number")
    handler.calibration_tab_handler.confirm_calibration()
    assert len(warning_calls) == 2
    assert handler.calibration_model.calibration_confirmed is False

    # 3. Non-positive float
    handler.gui.manual_px_mm_input.setText("-5.0")
    handler.calibration_tab_handler.confirm_calibration()
    assert len(warning_calls) == 3
    assert handler.calibration_model.calibration_confirmed is False

    # 4. Valid float with non-existent background image
    handler.gui.manual_px_mm_input.setText("12.5")
    handler.gui.bg_corr_image_name.setText("non_existent_background_file.png")
    handler.calibration_tab_handler.confirm_calibration()
    assert handler.calibration_model.calibration_confirmed is True
    assert handler.calibration_model.px2mm_display == 12.5
    assert handler.calibration_model.if_bknd is False
    assert handler.calibration_model.bknd_img_path is None


def test_restart_mission_disconnect_and_clear():
    _ = QApplication.instance() or QApplication([])
    handler = MainHandler()
    handler.gui = MainWindow()
    handler.load_gui_for_handlers()
    handler.connect_gui_and_handlers()

    handler.gui.folder_path_edit.setText("some/path")
    handler.gui.manual_px_mm_input.setText("15.0")
    handler.gui.tabs.setCurrentIndex(2)

    # Disconnect should be clean and idempotent
    handler.disconnect_gui_and_handlers()
    handler.disconnect_gui_and_handlers()

    # Clear GUI contents
    handler.clear_all_gui_contents()
    assert handler.gui.folder_path_edit.text() == ""
    assert handler.gui.manual_px_mm_input.text() == ""
    assert handler.gui.tabs.currentIndex() == 0
