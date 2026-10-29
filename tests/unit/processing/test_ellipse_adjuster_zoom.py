import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from bubble_analyser.processing.fit_ellipse import EllipseAdjuster


@pytest.fixture(autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _make_dummy_adjuster(width: int = 1200, height: int = 800) -> EllipseAdjuster:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Circle at (600, 400), axes (100.0, 100.0), angle 0
    ellipses = [((600.0, 400.0), (100.0, 100.0), 0.0)]
    adjuster = EllipseAdjuster(ellipses, img)
    return adjuster


def test_ellipse_adjuster_initial_state():
    adjuster = _make_dummy_adjuster()
    assert adjuster.zoom_level == 1.0
    assert adjuster.scroll_area is not None
    assert adjuster.image_label is not None
    assert adjuster.image_label.parent() is adjuster.scroll_area.viewport()
    assert not adjuster.is_panning
    assert adjuster.zoom_label.text() == "100 %"


def test_ellipse_adjuster_zoom_in_out_fit_and_100():
    adjuster = _make_dummy_adjuster(width=2400, height=1600)
    adjuster.zoom_in()
    assert adjuster.zoom_level == pytest.approx(1.25)
    assert adjuster.zoom_label.text() == "125 %"

    adjuster.zoom_out()
    assert adjuster.zoom_level == pytest.approx(1.0)

    adjuster.set_zoom(0.01)
    assert adjuster.zoom_level == adjuster.min_zoom

    adjuster.set_zoom(50.0)
    assert adjuster.zoom_level == adjuster.max_zoom

    adjuster.zoom_fit()
    assert adjuster.zoom_level == pytest.approx(1.0)

    # In zoom_100, scale_factor should be 1.0
    adjuster.zoom_100()
    assert adjuster.scale_factor == pytest.approx(1.0)


def test_ellipse_adjuster_wheel_zoom():
    adjuster = _make_dummy_adjuster()

    wheel_in = QWheelEvent(
        QPointF(100, 100),
        QPointF(100, 100),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    adjuster.handle_wheel(wheel_in)
    assert adjuster.zoom_level > 1.0

    old_zoom = adjuster.zoom_level
    wheel_out = QWheelEvent(
        QPointF(100, 100),
        QPointF(100, 100),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    adjuster.handle_wheel(wheel_out)
    assert adjuster.zoom_level < old_zoom


def test_ellipse_adjuster_panning():
    adjuster = _make_dummy_adjuster(width=2400, height=1600)
    # Zoom in so scrollbars become active
    adjuster.set_zoom(3.0)

    # Simulate middle mouse press
    press_ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(200, 200),
        QPointF(200, 200),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    adjuster.handle_mouse_press(press_ev)
    assert adjuster.is_panning
    assert adjuster.last_pan_pos == QPoint(200, 200)

    # Simulate mouse move while panning
    move_ev = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(150, 170),
        QPointF(150, 170),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    adjuster.handle_mouse_move(move_ev)
    assert adjuster.last_pan_pos == QPoint(150, 170)

    # Release
    release_ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(150, 170),
        QPointF(150, 170),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    adjuster.handle_mouse_release(release_ev)
    assert not adjuster.is_panning
    assert adjuster.last_pan_pos is None
