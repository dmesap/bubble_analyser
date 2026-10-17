import numpy as np
from PySide6.QtCore import QPoint, QPointF
from PySide6.QtWidgets import QApplication, QWidget

from bubble_analyser.processing.calculate_px2mm import ImageLabel


class _Mouse:
    def __init__(self, x: float, y: float) -> None:
        self._position = QPointF(x, y)

    def position(self) -> QPointF:
        return self._position


class _Wheel(_Mouse):
    def __init__(self, direction: int) -> None:
        super().__init__(10, 10)
        self._direction = direction

    def angleDelta(self) -> QPoint:
        return QPoint(0, self._direction * 120)

    def globalPosition(self) -> QPointF:
        return QPointF(10, 10)


def _label(width: int = 200, height: int = 100) -> ImageLabel:
    QApplication.instance() or QApplication([])
    parent = QWidget()
    label = ImageLabel(parent)
    setattr(label, "parent_widget", parent)  # keep the parent alive, or Qt deletes the label with it
    label.set_image(np.zeros((height, width, 3), dtype=np.uint8))
    return label


def test_clicks_are_mapped_to_image_pixels_at_any_zoom():
    label = _label()
    label.display_scale = 2.0

    assert label.event_to_image_position(_Mouse(100, 50)) == (50, 25)


def test_clicks_outside_the_image_are_clamped():
    label = _label(200, 100)

    assert label.event_to_image_position(_Mouse(10_000, 10_000)) == (199, 99)
    assert label.event_to_image_position(_Mouse(-5, -5)) == (0, 0)


def test_wheel_zoom_steps_and_stays_within_limits():
    label = _label()

    label.wheelEvent(_Wheel(+1))
    assert label.display_scale == 1.25

    for _ in range(40):
        label.wheelEvent(_Wheel(+1))
    assert label.display_scale == label.max_display_scale

    for _ in range(80):
        label.wheelEvent(_Wheel(-1))
    assert label.display_scale == label.min_display_scale
