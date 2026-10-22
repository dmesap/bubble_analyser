"""Unit tests for ZoomableImageView, ZoomablePreviewLabel, and wheel zooming."""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import QApplication

from bubble_analyser.widgets.zoom_view import (
    ImageViewerDialog,
    ZoomableImageView,
    ZoomablePreviewLabel,
    wheel_zoom_factor,
)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def make_pixmap(width: int = 200, height: int = 100) -> QPixmap:
    img = QImage(width, height, QImage.Format.Format_RGB888)
    img.fill(0)
    return QPixmap.fromImage(img)


def test_wheel_zoom_factor():
    class DummyWheel:
        def __init__(self, delta: int):
            self._delta = delta

        def angleDelta(self):
            return QPoint(0, self._delta)

    # 120 angle delta (1 click) is 1.25x
    factor = wheel_zoom_factor(DummyWheel(120))
    assert factor is not None
    assert pytest.approx(factor, 0.01) == 1.25

    # -120 angle delta (-1 click) is 0.8x
    factor_out = wheel_zoom_factor(DummyWheel(-120))
    assert factor_out is not None
    assert pytest.approx(factor_out, 0.01) == 0.8

    # 0 delta returns None
    assert wheel_zoom_factor(DummyWheel(0)) is None


def test_zoomable_image_view_zoom(qapp):
    view = ZoomableImageView()
    pixmap = make_pixmap(400, 300)
    view.resize(500, 400)
    view.set_pixmap(pixmap)

    assert view.has_image()
    initial_scale = view.zoom_scale

    view.zoom_in()
    assert view.zoom_scale > initial_scale

    view.zoom_out()
    assert pytest.approx(view.zoom_scale, 0.01) == initial_scale

    view.actual_size()
    assert view.zoom_scale == 1.0


def test_zoomable_preview_label(qapp):
    label = ZoomablePreviewLabel(viewer_title="Test preview")
    label.resize(200, 150)
    pixmap = make_pixmap(800, 600)
    label.set_preview_pixmap(pixmap)

    assert label.full_pixmap() is not None
    assert label.full_pixmap().size() == pixmap.size()
    # The displayed pixmap is scaled to fit within 200x150
    assert label.pixmap() is not None
    assert label.pixmap().width() <= 200
    assert label.pixmap().height() <= 150

    dialog = label.open_viewer()
    assert isinstance(dialog, ImageViewerDialog)
    dialog.close()
