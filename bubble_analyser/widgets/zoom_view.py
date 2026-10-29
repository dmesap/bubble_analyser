"""Zoomable image widgets used across the Bubble Analyser GUI.

``ZoomableImageView``    A scroll area that shows a pixmap with wheel zoom (around the cursor),
                         drag-to-pan and fit/100 % helpers.
``ImageViewerDialog``    A resizable window wrapping ``ZoomableImageView`` with zoom buttons and
                         keyboard shortcuts.
``ZoomablePreviewLabel`` A drop-in replacement for the fixed-size preview ``QLabel``s: it keeps the
                         full-resolution pixmap and opens it in ``ImageViewerDialog`` on double-click.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QKeySequence, QMouseEvent, QPixmap, QResizeEvent, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

WHEEL_STEP_FACTOR = 1.25  # zoom factor for one standard mouse-wheel notch (120 units)


def wheel_zoom_factor(ev: QWheelEvent) -> float | None:
    """Return the zoom factor for a wheel event, or ``None`` if it carries no vertical motion.

    Mouse wheels report multiples of 120 in ``angleDelta``; touchpads report small, frequent deltas
    (sometimes only in ``pixelDelta``). Scaling the factor by the delta makes both feel natural.
    """
    delta = ev.angleDelta().y()
    if delta == 0:
        pixel_delta = getattr(ev, "pixelDelta", None)
        delta = pixel_delta().y() if callable(pixel_delta) else 0
    if delta == 0:
        return None
    return float(WHEEL_STEP_FACTOR ** (delta / 120.0))


class ZoomableImageView(QScrollArea):
    """Scrollable pixmap view with mouse-wheel zoom and drag panning."""

    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty view; call :meth:`set_pixmap` to show an image."""
        super().__init__(parent)
        self.min_scale = 0.05
        self.max_scale = 16.0
        self.zoom_scale = 1.0
        self._source: QPixmap | None = None
        self._pan_origin: QPoint | None = None

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setWidget(self._label)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

    # ----------------------------------------------------------------- image
    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Show ``pixmap`` and fit it to the window."""
        self._source = pixmap
        self.fit_to_window()

    def has_image(self) -> bool:
        """Return True when a non-empty pixmap is shown."""
        return self._source is not None and not self._source.isNull()

    # ------------------------------------------------------------------ zoom
    def set_scale(self, scale: float, viewport_anchor: QPointF | None = None) -> None:
        """Zoom to ``scale``, keeping the image point under ``viewport_anchor`` fixed on screen."""
        if not self.has_image():
            return
        assert self._source is not None
        new_scale = min(self.max_scale, max(self.min_scale, scale))
        if viewport_anchor is None:
            viewport_anchor = QPointF(self.viewport().width() / 2, self.viewport().height() / 2)

        # Image coordinates (in source pixels) of the anchor before zooming.
        label_pos = self._label.mapFrom(self.viewport(), viewport_anchor.toPoint())
        old_size = self._label.size()
        offset_x = (old_size.width() - self._source.width() * self.zoom_scale) / 2
        offset_y = (old_size.height() - self._source.height() * self.zoom_scale) / 2
        image_x = (label_pos.x() - offset_x) / self.zoom_scale
        image_y = (label_pos.y() - offset_y) / self.zoom_scale

        self.zoom_scale = new_scale
        scaled = self._source.scaled(
            max(1, round(self._source.width() * new_scale)),
            max(1, round(self._source.height() * new_scale)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

        self.horizontalScrollBar().setValue(round(image_x * new_scale - viewport_anchor.x()))
        self.verticalScrollBar().setValue(round(image_y * new_scale - viewport_anchor.y()))
        self.zoom_changed.emit(new_scale)

    def zoom_by(self, factor: float, viewport_anchor: QPointF | None = None) -> None:
        """Multiply the current zoom by ``factor``."""
        self.set_scale(self.zoom_scale * factor, viewport_anchor)

    def zoom_in(self) -> None:
        """Zoom in one step around the centre of the view."""
        self.zoom_by(WHEEL_STEP_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out one step around the centre of the view."""
        self.zoom_by(1 / WHEEL_STEP_FACTOR)

    def actual_size(self) -> None:
        """Show the image at 100 % (one image pixel per screen pixel)."""
        self.set_scale(1.0)

    def fit_to_window(self) -> None:
        """Scale the image so it fits entirely inside the view."""
        if not self.has_image():
            return
        assert self._source is not None
        available = self.viewport().size()
        scale = min(
            max(available.width() - 2, 1) / max(self._source.width(), 1),
            max(available.height() - 2, 1) / max(self._source.height(), 1),
        )
        self.set_scale(scale)

    # ---------------------------------------------------------------- events
    def wheelEvent(self, ev: QWheelEvent) -> None:  # noqa: N802 (Qt name)
        """Zoom around the cursor instead of scrolling."""
        factor = wheel_zoom_factor(ev)
        if factor is not None:
            self.zoom_by(factor, ev.position())
        ev.accept()

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802 (Qt name)
        """Start panning with the left, middle or right button."""
        self._pan_origin = ev.globalPosition().toPoint()
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        ev.accept()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802 (Qt name)
        """Pan while a mouse button is held down."""
        if self._pan_origin is None:
            return
        current = ev.globalPosition().toPoint()
        delta = current - self._pan_origin
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        self._pan_origin = current
        ev.accept()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: N802 (Qt name)
        """Stop panning."""
        self._pan_origin = None
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        ev.accept()


def add_zoom_shortcuts(widget: QWidget, zoom_in, zoom_out, fit=None, actual=None) -> None:  # type: ignore[no-untyped-def]
    """Bind the usual zoom shortcuts (Ctrl +/-, +/-, Ctrl 0, Ctrl 1) on ``widget``."""
    bindings = [
        (QKeySequence.StandardKey.ZoomIn, zoom_in),
        (QKeySequence("Ctrl+="), zoom_in),
        (QKeySequence("+"), zoom_in),
        (QKeySequence("="), zoom_in),
        (QKeySequence.StandardKey.ZoomOut, zoom_out),
        (QKeySequence("-"), zoom_out),
    ]
    if fit is not None:
        bindings += [(QKeySequence("Ctrl+0"), fit), (QKeySequence("F"), fit)]
    if actual is not None:
        bindings.append((QKeySequence("Ctrl+1"), actual))
    for sequence, slot in bindings:
        shortcut = QShortcut(sequence, widget)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(slot)


def make_zoom_toolbar(zoom_in, zoom_out, fit=None, actual=None, hint: str = "") -> tuple[QHBoxLayout, QLabel]:  # type: ignore[no-untyped-def]
    """Build a row of zoom buttons; returns the layout and a label for showing the zoom level."""
    row = QHBoxLayout()
    buttons = [("−", "Zoom out (-)", zoom_out), ("+", "Zoom in (+)", zoom_in)]
    if fit is not None:
        buttons.append(("Fit", "Fit the whole image in the window (Ctrl+0)", fit))
    if actual is not None:
        buttons.append(("100%", "Show at 100 % (Ctrl+1)", actual))
    for text, tooltip, slot in buttons:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.setAutoDefault(False)
        button.setFixedWidth(48 if len(text) > 1 else 32)
        button.clicked.connect(slot)
        row.addWidget(button)
    zoom_label = QLabel("100 %")
    zoom_label.setMinimumWidth(60)
    row.addWidget(zoom_label)
    if hint:
        hint_label = QLabel(hint)
        hint_label.setStyleSheet("color: #666;")
        row.addWidget(hint_label)
    row.addStretch()
    return row, zoom_label


class ImageViewerDialog(QDialog):
    """A resizable window to inspect an image with zoom and pan."""

    def __init__(self, pixmap: QPixmap, title: str = "Image viewer", parent: QWidget | None = None) -> None:
        """Open ``pixmap`` fitted to a window that covers most of the screen."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)

        self.view = ZoomableImageView(self)
        toolbar, self.zoom_label = make_zoom_toolbar(
            self.view.zoom_in,
            self.view.zoom_out,
            self.view.fit_to_window,
            self.view.actual_size,
            hint="Mouse wheel: zoom  ·  drag: pan",
        )
        self.view.zoom_changed.connect(lambda scale: self.zoom_label.setText(f"{scale * 100:.0f} %"))
        add_zoom_shortcuts(self, self.view.zoom_in, self.view.zoom_out, self.view.fit_to_window, self.view.actual_size)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.view)

        screen = self.screen().availableGeometry() if self.screen() is not None else None
        if screen is not None:
            self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        else:
            self.resize(1200, 800)
        self._pixmap = pixmap
        self._fitted_once = False

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802 (Qt name)
        """Fit the image once the window has its final size."""
        super().showEvent(event)
        if not self._fitted_once:
            self._fitted_once = True
            self.view.set_pixmap(self._pixmap)


class ZoomablePreviewLabel(QLabel):
    """Fixed-size preview that keeps the full image and opens a zoom viewer on double-click."""

    def __init__(self, text: str = "", viewer_title: str = "Image viewer", parent: QWidget | None = None) -> None:
        """Create the preview; ``viewer_title`` is used as the viewer window title."""
        super().__init__(text, parent)
        self.viewer_title = viewer_title
        self._full_pixmap: QPixmap | None = None
        self.setToolTip("Double-click to open the image in a zoomable viewer")

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        """Show ``pixmap`` scaled to the label while keeping the full-resolution copy for zooming."""
        self._full_pixmap = pixmap
        self._refresh()

    def full_pixmap(self) -> QPixmap | None:
        """Return the full-resolution pixmap last shown, if any."""
        return self._full_pixmap

    def _refresh(self) -> None:
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        self.setPixmap(
            self._full_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt name)
        """Rescale the preview when the label changes size."""
        super().resizeEvent(event)
        self._refresh()

    def open_viewer(self) -> ImageViewerDialog | None:
        """Open the full-resolution image in a zoom viewer window."""
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return None
        dialog = ImageViewerDialog(self._full_pixmap, self.viewer_title, self.window())
        dialog.show()
        return dialog

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt name)
        """Open the viewer on a left double-click."""
        if event.button() == Qt.MouseButton.LeftButton and self.open_viewer() is not None:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
