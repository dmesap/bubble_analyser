"""Dialog for choosing the dataset format of an export."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from bubble_analyser.processing.dataset_export import (
    DATASET_FORMATS,
    DEFAULT_FORMAT_KEY,
    DatasetFormat,
)


class ExportFormatDialog(QDialog):
    """Let the user pick one of the registered dataset formats.

    The description under the selector changes with the format, so the folder layout
    each tool expects is visible before anything is written.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        formats: tuple[DatasetFormat, ...] = DATASET_FORMATS,
        default_key: str = DEFAULT_FORMAT_KEY,
    ) -> None:
        """Build the dialog.

        Args:
            parent: Parent widget.
            formats: Formats offered in the selector.
            default_key: Key of the format selected when the dialog opens.
        """
        super().__init__(parent)
        self.setWindowTitle("Export Dataset")
        self.setMinimumWidth(460)
        self._formats = formats

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Format of the exported dataset:"))

        self.format_combo = QComboBox()
        for item in formats:
            self.format_combo.addItem(item.label, item.key)
        layout.addWidget(self.format_combo)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #555; padding: 6px 2px;")
        layout.addWidget(self.description_label)

        note = QLabel(
            "Bubbles are exported at the original image resolution, using the manual polygon "
            "when there is one and the fitted ellipse otherwise."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Choose Folder...")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.format_combo.currentIndexChanged.connect(self._update_description)
        default_index = self.format_combo.findData(default_key)
        self.format_combo.setCurrentIndex(max(default_index, 0))
        self._update_description()

    def _update_description(self) -> None:
        index = self.format_combo.currentIndex()
        self.description_label.setText(self._formats[index].description if index >= 0 else "")

    def selected_format(self) -> DatasetFormat:
        """Return the format currently selected."""
        return self._formats[self.format_combo.currentIndex()]
