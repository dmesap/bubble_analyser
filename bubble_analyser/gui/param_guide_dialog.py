"""Window that shows the parameter guide of the selected segmentation method.

It is opened from the information button next to the method drop-down in the image
processing tab. The window is not modal, so the guide can stay open while parameters are
being adjusted; it follows the selected method.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout, QWidget


class ParameterGuideDialog(QDialog):
    """Non-modal window showing what each parameter of a method does."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the window.

        Args:
            parent: Parent widget (the main window).
        """
        super().__init__(parent)
        self.setWindowTitle("Parameter guide")
        self.setModal(False)
        self.resize(560, 680)

        layout = QVBoxLayout(self)
        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        layout.addWidget(self.browser)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def show_guide(self, method_name: str, html: str) -> None:
        """Replace the window content with the guide of a method.

        Args:
            method_name: Name of the method, used in the window title.
            html: The guide as HTML (see bubble_analyser.core.param_guide.build_guide_html).
        """
        self.setWindowTitle(f"Parameter guide: {method_name}")
        self.browser.setHtml(html)
