"""The Step 1 parameter table in the image processing tab."""

from types import SimpleNamespace

from PySide6.QtWidgets import QApplication, QTableWidget

from bubble_analyser.gui.event_handlers import ImageProcessingTabHandler
from bubble_analyser.methods.sam_method import SamSegmentation
from bubble_analyser.methods.watershed_methods import NormalWatershed


def test_parameter_table_keeps_its_header_after_switching_method():
    _ = QApplication.instance() or QApplication([])
    table = QTableWidget(0, 2)
    table.setHorizontalHeaderLabels(["Parameter", "Value"])
    methods = {"Default": NormalWatershed({}), "SAM 2.1 (Segment Anything)": SamSegmentation({})}
    model = SimpleNamespace(
        methods_handler=SimpleNamespace(all_classes=methods),
        all_methods_n_params={name: method.get_needed_params() for name, method in methods.items()},
    )
    handler = SimpleNamespace(gui=SimpleNamespace(param_sandbox1=table), model=model, current_algorithm=None)

    for algorithm in ("Default", "SAM 2.1 (Segment Anything)"):
        ImageProcessingTabHandler.load_parameter_table_1(handler, algorithm)  # type: ignore[arg-type]
        assert [table.horizontalHeaderItem(column).text() for column in range(2)] == ["Parameter", "Value"]

    assert table.item(0, 0).text() == "weights_path"
    assert "SAM 2.1 checkpoint" in table.item(0, 0).toolTip()
