"""The parameter guides shown by the information button next to the method drop-down."""

from html import escape

import pytest

from bubble_analyser.core.param_guide import GENERAL_TIPS, build_guide_html, format_param_tooltip, guide_tooltips
from bubble_analyser.methods import watershed_methods
from bubble_analyser.methods.bubmask_method import BubMaskWatershed
from bubble_analyser.methods.peak_watershed import PeakWatershed
from bubble_analyser.methods.sam_method import SamSegmentation
from bubble_analyser.methods.watershed_methods import IterativeWatershed, NormalWatershed
from bubble_analyser.methods.yolo_method import YOLOSegmentation

GUIDED_METHODS = [
    NormalWatershed,
    IterativeWatershed,
    PeakWatershed,
    BubMaskWatershed,
    SamSegmentation,
    YOLOSegmentation,
]


@pytest.mark.parametrize("method_class", GUIDED_METHODS)
def test_guide_covers_exactly_the_method_parameters(method_class):
    method = method_class({})
    guide = method.get_param_guide()

    assert guide["summary"]
    assert set(guide["params"]) == set(method.get_needed_params())
    assert all(entry["what"] for entry in guide["params"].values())


@pytest.mark.parametrize("method_class", GUIDED_METHODS)
def test_table_tooltips_use_the_guide_wording(method_class):
    method = method_class({})

    assert method.get_param_descriptions() == guide_tooltips(method.get_param_guide())


def test_tooltip_lists_effects_on_separate_lines():
    tooltip = format_param_tooltip({"what": "Does A.", "higher": "More B.", "lower": "Less B.", "tip": "Try 3."})

    assert tooltip.splitlines() == ["Does A.", "Higher: More B.", "Lower: Less B.", "Tip: Try 3."]


def test_guide_html_follows_the_parameter_table_order():
    method = NormalWatershed({})
    names = list(method.get_needed_params())

    html = build_guide_html("Default", method.description, method.get_param_guide(), names)

    assert "General tips" in html
    assert all(escape(tip) in html for tip in GENERAL_TIPS)
    positions = [html.index(f"<b>{name}</b>") for name in names]
    assert positions == sorted(positions)


def test_guide_html_escapes_text_and_handles_methods_without_a_guide():
    html = build_guide_html("A <new> method", "Uses x < y.", None, ["param1"])

    assert "A &lt;new&gt; method" in html
    assert "Uses x &lt; y." in html
    assert "No parameter guide is available" in html


def test_new_method_has_no_guide_and_its_ksize_tooltip_describes_the_sobel_filter():
    method = watershed_methods.TestWatershed({})  # listed as 'NEW method'

    assert not hasattr(method, "get_param_guide")
    assert "Sobel" in method.get_param_descriptions()["ksize"]
