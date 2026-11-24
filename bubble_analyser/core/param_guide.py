"""Parameter guide shared by the segmentation methods and the GUI.

Each segmentation method can describe its parameters with a ``MethodGuide`` returned by an
optional ``get_param_guide()`` method. The GUI shows the guide in the parameter guide window
(the information button next to the method drop-down), and the parameter table tooltips use
the same wording through ``guide_tooltips``.
"""

from __future__ import annotations

from html import escape
from typing import NotRequired, TypedDict


class ParamHelp(TypedDict):
    """Help for one parameter: what it does and what raising or lowering it changes."""

    what: str
    higher: NotRequired[str]
    lower: NotRequired[str]
    tip: NotRequired[str]


class MethodGuide(TypedDict):
    """Help for one segmentation method and its parameters."""

    summary: str
    how_it_works: NotRequired[str]
    params: dict[str, ParamHelp]


GENERAL_TIPS: tuple[str, ...] = (
    "Change one parameter at a time, then click 'Confirm parameter and preview' and compare the result.",
    "Settings measured in pixels (element_size, smoothing_sigma, min_mask_region_area) count pixels of the "
    "resampled image, so re-tune them after changing resample.",
    "Bubble sizes in mm are not affected by resample: the app corrects for it.",
    "Hover over a parameter name in the table to see its entry from this guide.",
)

# Entries shared by the classic (watershed) methods.
RESAMPLE_HELP: ParamHelp = {
    "what": "Shrinks the image before segmentation (0.4 = 40% of the width and height).",
    "higher": "Keeps small bubbles and sharper outlines, but it is slower and uses more memory.",
    "lower": "Faster, but small bubbles disappear and touching ones merge.",
    "tip": "Keep the smallest bubble you care about at least about 10 px wide after resampling.",
}

ELEMENT_SIZE_HELP: ParamHelp = {
    "what": (
        "Radius in px of the brush that closes gaps in the bubble mask before holes are filled. "
        "Choices: 0, 3 or 5 (0 = no gap closing)."
    ),
    "higher": (
        "Closes broken outlines, so bubbles that show up as rings with a bright centre get filled, "
        "but bubbles closer than about twice this distance merge."
    ),
    "lower": "Keeps close bubbles apart, but rings with gaps may stay hollow.",
}

CONNECTIVITY_HELP: ParamHelp = {
    "what": "4: pixels that touch only at a corner count as separate. 8: they count as connected.",
    "tip": (
        "8 can merge two seeds that touch diagonally. Leave it at 4; try 8 only if single bubbles "
        "come out split into pieces."
    ),
}


def format_param_tooltip(entry: ParamHelp) -> str:
    """Format one parameter entry as plain text for a tooltip.

    Args:
        entry: The parameter's help entry.

    Returns:
        The entry as lines: what it does, then Higher, Lower and Tip when present.
    """
    lines = [entry["what"]]
    if "higher" in entry:
        lines.append(f"Higher: {entry['higher']}")
    if "lower" in entry:
        lines.append(f"Lower: {entry['lower']}")
    if "tip" in entry:
        lines.append(f"Tip: {entry['tip']}")
    return "\n".join(lines)


def guide_tooltips(guide: MethodGuide) -> dict[str, str]:
    """Return the tooltip text for every parameter in a method guide.

    Args:
        guide: The method's guide.

    Returns:
        A mapping from parameter name to tooltip text.
    """
    return {name: format_param_tooltip(entry) for name, entry in guide["params"].items()}


def _param_html(name: str, entry: ParamHelp) -> str:
    indent = "&nbsp;&nbsp;&nbsp;"
    parts = [f"<b>{escape(name)}</b>: {escape(entry['what'])}"]
    if "higher" in entry:
        parts.append(f"{indent}<i>Higher:</i> {escape(entry['higher'])}")
    if "lower" in entry:
        parts.append(f"{indent}<i>Lower:</i> {escape(entry['lower'])}")
    if "tip" in entry:
        parts.append(f"{indent}<i>Tip:</i> {escape(entry['tip'])}")
    return "<p>" + "<br>".join(parts) + "</p>"


def build_guide_html(
    method_name: str,
    description: str,
    guide: MethodGuide | None,
    param_names: list[str],
    fallback_descriptions: dict[str, str] | None = None,
) -> str:
    """Build the HTML shown in the parameter guide window.

    Args:
        method_name: Name of the segmentation method, as shown in the drop-down.
        description: The method's one-line description, used when it has no guide.
        guide: The method's guide, or None if it does not provide one.
        param_names: The method's parameters in the order of the parameter table.
        fallback_descriptions: Short descriptions for parameters missing from the guide.

    Returns:
        The guide as an HTML string (the subset Qt rich text supports).
    """
    fallback = fallback_descriptions or {}
    html = [f"<h3>{escape(method_name)}</h3>"]

    tips = "".join(f"<li>{escape(tip)}</li>" for tip in GENERAL_TIPS)
    html.append(f"<p><b>General tips</b></p><ul>{tips}</ul>")

    if guide is None:
        if description:
            html.append(f"<p>{escape(description)}</p>")
        html.append("<p><i>No parameter guide is available for this method yet.</i></p>")
        return "".join(html)

    html.append(f"<p>{escape(guide['summary'])}</p>")
    if "how_it_works" in guide:
        html.append(f"<p><i>How it works:</i> {escape(guide['how_it_works'])}</p>")

    html.append("<p><b>Parameters</b></p>")
    params = guide["params"]
    ordered = [name for name in param_names if name in params]
    ordered += [name for name in params if name not in ordered]
    for name in ordered:
        html.append(_param_html(name, params[name]))
    for name in param_names:
        if name not in params and fallback.get(name):
            html.append(f"<p><b>{escape(name)}</b>: {escape(fallback[name])}</p>")
    return "".join(html)
