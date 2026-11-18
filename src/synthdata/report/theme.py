"""The one report theme: a Vega-Lite config plus the condition palette."""

from __future__ import annotations

import altair as alt

FONT = "Inter, Helvetica, Arial, sans-serif"
CATEGORY = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
            "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

# Quality-control calls. Colour-blind safe: the three calls differ in luminance (light blue > amber > dark red)
# as well as hue, and every cell also carries a letter (QC_MARKS).
QC_COLOURS = {"pass": "#a8d5e5", "warn": "#f0a202", "fail": "#8c1c13"}
QC_NOT_ASSESSED = "#d9d9d9"
QC_MARKS = {"pass": "P", "warn": "W", "fail": "F", "not assessed": "–"}
QC_TEXT = {"pass": "#1f2933", "warn": "#1f2933", "fail": "#ffffff", "not assessed": "#4a4a4a"}

THEME: dict = {
    "font": FONT,
    "background": "white",
    "title": {"font": FONT, "fontSize": 14, "fontWeight": 600, "subtitleFont": FONT, "subtitleFontSize": 11,
              "subtitleColor": "#5b6170", "subtitlePadding": 3},
    "axis": {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 11, "titleFontSize": 12,
             "gridColor": "#e6e6e6", "domainColor": "#9a9a9a", "tickColor": "#9a9a9a"},
    "legend": {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 11, "titleFontSize": 12,
               "titleLimit": 320, "labelLimit": 200, "symbolOpacity": 1},
    "header": {"labelFont": FONT, "titleFont": FONT, "labelFontSize": 11, "titleFontSize": 12},
    "view": {"stroke": "transparent"},
    "range": {"category": CATEGORY},
}


def condition_palette(conditions: list[str]) -> dict[str, str]:
    return {c: CATEGORY[i % len(CATEGORY)] for i, c in enumerate(conditions)}


def tint(hex_colour: str, f: float = 0.55) -> str:
    """Mix a colour with white; f is the share of white."""
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c + (255 - c) * f):02x}" for c in rgb)


def register() -> None:
    """Register and enable the theme as "synthdata" (idempotent)."""
    @alt.theme.register("synthdata", enable=True)
    def _synthdata() -> alt.theme.ThemeConfig:
        return alt.theme.ThemeConfig(config=THEME)


def apply(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Embed the config in a chart so a saved spec renders identically anywhere."""
    return chart.configure(**THEME)
