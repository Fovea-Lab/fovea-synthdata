"""Section 2: one compact plate heatmap per profile column, laid out as small multiples in the page."""

from __future__ import annotations

import altair as alt

from synthdata.report.context import ReportContext, SectionResult
from synthdata.report.sections._common import fig, plate_axes, table

HEAT_COLUMNS = ["count_per_site", "diameter_median", "eccentricity_median", "intensity_median",
                "orientation_coherence", "nn_distance", "clark_evans"]
CELL = 18


def _chart(ctx: ReportContext, df, col: str) -> alt.Chart:
    """One compact plate map: column numbers on top as on a plate, a horizontal colour bar underneath."""
    rows, cols = plate_axes(ctx)
    label = ctx.label(col)
    if col == "clark_evans":
        scale = alt.Scale(scheme="blueorange", domainMid=1)
        title = f"{label} (1 = random)"
    else:
        scale = alt.Scale(scheme="viridis")
        title = label
    width = CELL * len(cols)
    chart = alt.Chart(df).mark_rect(stroke="white", strokeWidth=0.6).encode(
        x=alt.X("col:O", title=None, sort=cols, axis=alt.Axis(labelAngle=0, orient="top", ticks=False,
                                                                domain=False, labelPadding=2)),
        y=alt.Y("row:O", title=None, sort=rows, axis=alt.Axis(ticks=False, domain=False, labelPadding=4)),
        color=alt.Color(f"{col}:Q", title=None, scale=scale,
                        legend=alt.Legend(orient="bottom", direction="horizontal", gradientLength=width,
                                          gradientThickness=10, labelFontSize=10, offset=6)),
        tooltip=[alt.Tooltip("well:N"), alt.Tooltip("condition:N"),
                 alt.Tooltip(f"{col}:Q", title=label, format=".4g")],
    ).properties(width=width, height=CELL * len(rows),
                 title=alt.TitleParams(title, fontSize=13, anchor="start"))
    if df["plate"].nunique() > 1:
        chart = chart.facet(column=alt.Column("plate:N", title=None))
    return chart


def build(ctx: ReportContext) -> SectionResult:
    from synthdata.report.sections import caption_for

    res = SectionResult("heatmaps", "Plate heatmaps", caption_for(ctx.inputs, "heatmaps"))
    df = ctx.profiles.reset_index().copy()
    df["condition"] = df["condition"].astype(str)
    df["row"], df["col"] = df["row"].astype(str), df["col"].astype(int)
    for col in HEAT_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].notna().sum() == 0:
            res.notes.append(f"{col}: no defined values; heatmap omitted")
            continue
        res.figures.append(fig(ctx, "heatmaps", f"plate_{col}", _chart(ctx, df, col), ctx.label(col)))
    res.tables.append(table(ctx, "heatmaps", "well_profiles", ctx.profiles.reset_index()))
    return res
