"""Section 1: plate layout, example images, summary tables."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image

from synthdata.report.context import FigureRecord, ReportContext, SectionResult
from synthdata.report.sections._common import condition_color, fig, plate_axes, table

CELL = 28


def _ink(hex_colour: str) -> str:
    """Dark text on light fills, white on dark ones (relative luminance)."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "#222222" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.6 else "#ffffff"


def _layout(ctx: ReportContext):
    rows, cols = plate_axes(ctx)
    w = ctx.profiles.reset_index()[["plate", "well", "row", "col", "condition", "role", "replicate"]].copy()
    w["condition"] = w["condition"].astype(str)
    w["row"], w["col"] = w["row"].astype(str), w["col"].astype(int)
    color = condition_color(ctx)
    w["ink"] = w["condition"].map(lambda c: _ink(ctx.palette[c]))
    base = alt.Chart(w).encode(
        x=alt.X("col:O", title="Column", sort=cols, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("row:O", title="Row", sort=rows),
        tooltip=[alt.Tooltip("plate:N"), alt.Tooltip("well:N"), alt.Tooltip("condition:N"),
                 alt.Tooltip("role:N"), alt.Tooltip("replicate:Q", title="Replicate")])
    rect = base.mark_rect(stroke="white", strokeWidth=1).encode(color=color)
    text = base.mark_text(fontSize=10, fontWeight=600).encode(
        text="replicate:Q", fill=alt.Fill("ink:N", scale=None))  # fill, not color: no clash with the condition legend
    chart = alt.layer(rect, text).properties(
        width=CELL * len(cols), height=CELL * len(rows),
        title=f"Plate layout: {len(w)} wells, {w['condition'].nunique()} conditions")
    if w["plate"].nunique() > 1:
        chart = chart.facet(column=alt.Column("plate:N", title=None))
    return chart


def _stretch(images: dict[str, np.ndarray]) -> tuple[float, float]:
    stack = np.concatenate([np.asarray(a, float).ravel() for a in images.values()])
    lo, hi = np.percentile(stack, [1, 99.5])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _tiles(ctx: ReportContext) -> list[FigureRecord]:
    images = ctx.inputs.images
    if not images:
        return []
    lo, hi = _stretch(images)
    out_dir = ctx.root / "figures" / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)
    wells = ctx.inputs.wells
    recs = []
    for i, cond in enumerate(ctx.conditions):  # file names are indices: a condition name is never a path
        if cond not in images:
            continue
        arr = np.clip((np.asarray(images[cond], float) - lo) / (hi - lo), 0, 1)
        path = out_dir / f"example_{i:02d}.png"
        Image.fromarray(np.round(arr * 255).astype(np.uint8)).save(path)
        role = wells.loc[wells["condition"].astype(str) == cond, "role"].iloc[0]
        recs.append(FigureRecord(name=f"example_{i:02d}", title=f"{cond} ({role})", spec=None, svg=None,
                                 png=path.relative_to(ctx.root).as_posix()))
    return recs


def build(ctx: ReportContext) -> SectionResult:
    from synthdata.report.sections import caption_for

    res = SectionResult("overview", "Overview", caption_for(ctx.inputs, "overview"))
    res.figures.append(fig(ctx, "overview", "plate_layout", _layout(ctx), "Plate layout"))
    res.figures.extend(_tiles(ctx))
    if not ctx.inputs.images:
        res.notes.append("No example images were supplied; the example tiles are omitted.")

    p = ctx.profiles.reset_index()
    cond = p["condition"].astype(str)
    n_obj = ctx.inputs.objects.groupby(["plate", "well"]).size()
    p["n_objects"] = [int(n_obj.get((a, b), 0)) for a, b in zip(p["plate"], p["well"])]
    rows = []
    for c in ctx.conditions:
        m = cond == c
        rows.append({"condition": c, "role": str(p.loc[m, "role"].iloc[0]), "n_wells": int(m.sum()),
                     "n_objects": int(p.loc[m, "n_objects"].sum())})
    conditions = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"metric": "conditions", "value": str(len(conditions))},
        {"metric": "wells", "value": str(len(p))},
        {"metric": "objects", "value": str(len(ctx.inputs.objects))},
        {"metric": "sites per well", "value": str(ctx.n_sites)},
        {"metric": "reference", "value": ctx.inputs.reference},
    ])
    res.tables.append(table(ctx, "overview", "summary", summary))
    res.tables.append(table(ctx, "overview", "conditions", conditions))
    return res
