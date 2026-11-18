"""Section 6: replicate strips per key feature, mAP per condition, CV table."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from synthdata.report.context import ReportContext, SectionResult
from synthdata.report.sections._common import condition_color, fig, table
from synthdata.report.stats import replicate_cv, replicate_map

STRIP_W, STRIP_H = 300, 200


def key_columns(ctx: ReportContext) -> list[str]:
    cols = [f"{f.name}_median" for f in ctx.inputs.features if f.kind == "linear"]
    cols += ["orientation_coherence", "count_per_site"]
    return [c for c in cols if c in ctx.profiles.columns]


def strip_data(ctx: ReportContext, columns: list[str]) -> pd.DataFrame:
    """Well values with a deterministic horizontal jitter and the per-condition median."""
    rng = np.random.default_rng(0)
    idx = {c: i for i, c in enumerate(ctx.conditions)}
    frames = []
    for col in columns:
        d = ctx.profiles[["condition", col]].copy()
        d["condition"] = d["condition"].astype(str)
        d = d.rename(columns={col: "value"}).dropna(subset=["value"])
        d["column"] = col
        d["label"] = ctx.label(col)
        d["pos"] = d["condition"].map(idx).astype(float) + rng.uniform(-0.28, 0.28, len(d))
        d["median"] = d.groupby("condition")["value"].transform("median")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["idx"] = out["condition"].map(idx)
    return out.reset_index(drop=True)


def _strips(ctx: ReportContext, data: pd.DataFrame, columns: list[str]):
    n = len(ctx.conditions)
    names = ctx.conditions
    label_expr = "[" + ",".join("'" + c.replace("'", "\\'") + "'" for c in names) + "][datum.value]"
    charts = []
    for col in columns:
        d = data[data["column"] == col]
        lab = ctx.label(col)
        d = d.assign(value=d["value"].round(5), pos=d["pos"].round(3))
        x = alt.X("pos:Q", title=None, scale=alt.Scale(domain=[-0.6, n - 0.4], nice=False),
                  axis=alt.Axis(values=list(range(n)), labelExpr=label_expr, labelAngle=-45, grid=False,
                                labelLimit=110, labelOverlap=False, labelFontSize=10))
        pts = alt.Chart(d).mark_circle(size=42, opacity=0.8, stroke="white", strokeWidth=0.4).encode(
            x=x, y=alt.Y("value:Q", title=lab, scale=alt.Scale(zero=False)),
            color=condition_color(ctx, legend=False),
            tooltip=[alt.Tooltip("condition:N"), alt.Tooltip("value:Q", title=lab, format=".4g")])
        med = alt.Chart(d.drop_duplicates("condition")).mark_rule(color="#222", strokeWidth=2).encode(
            x=alt.X("x1:Q"), x2="x2:Q", y="median:Q",
            tooltip=[alt.Tooltip("condition:N"), alt.Tooltip("median:Q", format=".4g", title="Median")]
        ).transform_calculate(x1="datum.idx - 0.36", x2="datum.idx + 0.36")
        charts.append(alt.layer(pts, med).properties(width=STRIP_W, height=STRIP_H, title=lab))
    return alt.concat(*charts, columns=3, spacing=18).properties(
        title="Well values by condition (one dot per well; bar: condition median)")


def _map_chart(ctx: ReportContext, m: pd.DataFrame):
    m = m.assign(mAP=m["mAP"].round(4))
    base = alt.Chart(m).encode(y=alt.Y("condition:N", sort=ctx.conditions, title=None),
                               x=alt.X("mAP:Q", title="Replicate mAP (1 = wells always retrieve their own condition)",
                                       scale=alt.Scale(domain=[0, 1], nice=False)))
    bars = base.mark_bar().encode(color=condition_color(ctx, legend=False),
                                  tooltip=[alt.Tooltip("condition:N"), alt.Tooltip("mAP:Q", format=".3f"),
                                           alt.Tooltip("n_wells:Q", title="Wells")])
    txt = base.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text("mAP:Q", format=".2f"))
    return alt.layer(bars, txt).properties(width=420, height=24 * len(m), title="Replicate mean average precision")


def build(ctx: ReportContext) -> SectionResult:
    from synthdata.report.sections import caption_for

    res = SectionResult("replicates", "Replicate consistency", caption_for(ctx.inputs, "replicates"))
    cols = key_columns(ctx)
    if cols:
        data = strip_data(ctx, cols)
        res.figures.append(fig(ctx, "replicates", "replicate_strips", _strips(ctx, data, cols),
                               "Replicate strips"))
    m = replicate_map(ctx.z, ctx.profiles["condition"], ctx.columns)
    if len(m):
        res.figures.append(fig(ctx, "replicates", "map_bars", _map_chart(ctx, m), "Replicate mAP"))
    else:
        res.notes.append("No condition has two or more wells; the mAP bars are omitted.")
    res.tables.append(table(ctx, "replicates", "replicate_cv", replicate_cv(ctx.profiles, ctx.columns)))
    res.tables.append(table(ctx, "replicates", "map", m))
    return res
