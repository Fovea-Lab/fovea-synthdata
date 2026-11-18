"""Section 8: each well's quality call, from a generic qc table (label + optional numeric metrics)."""

from __future__ import annotations

import altair as alt
import pandas as pd

from synthdata.report.context import ReportContext, SectionResult
from synthdata.report.contract import QC_LABELS
from synthdata.report.sections._common import fig, plate_axes, table
from synthdata.report.theme import QC_COLOURS, QC_MARKS, QC_NOT_ASSESSED, QC_TEXT

CELL = 22
NOT_ASSESSED = "not assessed"
ORDER = [*QC_LABELS, NOT_ASSESSED]
RANGE = [*(QC_COLOURS[k] for k in QC_LABELS), QC_NOT_ASSESSED]


def well_table(ctx: ReportContext) -> pd.DataFrame:
    """Every well of the plate with its call; a well absent from `qc` is "not assessed"."""
    w = ctx.inputs.wells[["plate", "well", "row", "col", "condition"]].copy()
    w["condition"] = w["condition"].astype(str)
    df = w.merge(ctx.inputs.qc, on=["plate", "well"], how="left")
    df["label"] = df["label"].astype(object).where(df["label"].notna(), NOT_ASSESSED).astype(str)
    df["row"], df["col"] = df["row"].astype(str), df["col"].astype(int)
    return df.sort_values(["plate", "row", "col"], kind="stable").reset_index(drop=True)


def _metrics(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("plate", "well", "row", "col", "condition", "label", "mark", "ink")]


def _plate_chart(ctx: ReportContext, df: pd.DataFrame) -> alt.Chart:
    rows, cols = plate_axes(ctx)
    d = df.assign(mark=df["label"].map(QC_MARKS), ink=df["label"].map(QC_TEXT))
    tip = [alt.Tooltip("well:N"), alt.Tooltip("condition:N"), alt.Tooltip("label:N", title="QC call"),
           *(alt.Tooltip(f"{m}:Q", format=".4g") for m in _metrics(d))]
    x = alt.X("col:O", title=None, sort=cols, axis=alt.Axis(labelAngle=0, orient="top", ticks=False, domain=False))
    y = alt.Y("row:O", title=None, sort=rows, axis=alt.Axis(ticks=False, domain=False))
    width = CELL * len(cols)
    rect = alt.Chart().mark_rect(stroke="white", strokeWidth=0.8).encode(
        x=x, y=y, tooltip=tip,
        color=alt.Color("label:N", title="QC call", scale=alt.Scale(domain=ORDER, range=RANGE),
                        legend=alt.Legend(orient="bottom", direction="horizontal")))
    text = alt.Chart().mark_text(fontSize=11, fontWeight="bold").encode(
        x=x, y=y, text="mark:N",
        color=alt.Color("ink:N", scale=None, legend=None))  # literal colour; avoids merging with the fill scale
    chart = alt.layer(rect, text, data=d).properties(
        width=width, height=CELL * len(rows), title=alt.TitleParams("QC call per well", fontSize=13, anchor="start"))
    if d["plate"].nunique() > 1:
        chart = chart.facet(column=alt.Column("plate:N", title=None))
    return chart


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    allw = df.assign(scope="All wells")
    per = df.assign(scope=df["condition"])
    both = pd.concat([allw, per], ignore_index=True)
    c = both.groupby(["scope", "label"]).size().rename("wells").reset_index()
    return c


def _counts_chart(df: pd.DataFrame, conditions: list[str]) -> alt.Chart:
    c = counts_table(df)
    return alt.Chart(c).mark_bar(stroke="#444", strokeWidth=0.4).encode(
        y=alt.Y("scope:N", title=None, sort=["All wells", *conditions]),
        x=alt.X("wells:Q", title="Wells", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("label:N", title="QC call", scale=alt.Scale(domain=ORDER, range=RANGE)),
        order=alt.Order("label_order:Q"),
        tooltip=[alt.Tooltip("scope:N", title="Group"), alt.Tooltip("label:N", title="QC call"),
                 alt.Tooltip("wells:Q")],
    ).transform_calculate(label_order=f"indexof({ORDER!r}, datum.label)").properties(
        width=360, height=18 * (len(conditions) + 1) + 20,
        title=alt.TitleParams("Wells per QC call", fontSize=13, anchor="start"))


def build(ctx: ReportContext) -> SectionResult:
    from synthdata.report.sections import caption_for

    title = ctx.inputs.qc_title or "Quality control"
    res = SectionResult("qc", title, caption_for(ctx.inputs, "qc"))
    df = well_table(ctx)
    res.figures.append(fig(ctx, "qc", "qc_plate", _plate_chart(ctx, df), "QC call per well"))
    res.figures.append(fig(ctx, "qc", "qc_counts", _counts_chart(df, ctx.conditions), "Wells per QC call"))
    res.tables.append(table(ctx, "qc", "qc_wells", df.drop(columns=["row", "col"])))
    return res
