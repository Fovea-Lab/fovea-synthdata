"""Helpers shared by the section modules."""

from __future__ import annotations

import altair as alt
import pandas as pd

from synthdata.report.context import FigureRecord, ReportContext, TableRecord
from synthdata.report.export import save_figure, save_table


def fig(ctx: ReportContext, section: str, name: str, chart, title: str) -> FigureRecord:
    rec = save_figure(chart, ctx.root / "figures" / section, name, root=ctx.root)
    return FigureRecord(name=name, title=title, spec=rec["spec"], svg=rec["svg"], png=rec["png"])


def table(ctx: ReportContext, section: str, name: str, df: pd.DataFrame) -> TableRecord:
    rec = save_table(df, ctx.root / "tables" / section, name, root=ctx.root)
    return TableRecord(name=name, parquet=rec["parquet"], csv=rec["csv"])


def condition_color(ctx: ReportContext, title: str | None = "Condition", legend: bool = True,
                    field: str = "condition") -> alt.Color:
    return alt.Color(f"{field}:N", title=title,
                     scale=alt.Scale(domain=list(ctx.palette), range=list(ctx.palette.values())),
                     legend=alt.Legend() if legend else None)


def plate_axes(ctx: ReportContext) -> tuple[list[str], list[int]]:
    w = ctx.inputs.wells
    if ctx.inputs.grid:
        nr, nc = ctx.inputs.grid
        rows = [chr(ord("A") + i) for i in range(nr)]
        return rows, list(range(1, nc + 1))
    return sorted(w["row"].astype(str).unique()), sorted(w["col"].astype(int).unique())
