"""Section 4: condition x profile-column z fingerprint, control-vs-twin dot plot, effects table."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from synthdata.report.context import ReportContext, SectionResult
from synthdata.report.sections._common import fig, table

CELL_W, CELL_H = 54, 32
CLAMP = 10.0


def _long(ctx: ReportContext) -> pd.DataFrame:
    z = ctx.z.copy()
    z["condition"] = ctx.profiles["condition"].astype(str).to_numpy()
    med = z.groupby("condition", sort=False)[ctx.columns].median()
    med = med.loc[[c for c in ctx.conditions if c != ctx.inputs.reference]]
    long = med.reset_index().melt(id_vars="condition", var_name="column", value_name="z")
    e = ctx.effects[["condition", "column", "d", "q", "flagged"]]
    long = long.merge(e, on=["condition", "column"], how="left")
    long["label"] = long["column"].map(lambda c: ctx.label(c))
    long["z_clamped"] = long["z"].clip(-CLAMP, CLAMP)
    long["abs_z"] = long["z_clamped"].abs()
    long["flagged"] = long["flagged"].fillna(False).astype(bool)
    return long


def _fingerprint(ctx: ReportContext, long: pd.DataFrame):
    labels = [ctx.label(c) for c in ctx.columns]
    conds = [c for c in ctx.conditions if c != ctx.inputs.reference]
    base = alt.Chart(long).encode(
        x=alt.X("label:N", title=None, sort=labels, axis=alt.Axis(labelAngle=-40, labelLimit=220)),
        y=alt.Y("condition:N", title=None, sort=conds),
        tooltip=[alt.Tooltip("condition:N"), alt.Tooltip("label:N", title="Column"),
                 alt.Tooltip("z:Q", title="Median z", format=".3g"),
                 alt.Tooltip("d:Q", title="Effect size d", format=".3g"),
                 alt.Tooltip("q:Q", title="q", format=".3g"), alt.Tooltip("flagged:N", title="Flagged")])
    rect = base.mark_rect(stroke="white", strokeWidth=1).encode(
        color=alt.Color("z_clamped:Q", title="Median robust z",
                        scale=alt.Scale(scheme="redblue", reverse=True, domain=[-CLAMP, CLAMP], domainMid=0,
                                        clamp=True),
                        legend=alt.Legend(gradientLength=160, orient="right")))
    dots = (base.transform_filter(alt.datum.flagged == True)  # noqa: E712
            .mark_text(text="●", fontSize=14, baseline="middle")
            .encode(color=alt.condition(alt.datum.abs_z > 6, alt.value("white"), alt.value("#222"))))
    return alt.layer(rect, dots).properties(
        width=CELL_W * len(labels), height=CELL_H * len(conds),
        title=f"Fingerprint against {ctx.inputs.reference} (dot: q < 0.05 and abs(d) ≥ 1; colour clamped at ±10)")


def _twin(ctx: ReportContext):
    w = ctx.inputs.wells
    roles = dict(zip(w["condition"].astype(str), w["role"].astype(str)))
    twins = [c for c in ctx.conditions if c != ctx.inputs.reference and roles.get(c) == "control"]
    if not twins:
        return None, twins
    e = ctx.effects[ctx.effects["condition"].isin(twins)].copy()
    e["abs_d"] = e["d"].abs()
    e["abs_d_shown"] = e["abs_d"].clip(upper=5.0)
    e["label"] = e["column"].map(lambda c: ctx.label(c))
    labels = [ctx.label(c) for c in ctx.columns]
    dots = alt.Chart(e).mark_point(filled=True, size=90, opacity=0.9).encode(
        x=alt.X("abs_d_shown:Q", title="Absolute effect size abs(d) against the reference (shown up to 5)", scale=alt.Scale(domain=[0, 5], nice=False)),
        y=alt.Y("label:N", title=None, sort=labels),
        color=alt.Color("condition:N", title="Twin condition",
                        scale=alt.Scale(domain=twins, range=[ctx.palette[c] for c in twins]),
                        legend=alt.Legend() if len(twins) > 1 else None),
        tooltip=[alt.Tooltip("condition:N"), alt.Tooltip("label:N", title="Column"),
                 alt.Tooltip("abs_d:Q", title="abs(d)", format=".3g"),
                 alt.Tooltip("q:Q", title="q", format=".3g"), alt.Tooltip("flagged:N", title="Flagged")])
    rule = alt.Chart(pd.DataFrame({"x": [1.0], "note": ["abs(d) = 1: the effect-size threshold"]})).mark_rule(
        color="#c0392b", strokeDash=[5, 4], strokeWidth=1.5).encode(
        x="x:Q", tooltip=[alt.Tooltip("note:N", title="Threshold")])
    below = bool((e["abs_d"].dropna() < 1).all())
    verdict = "every effect size stays below the abs(d) = 1 line" if below else "effect sizes against the abs(d) = 1 line"
    return alt.layer(dots, rule).properties(
        width=420, height=22 * len(labels),
        title=f"{', '.join(twins)} against {ctx.inputs.reference}: {verdict}"), twins


def build(ctx: ReportContext) -> SectionResult:
    from synthdata.report.sections import caption_for

    res = SectionResult("fingerprints", "Fingerprints", caption_for(ctx.inputs, "fingerprints"))
    long = _long(ctx)
    res.figures.append(fig(ctx, "fingerprints", "fingerprint", _fingerprint(ctx, long), "Condition fingerprint"))
    twin, twins = _twin(ctx)
    if twin is not None:
        res.figures.append(fig(ctx, "fingerprints", "twin_vs_reference", twin, "Twin against the reference"))
    else:
        res.notes.append("No non-reference condition has the control role; the twin plot is omitted.")
    res.tables.append(table(ctx, "fingerprints", "effects", ctx.effects))
    return res
