"""Section 7: object-level scatter, size histogram per condition and object UMAP (stratified sample)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from synthdata.report import embed
from synthdata.report.context import ReportContext, SectionResult
from synthdata.report.sections._common import condition_color, fig, table

N_SAMPLE = 5000
BINS = 40


def compute(ctx: ReportContext):
    """Stratified object sample and its UMAP, cached on the context. (None, None) when skipped."""
    if ctx.object_sample is not None:
        return ctx.object_sample, ctx.object_embedding
    objs = ctx.inputs.objects.drop(columns=["condition"], errors="ignore")
    s = embed.object_sample(objs, ctx.inputs.wells, ctx.inputs.features, n=N_SAMPLE, seed=0)
    if s is None:
        return None, None
    u = embed.object_embedding(s, ctx.inputs.features)
    ctx.object_sample, ctx.object_embedding = s, u
    return s, u


def _features(ctx: ReportContext, s: pd.DataFrame) -> tuple[str, str] | None:
    """Two linear features with values in the sample: diameter and eccentricity when present."""
    lin = [f.name for f in ctx.inputs.features if f.kind == "linear" and f.name in s.columns
           and s[f.name].notna().any()]
    if len(lin) < 2:
        return None
    a = "diameter" if "diameter" in lin else lin[0]
    b = "eccentricity" if "eccentricity" in lin else next(n for n in lin if n != a)
    return a, b


def _scatter(ctx: ReportContext, s: pd.DataFrame, fx: str, fy: str):
    d = s[["condition", fx, fy]].dropna().round(4)
    size = d["condition"].map(d["condition"].value_counts())
    d = d.assign(_n=size).sort_values("_n", ascending=False, kind="stable").drop(columns="_n")  # largest drawn first
    total = len(ctx.inputs.objects)
    how = f"all {len(d):,} objects" if total <= N_SAMPLE else f"stratified sample of {len(d):,} objects"
    return alt.Chart(d).mark_circle(size=8, opacity=0.5).encode(
        x=alt.X(f"{fx}:Q", title=ctx.label(fx), scale=alt.Scale(zero=False)),
        y=alt.Y(f"{fy}:Q", title=ctx.label(fy), scale=alt.Scale(zero=False)),
        color=condition_color(ctx),
        tooltip=[alt.Tooltip("condition:N"), alt.Tooltip(f"{fx}:Q", format=".4g"), alt.Tooltip(f"{fy}:Q", format=".4g")]
    ).properties(width=360, height=300,
                 title=alt.TitleParams(f"{ctx.label(fy, False)} against {ctx.label(fx, False).lower()}",
                                       subtitle=how[0].upper() + how[1:]))


def hist_data(ctx: ReportContext, feature: str) -> pd.DataFrame:
    """Per-condition share of objects in 40 shared bins, computed on all objects (pre-binned)."""
    o = ctx.inputs.objects[["plate", "well", feature]].merge(
        ctx.profiles.reset_index()[["plate", "well", "condition"]], on=["plate", "well"])
    o["condition"] = o["condition"].astype(str)
    v = o[feature].dropna().to_numpy(float)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, BINS + 1)
    rows = []
    for c in ctx.conditions:
        x = o.loc[o["condition"] == c, feature].dropna().to_numpy(float)
        counts, _ = np.histogram(x, bins=edges)
        tot = max(int(counts.sum()), 1)
        for i, n in enumerate(counts):
            rows.append({"condition": c, "bin_start": float(edges[i]), "bin_end": float(edges[i + 1]),
                         "count": int(n), "share": float(n / tot)})
    return pd.DataFrame(rows)


def _hist(ctx: ReportContext, h: pd.DataFrame, feature: str):
    lab = ctx.label(feature)
    d = h.assign(bin_start=h["bin_start"].round(4), bin_end=h["bin_end"].round(4), share=h["share"].round(5))
    lo, hi = float(d["bin_start"].min()), float(d["bin_end"].max())
    ncols = 4 if len(ctx.conditions) > 4 else len(ctx.conditions)
    ch = alt.Chart().transform_calculate(zero="0").mark_bar(stroke="white", strokeWidth=0.3).encode(
        x=alt.X("bin_start:Q", title=lab, scale=alt.Scale(domain=[lo, hi], nice=False)), x2="bin_end:Q",
        y=alt.Y("share:Q", title=None, axis=alt.Axis(format="%"), stack=None,
                scale=alt.Scale(domain=[0, float(d['share'].max()) * 1.08], nice=False)),
        y2=alt.Y2("zero:Q"),
        color=condition_color(ctx, legend=False),
        tooltip=[alt.Tooltip("condition:N"), alt.Tooltip("bin_start:Q", title="From", format=".4g"),
                 alt.Tooltip("bin_end:Q", title="To", format=".4g"), alt.Tooltip("count:Q", title="Objects"),
                 alt.Tooltip("share:Q", title="Share", format=".1%")]).properties(width=200, height=120)
    return ch.facet(data=d, facet=alt.Facet("condition:N", sort=ctx.conditions, title=None,
                                            header=alt.Header(labelFontSize=12)),
                    columns=ncols, spacing=14,
                    title=alt.TitleParams(f"{lab} by condition: share of the condition's objects per bin",
                                          subtitle=f"{BINS} shared bins spanning the observed minimum to maximum")
                    ).resolve_scale(x="shared", y="shared")


def _umap(ctx: ReportContext, s: pd.DataFrame, u: pd.DataFrame):
    d = pd.concat([s[["condition"]].reset_index(drop=True), u.reset_index(drop=True)], axis=1).round(3)
    return alt.Chart(d).mark_circle(size=8, opacity=0.5).encode(
        x=alt.X("u1:Q", title="UMAP 1", scale=alt.Scale(zero=False)),
        y=alt.Y("u2:Q", title="UMAP 2", scale=alt.Scale(zero=False)),
        color=condition_color(ctx), tooltip=[alt.Tooltip("condition:N")]
    ).properties(width=360, height=300, title=alt.TitleParams(
        "UMAP of sampled objects", subtitle="Standardised features; angle as cos 2θ, sin 2θ; axes have no units"))


def build(ctx: ReportContext) -> SectionResult:
    from synthdata.report.sections import caption_for

    res = SectionResult("objects", "Object-level views", caption_for(ctx.inputs, "objects"))
    s, u = compute(ctx)
    if s is None:
        res.notes.append("Object-level views need at least 3 wells and 2 conditions; this section is skipped.")
        return res
    pair = _features(ctx, s)
    if pair is None:
        res.notes.append("Fewer than two linear features have values; the scatter and histogram are omitted.")
    else:
        fx, fy = pair
        sname = "diameter_vs_eccentricity" if (fx, fy) == ("diameter", "eccentricity") else f"{fx}_vs_{fy}"
        hname = "two_peaks" if fx == "diameter" else f"{fx}_histogram"
        res.figures.append(fig(ctx, "objects", sname, _scatter(ctx, s, fx, fy),
                               f"{ctx.label(fx, False)} against {ctx.label(fy, False).lower()}"))
        h = hist_data(ctx, fx)
        res.figures.append(fig(ctx, "objects", hname, _hist(ctx, h, fx), f"{ctx.label(fx, False)} histogram per condition"))
        res.tables.append(table(ctx, "objects", f"{fx}_histogram" if fx != "diameter" else "diameter_histogram", h))
    if u is not None:
        res.figures.append(fig(ctx, "objects", "umap_objects", _umap(ctx, s, u), "Object UMAP"))
    else:
        res.notes.append("Too few objects for an object UMAP.")
    # the two scatters sit side by side in the page; the full-width histogram grid goes last
    res.figures.sort(key=lambda f: f.name == "two_peaks" or f.name.endswith("_histogram"))
    t = s.copy()
    if u is not None:
        t = pd.concat([t.reset_index(drop=True), u.reset_index(drop=True)], axis=1)
    res.tables.append(table(ctx, "objects", "object_sample", t))
    return res
