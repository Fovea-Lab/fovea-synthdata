"""Section 3: per-condition violins with the intended pdf, orientation rose, quantile table."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from synthdata.report.context import ReportContext, SectionResult
from synthdata.report.profiles import sarle_bimodality
from synthdata.report.sections._common import condition_color, fig, table

VIOLIN_W, VIOLIN_H = 96, 260
ROSE = 130
ROSE_COLUMNS = 4
BIN_DEG = 10


def _obs(ctx: ReportContext, feature: str) -> pd.DataFrame:
    o = ctx.inputs.objects[["plate", "well", feature]].merge(
        ctx.profiles.reset_index()[["plate", "well", "condition"]], on=["plate", "well"])
    o["condition"] = o["condition"].astype(str)
    return o[["condition", feature]].dropna()


def _scott(v: np.ndarray) -> float:
    n = len(v)
    if n < 2:
        return np.nan
    iqr = np.subtract(*np.percentile(v, [75, 25])) / 1.34
    spread = min(v.std(ddof=1), iqr) if iqr > 0 else v.std(ddof=1)
    return float(1.06 * spread * n ** -0.2)


def _kde(v: np.ndarray, xs: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian KDE of v on the grid xs (numpy only)."""
    if len(v) == 0:
        return np.full(len(xs), np.nan)
    out = np.zeros(len(xs))
    for i in range(0, len(v), 2000):
        z = (xs[:, None] - v[None, i:i + 2000]) / bw
        out += np.exp(-0.5 * z * z).sum(axis=1)
    return out / (len(v) * bw * np.sqrt(2 * np.pi))


def _condition_bandwidth(v: np.ndarray, scott: float, base: float) -> float:
    """The feature's base bandwidth, widened up to 2x toward the condition's own Scott bandwidth.

    Only for a unimodal sample (Sarle's coefficient below 5/9): Scott's rule assumes one mode and would blur a
    mixture, so a multimodal condition keeps the base bandwidth.
    """
    if not (np.isfinite(scott) and scott > 0) or len(v) < 4 or not sarle_bimodality(v) < 5 / 9:
        return base
    return float(np.clip(scott, base, 2 * base))


def densities(ctx: ReportContext, feature: str) -> pd.DataFrame:
    """condition x 200-step grid: observed KDE and the intended pdf.

    The base bandwidth is set per feature (median of the per-condition Scott bandwidths). A unimodal condition may
    widen it up to twice, toward its own Scott bandwidth, so a broad condition (large sd) is not drawn as sampling
    noise; a narrow or multimodal one keeps the base, and all violins stay comparable.
    The grid runs three bandwidths past the extreme 0.1 % / 99.9 % quantiles so the tails taper instead of stopping.
    """
    obs = _obs(ctx, feature)
    groups = {c: obs.loc[obs["condition"] == c, feature].to_numpy(float) for c in ctx.conditions}
    scott = {c: _scott(v) for c, v in groups.items()}
    bws = [b for b in scott.values() if np.isfinite(b) and b > 0]
    q = obs.groupby("condition")[feature].quantile([0.001, 0.999]).unstack()
    lo, hi = float(q.iloc[:, 0].min()), float(q.iloc[:, 1].max())
    if hi <= lo:
        hi = lo + 1.0
    bw = float(np.median(bws)) if bws else (hi - lo) / 20
    lo, hi = lo - 3 * bw, hi + 3 * bw
    if len(obs) and obs[feature].min() >= 0:  # a non-negative feature's axis does not run below zero
        lo = max(lo, 0.0)
    xs = np.linspace(lo, hi, 200)
    dists = (ctx.inputs.intended or {}).get(feature) or {}
    frames = []
    for c in ctx.conditions:
        bw_c = _condition_bandwidth(groups[c], scott[c], bw)
        ipdf = np.asarray(dists[c].pdf(xs), float) if c in dists else np.full(200, np.nan)
        frames.append(pd.DataFrame({"condition": c, "feature": feature, "value": xs,
                                    "density_observed": _kde(groups[c], xs, bw_c),
                                    "density_intended": ipdf, "bandwidth": bw_c}))
    return pd.concat(frames, ignore_index=True)


def _sig(x: pd.Series, n: int = 4) -> pd.Series:
    return x.map(lambda v: float(f"{v:.{n}g}") if pd.notna(v) else None).astype(object)


def _violin(ctx: ReportContext, feature: str, dens: pd.DataFrame):
    label = ctx.label(feature)
    d = dens.copy()
    peak = float(np.nanmax(np.r_[d["density_observed"].to_numpy(), d["density_intended"].to_numpy()]))
    has_pdf = bool(d["density_intended"].notna().any())
    ip = d["density_intended"]
    # drop rows whose densities are invisible (<1% of the peak) to keep the spec small
    keep = (d["density_observed"] >= 0.01 * peak) | (ip >= 0.01 * peak)
    d = d[keep.fillna(False)]
    lo, hi = float(d["value"].min()), float(d["value"].max())
    # the outline starts where the intended pdf is >= 1% of its own peak (no vertical stem)
    ipk = d.groupby("condition")["density_intended"].transform("max")
    d["density_intended"] = d["density_intended"].where(d["density_intended"] >= 0.01 * ipk)
    # outline segment id: a new segment after every gap, so a mixture's two lobes are not joined by a stem
    pos = pd.Series(d.index, index=d.index)[d["density_intended"].notna()]
    seg = (pos.groupby(d["condition"]).diff() != 1).groupby(d["condition"]).cumsum().reindex(d.index).fillna(0)
    # area segment id: the area is also broken wherever rows were trimmed, so no hairline bridges a gap
    area_seg = (pd.Series(d.index, index=d.index).groupby(d["condition"]).diff() != 1).groupby(d["condition"]).cumsum()
    cols = {"c": d["condition"].tolist(), "g": [int(x) or None for x in seg], "ga": [int(x) - 1 or None for x in area_seg], "v": d["value"].map(lambda x: float(f"{x:.4g}")).tolist(),
            "o": d["density_observed"].map(lambda x: float(f"{x:.4g}")).tolist(),
            "i": [float(f"{x:.4g}") if pd.notna(x) else None for x in d["density_intended"]]}
    # plain records; a missing intended value is omitted (never NaN, which is not valid JSON)
    data = alt.Data(values=[{k: v[n] for k, v in cols.items() if v[n] is not None} for n in range(len(d))])
    y = alt.Y("v:Q", title=label, scale=alt.Scale(domain=[lo, hi], nice=True, zero=False))

    area = (alt.Chart().transform_calculate(lower="-datum.o / 2", upper="datum.o / 2")
            .mark_area(orient="horizontal", opacity=0.85, stroke="white", strokeWidth=0.5)
            .encode(y=y,
                    x=alt.X("lower:Q", title=None, scale=alt.Scale(zero=False),
                            axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False)),
                    x2="upper:Q", detail="ga:O",
                    color=condition_color(ctx, legend=False, field="c"),
                    tooltip=[alt.Tooltip("c:N", title="Condition"), alt.Tooltip("v:Q", title=label, format=".4g"),
                             alt.Tooltip("o:Q", title="Observed density", format=".3g")]))
    layers = [area]
    if has_pdf:
        line = (alt.Chart().transform_filter("isValid(datum.i)")
                .transform_calculate(upper="datum.i / 2", lower="-datum.i / 2")
                .transform_fold(["upper", "lower"], as_=["side", "half"])
                .mark_line(color="#222", strokeWidth=1.2)
                .encode(y=alt.Y("v:Q"), x=alt.X("half:Q"), detail=["side:N", "g:O"], order=alt.Order("v:Q"),
                        tooltip=[alt.Tooltip("c:N", title="Condition"), alt.Tooltip("v:Q", title=label, format=".4g"),
                                 alt.Tooltip("i:Q", title="Intended density", format=".3g")]))
        layers.append(line)
    title = f"{label} by condition" + (" (dark line: intended distribution)" if has_pdf else "")
    return (alt.layer(*layers, width=VIOLIN_W, height=VIOLIN_H)
            .facet(data=data, column=alt.Column("c:N", sort=ctx.conditions, title=None,
                                                header=alt.Header(labelAngle=0, labelFontSize=11, labelPadding=4)),
                   spacing=4, title=title))


def rose_data(ctx: ReportContext, feature: str = "orientation") -> pd.DataFrame:
    """Axial angle counts per condition in 10-degree bins, mirrored to the full circle."""
    obs = _obs(ctx, feature)
    edges = np.arange(0, 180 + BIN_DEG, BIN_DEG)
    rows = []
    for c in ctx.conditions:
        a = np.mod(obs.loc[obs["condition"] == c, feature].to_numpy(float), 180.0)
        counts, _ = np.histogram(a, bins=edges)
        for i, n in enumerate(counts):
            for k, off in enumerate((0, 180)):
                start = float((edges[i] + 90 + off) % 360)  # vega theta 0 is 12 o'clock; image 0 deg is +x
                rows.append({"condition": c, "bin": f"{edges[i]:.0f}-{edges[i + 1]:.0f}",
                             "start": start, "end": start + BIN_DEG,
                             "count": int(n), "fraction": float(n / max(counts.sum(), 1)), "mirror": bool(k)})
    return pd.DataFrame(rows)


def _rose(ctx: ReportContext, feature: str):
    label = ctx.label(feature)
    df = rose_data(ctx, feature)
    ang = alt.Scale(domain=[0, 360], range=[0, 2 * np.pi])
    rmax = float(df["fraction"].max()) or 1.0
    base = alt.Chart().encode(
        theta=alt.Theta("start:Q", scale=ang, stack=None),
        theta2=alt.Theta2("end:Q"),
        radius=alt.Radius("fraction:Q", scale=alt.Scale(type="sqrt", zero=True, domain=[0, rmax], range=[0, ROSE / 2 - 4]),
                          stack=None),
        color=condition_color(ctx, legend=False),
        tooltip=[alt.Tooltip("condition:N"), alt.Tooltip("bin:N", title=f"{label} bin"),
                 alt.Tooltip("count:Q", title="Objects"),
                 alt.Tooltip("fraction:Q", title="Share of condition", format=".1%")])
    arcs = base.mark_arc(innerRadius=0, stroke="white", strokeWidth=0.5, opacity=0.9)
    return (arcs.properties(width=ROSE, height=ROSE)
            .facet(data=df, facet=alt.Facet("condition:N", sort=ctx.conditions, title=None,
                                            header=alt.Header(labelFontSize=12)),
                   columns=ROSE_COLUMNS, spacing=10,
                   title=f"{label} by condition (10° bins; radius = share of the condition's objects)"))


def _quantiles(ctx: ReportContext) -> pd.DataFrame:
    rows = []
    for f in ctx.inputs.features:
        obs = _obs(ctx, f.name)
        for c in ctx.conditions:
            v = obs.loc[obs["condition"] == c, f.name].to_numpy(float)
            q = np.percentile(v, [5, 25, 50, 75, 95]) if len(v) else [np.nan] * 5
            rows.append({"condition": c, "feature": f.name, "q05": q[0], "q25": q[1], "q50": q[2],
                         "q75": q[3], "q95": q[4], "n": int(len(v))})
    return pd.DataFrame(rows)


def build(ctx: ReportContext) -> SectionResult:
    from synthdata.report.sections import caption_for

    res = SectionResult("distributions", "Distributions", caption_for(ctx.inputs, "distributions"))
    dens_frames = []
    for f in ctx.inputs.features:
        if f.kind == "linear":
            d = densities(ctx, f.name)
            dens_frames.append(d)
            res.figures.append(fig(ctx, "distributions", f"dist_{f.name}", _violin(ctx, f.name, d),
                                   f"{f.label} distribution"))
        elif f.kind == "axial":
            res.figures.append(fig(ctx, "distributions", f"rose_{f.name}", _rose(ctx, f.name),
                                   f"{f.label} rose"))
    res.figures.sort(key=lambda f: f.name.startswith("rose_"))  # violins together, the rose last
    if not (ctx.inputs.intended or {}):
        res.notes.append("No intended distributions supplied; violins show observed data only.")
    if dens_frames:
        res.tables.append(table(ctx, "distributions", "densities", pd.concat(dens_frames, ignore_index=True)))
    res.tables.append(table(ctx, "distributions", "condition_quantiles", _quantiles(ctx)))
    return res
