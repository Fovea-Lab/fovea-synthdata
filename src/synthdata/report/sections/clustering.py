"""Section 5: PCA, well UMAP, Ward dendrogram, cosine similarity and the ARI/NMI card."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from synthdata.report import embed
from synthdata.report.context import ReportContext, SectionResult
from synthdata.report.sections._common import condition_color, fig, table

SIM_MAX_WELLS = 100  # above this the similarity matrix is shown per condition, not per well
POINT = 70


def well_ids(ctx: ReportContext) -> list[str]:
    """Reader-facing well names: the bare well when one plate, else plate/well."""
    idx = ctx.z.index
    single = idx.get_level_values("plate").nunique() == 1
    return [w if single else f"{p}/{w}" for p, w in idx]


def compute(ctx: ReportContext) -> dict | None:
    """PCA, UMAP, Ward and cluster scores, cached on `ctx.embedding` / `ctx.clustering`. None when skipped."""
    if ctx.embedding is not None:
        return ctx.embedding or None
    conds = ctx.profiles["condition"].astype(str)
    if len(ctx.z) < 3 or conds.nunique() < 2:
        ctx.embedding = {}
        ctx.clustering = {}
        return None
    tree = embed.ward(ctx.z, ctx.columns)
    Z, order = tree
    k = int(conds.nunique())
    scores = embed.cluster_scores(Z, conds.to_numpy(), k)
    p = embed.pca(ctx.z, ctx.columns)
    u = embed.umap_embed(np.nan_to_num(ctx.z[ctx.columns].to_numpy(float), nan=0.0))
    ids = well_ids(ctx)
    frame = pd.DataFrame({"well": ids, "condition": conds.to_numpy(),
                          "replicate": ctx.profiles["replicate"].to_numpy()})
    for i in (1, 2, 3):
        name = f"PC{i}"
        frame[name] = p[0][name].to_numpy() if p is not None and name in p[0] else np.nan
    frame["UMAP1"] = u[:, 0] if u is not None else np.nan
    frame["UMAP2"] = u[:, 1] if u is not None else np.nan
    frame["cluster"] = scores["labels"]
    frame["leaf_order"] = np.argsort(np.asarray(order))
    ctx.embedding = {"frame": frame, "pca": p, "umap": u, "ward": tree, "k": k, "ids": ids}
    ctx.clustering = scores
    return ctx.embedding


def _pca(ctx: ReportContext, e: dict):
    _, expl, _ = e["pca"]
    f = e["frame"]
    ax = lambda i: f"PC{i + 1} ({expl[i] * 100:.1f}%)"  # noqa: E731
    return alt.Chart(f).mark_point(filled=True, size=POINT, opacity=0.85, stroke="white", strokeWidth=0.5).encode(
        x=alt.X("PC1:Q", title=ax(0), scale=alt.Scale(zero=False)),
        y=alt.Y("PC2:Q", title=ax(1) if len(expl) > 1 else "PC2", scale=alt.Scale(zero=False)),
        color=condition_color(ctx),
        tooltip=[alt.Tooltip("well:N"), alt.Tooltip("condition:N"), alt.Tooltip("replicate:Q"),
                 alt.Tooltip("PC1:Q", format=".3g"), alt.Tooltip("PC2:Q", format=".3g")]
    ).properties(width=340, height=320, title=alt.TitleParams(
        "PCA of well profiles", subtitle="Robust z against the reference; axes give the variance explained"))


def _umap(ctx: ReportContext, e: dict):
    return alt.Chart(e["frame"]).mark_point(filled=True, size=POINT, opacity=0.85, stroke="white",
                                            strokeWidth=0.5).encode(
        x=alt.X("UMAP1:Q", title="UMAP 1", scale=alt.Scale(zero=False)),
        y=alt.Y("UMAP2:Q", title="UMAP 2", scale=alt.Scale(zero=False)),
        color=condition_color(ctx),
        tooltip=[alt.Tooltip("well:N"), alt.Tooltip("condition:N"), alt.Tooltip("replicate:Q")]
    ).properties(width=340, height=320, title=alt.TitleParams(
        "UMAP of wells", subtitle="Axes have no units; only local distances are meaningful"))


def _dendrogram(ctx: ReportContext, e: dict):
    Z, order = e["ward"]
    n = len(order)
    seg = embed.dendrogram_segments(Z, [e["ids"][i] for i in order])
    seg = seg.round({"x": 3, "y": 4})
    leaves = pd.DataFrame({"x": np.arange(n), "well": [e["ids"][i] for i in order],
                           "condition": [e["frame"]["condition"].iloc[i] for i in order]})
    W = int(min(900, max(420, 8 * n)))
    xs = alt.Scale(domain=[-0.5, n - 0.5], nice=False)
    lines = alt.Chart(seg).mark_line(color="#555", strokeWidth=1, interpolate="linear").encode(
        x=alt.X("x:Q", scale=xs, axis=None), y=alt.Y("y:Q", title="Ward distance"),
        detail="segment:N", order="order:Q").properties(width=W, height=240,
                                                         title="Ward dendrogram of wells (colour bar: true condition)")
    strip = alt.Chart(leaves).mark_rect().encode(
        x=alt.X("x:Q", scale=xs, axis=None), x2=alt.X2("x2:Q"), color=condition_color(ctx),
        tooltip=[alt.Tooltip("well:N"), alt.Tooltip("condition:N")]
    ).transform_calculate(x2="datum.x + 0.5").transform_calculate(x="datum.x - 0.5").properties(width=W, height=16)
    return alt.vconcat(lines, strip, spacing=2)


def _sim_long(ctx: ReportContext, e: dict) -> pd.DataFrame:
    Z, order = e["ward"]
    sim = embed.similarity(ctx.z, ctx.columns, order)
    name = dict(zip(ctx.z.index, e["ids"]))
    return sim.assign(a=sim["a"].map(name), b=sim["b"].map(name))


def _similarity(ctx: ReportContext, e: dict, sim: pd.DataFrame):
    cond_of = dict(zip(e["ids"], e["frame"]["condition"]))
    names = list(dict.fromkeys(sim["a"]))
    n = len(names)
    if n <= SIM_MAX_WELLS:
        pos = {w: i for i, w in enumerate(names)}
        d = pd.DataFrame({"ia": sim["a"].map(pos), "ib": sim["b"].map(pos), "cosine": sim["cosine"].round(3)})
        strip = pd.DataFrame({"i": range(n), "well": names, "condition": [cond_of[w] for w in names]})
        look = lambda k: alt.LookupData(alt.Data(values=strip.to_dict("records")), "i", ["well", "condition"])  # noqa: E731
        cell = int(max(2, min(6, 420 // n)))
        side = cell * n
        col = alt.Color("cosine:Q", title="Cosine similarity",
                        scale=alt.Scale(scheme="blueorange", domain=[-1, 1], domainMid=0),
                        legend=alt.Legend(gradientLength=140))
        heat = alt.Chart(alt.Data(values=d.to_dict('records'))).mark_rect().encode(
            x=alt.X("ib:O", axis=None, sort=list(range(n))), y=alt.Y("ia:O", axis=None, sort=list(range(n))),
            color=col, tooltip=[alt.Tooltip("well_a:N", title="Well A"), alt.Tooltip("cond_a:N", title="Condition A"),
                                alt.Tooltip("well_b:N", title="Well B"), alt.Tooltip("cond_b:N", title="Condition B"),
                                alt.Tooltip("cosine:Q", format=".3f")]
        ).transform_lookup("ia", look(0), as_=["well_a", "cond_a"]).transform_lookup(
            "ib", look(1), as_=["well_b", "cond_b"]).properties(
            width=side, height=side)
        bar = alt.Chart(strip).mark_rect().encode(
            x=alt.X("i:O", axis=None, sort=list(range(n))),
            color=alt.Color("condition:N", title="True condition",
                            scale=alt.Scale(domain=list(ctx.palette), range=list(ctx.palette.values())),
                            legend=alt.Legend(orient="top", direction="horizontal", columns=4, labelLimit=140)),
            tooltip=[alt.Tooltip("well:N"), alt.Tooltip("condition:N")]).properties(width=side, height=14)
        return alt.vconcat(bar, heat, spacing=2).properties(
            title="Well-to-well cosine similarity (dendrogram order; colour bar: true condition)"), None
    agg, order = condition_matrix(sim, cond_of, names)
    chart = alt.Chart(agg).mark_rect().encode(
        x=alt.X("cb:N", title=None, sort=order, axis=alt.Axis(labelAngle=-40)), y=alt.Y("ca:N", title=None, sort=order),
        color=alt.Color("cosine:Q", title="Mean cosine similarity",
                        scale=alt.Scale(scheme="blueorange", domain=[-1, 1], domainMid=0)),
        tooltip=[alt.Tooltip("ca:N", title="Condition A"), alt.Tooltip("cb:N", title="Condition B"),
                 alt.Tooltip("cosine:Q", format=".3f")]).properties(
        width=30 * len(order), height=30 * len(order),
        title="Mean cosine similarity between conditions (well-level matrix in the table)")
    return chart, f"More than {SIM_MAX_WELLS} wells: the similarity figure averages wells per condition; the table keeps every pair."


def condition_matrix(sim: pd.DataFrame, cond_of: dict, names: list) -> tuple[pd.DataFrame, list]:
    """Mean cosine per condition pair, self-pairs (a == b) excluded; conditions in mean dendrogram position."""
    d = sim[sim["a"] != sim["b"]]
    d = d.assign(ca=d["a"].map(cond_of), cb=d["b"].map(cond_of))
    agg = d.groupby(["ca", "cb"], sort=False)["cosine"].mean().reset_index()
    pos = pd.Series({w: i for i, w in enumerate(names)})
    order = list(pos.groupby(pd.Series(cond_of)).mean().sort_values().index)
    agg["cosine"] = agg["cosine"].round(3)
    return agg, order


def _card(ctx: ReportContext, e: dict):
    s = ctx.clustering
    rows = pd.DataFrame({"i": [0, 1, 2, 3], "label": ["Adjusted Rand index", "Normalised mutual information",
                                                       "Clusters (k = conditions)", "Wells"],
                         "value": [f"{s['ari']:.2f}", f"{s['nmi']:.2f}", str(e["k"]), str(len(e["frame"]))]})
    base = alt.Chart(rows).encode(y=alt.Y("i:O", axis=None, sort=[0, 1, 2, 3]))
    lab = base.mark_text(align="left", x=0, fontSize=13, color="#444").encode(text="label:N")
    val = base.mark_text(align="left", x=250, fontSize=18, fontWeight="bold").encode(text="value:N")
    return alt.layer(lab, val).properties(
        width=320, height=110, title="Ward cut at k = number of conditions, against the true conditions"
    ).configure_view(stroke=None)


def _clusters(ctx: ReportContext, e: dict) -> pd.DataFrame:
    f = e["frame"]
    plate = ctx.z.index.get_level_values("plate")
    df = pd.DataFrame({"plate": plate, "well": ctx.z.index.get_level_values("well"), "condition": f["condition"],
                       "cluster": f["cluster"]})
    maj = f.groupby("condition")["cluster"].agg(lambda s: int(s.value_counts().sort_index().idxmax()))
    df["condition_majority_cluster"] = df["condition"].map(maj)
    return df


def build(ctx: ReportContext) -> SectionResult:
    from synthdata.report.sections import caption_for

    res = SectionResult("clustering", "Well profiles and clustering", caption_for(ctx.inputs, "clustering"))
    e = compute(ctx)
    if e is None:
        res.notes.append("Clustering needs at least 3 wells and 2 conditions; this section is skipped.")
        return res
    if e["pca"] is not None:
        res.figures.append(fig(ctx, "clustering", "pca", _pca(ctx, e), "PCA of wells"))
    if e["umap"] is not None:
        res.figures.append(fig(ctx, "clustering", "umap_wells", _umap(ctx, e), "UMAP of wells"))
    else:
        res.notes.append("Fewer than 4 wells: the UMAP is omitted.")
    res.figures.append(fig(ctx, "clustering", "dendrogram", _dendrogram(ctx, e), "Ward dendrogram"))
    sim = _sim_long(ctx, e)
    chart, note = _similarity(ctx, e, sim)
    if note:
        res.notes.append(note)
    res.figures.append(fig(ctx, "clustering", "similarity", chart, "Well similarity"))
    res.figures.append(fig(ctx, "clustering", "cluster_card", _card(ctx, e), "Clustering scores"))
    emb = e["frame"].copy()
    emb.insert(0, "plate", ctx.z.index.get_level_values("plate"))
    res.tables.append(table(ctx, "clustering", "embedding", emb))
    if e["pca"] is not None:
        load = e["pca"][2].rename_axis("column").reset_index()
        load.insert(1, "label", load["column"].map(lambda c: ctx.label(c)))
        res.tables.append(table(ctx, "clustering", "pca_loadings", load))
    if e["pca"] is not None:
        pv = pd.DataFrame({"PC": [f"PC{i + 1}" for i in range(len(e["pca"][1]))], "ratio": e["pca"][1]})
        res.tables.append(table(ctx, "clustering", "pca_variance", pv))
    s = ctx.clustering
    res.tables.append(table(ctx, "clustering", "cluster_scores", pd.DataFrame(
        [{"ari": s["ari"], "nmi": s["nmi"], "k": e["k"], "n_wells": len(e["frame"])}])))
    res.tables.append(table(ctx, "clustering", "clusters", _clusters(ctx, e)))
    res.tables.append(table(ctx, "clustering", "similarity", sim))
    return res
