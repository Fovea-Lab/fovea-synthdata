import json
from pathlib import Path

import altair as alt
import jsonschema
import pandas as pd
import pytest
from altair.vegalite.v6.schema import load_schema

from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report.context import ReportContext
from synthdata.report.sections import distributions, fingerprints, heatmaps, overview

SCHEMA = load_schema()


@pytest.fixture(scope="module")
def ctx_small(small_truth_dataset, tmp_path_factory):
    return ReportContext.create(truth_inputs(small_truth_dataset), tmp_path_factory.mktemp("rep_small"))


@pytest.fixture(scope="module")
def ctx_rep(replicate_truth_dataset, tmp_path_factory):
    return ReportContext.create(truth_inputs(replicate_truth_dataset), tmp_path_factory.mktemp("rep_rep"))


def _check_figures(ctx, res, min_figs=1):
    assert len(res.figures) >= min_figs
    for f in res.figures:
        assert (ctx.root / f.png).exists()
        if f.spec is None:
            continue
        assert (ctx.root / f.svg).exists()
        jsonschema.validate(json.loads((ctx.root / f.spec).read_text()), SCHEMA)


def _cols(ctx, res, name):
    t = next(t for t in res.tables if t.name == name)
    assert (ctx.root / t.parquet).exists() and (ctx.root / t.csv).exists()
    return pd.read_parquet(ctx.root / t.parquet)


def test_overview(ctx_small):
    res = overview.build(ctx_small)
    _check_figures(ctx_small, res, 1 + len(ctx_small.conditions))
    names = [f.name for f in res.figures]
    assert names[0] == "plate_layout"
    tiles = [f for f in res.figures if f.spec is None]
    assert len(tiles) == len(ctx_small.conditions)
    assert all(f.png.startswith("figures/overview/example_") for f in tiles)
    conds = _cols(ctx_small, res, "conditions")
    assert list(conds.columns) == ["condition", "role", "n_wells", "n_objects"]
    assert conds["n_wells"].sum() == len(ctx_small.profiles)
    summ = _cols(ctx_small, res, "summary")
    assert set(summ["metric"]) >= {"conditions", "wells", "objects", "sites per well", "reference"}


def test_overview_tiles_share_a_stretch(ctx_small):
    from PIL import Image
    import numpy as np
    res = overview.build(ctx_small)
    arrs = [np.asarray(Image.open(ctx_small.root / f.png)) for f in res.figures if f.spec is None]
    assert all(a.dtype == np.uint8 for a in arrs)
    assert min(a.min() for a in arrs) == 0 and max(a.max() for a in arrs) == 255  # shared, not per-tile


def test_heatmaps(ctx_small):
    res = heatmaps.build(ctx_small)
    _check_figures(ctx_small, res, 7)
    assert {f.name for f in res.figures} == {f"plate_{c}" for c in heatmaps.HEAT_COLUMNS}
    spec = json.loads((ctx_small.root / next(f for f in res.figures if f.name == "plate_clark_evans").spec).read_text())
    scale = spec["layer"][0]["encoding"]["color"]["scale"] if "layer" in spec else spec["encoding"]["color"]["scale"]
    assert scale["scheme"] == "blueorange" and scale["domainMid"] == 1
    wp = _cols(ctx_small, res, "well_profiles")
    assert "clark_evans" in wp.columns and len(wp) == len(ctx_small.profiles)


def test_distributions(ctx_small):
    res = distributions.build(ctx_small)
    names = {f.name for f in res.figures}
    assert {"dist_diameter", "dist_eccentricity", "dist_intensity", "rose_orientation"} <= names
    _check_figures(ctx_small, res, 4)
    rose = json.loads((ctx_small.root / next(f for f in res.figures if f.name == "rose_orientation").spec).read_text())
    n_facets = len(ctx_small.conditions)
    assert n_facets == 8
    assert rose["facet"]["field"] == "condition" and rose["columns"] == distributions.ROSE_COLUMNS
    data = rose.get("datasets") and next(iter(rose["datasets"].values()))
    assert len({r["condition"] for r in data}) == 8
    q = _cols(ctx_small, res, "condition_quantiles")
    assert {"condition", "feature", "q05", "q25", "q50", "q75", "q95", "n"} <= set(q.columns)
    assert len(q) == 8 * 4


def test_violin_layers_intended_pdf(ctx_small):
    res = distributions.build(ctx_small)
    spec = json.loads((ctx_small.root / next(f for f in res.figures if f.name == "dist_diameter").spec).read_text())
    marks = [l["mark"] if isinstance(l["mark"], str) else l["mark"]["type"] for l in spec["spec"]["layer"]]
    assert marks == ["area", "line"]


def test_fingerprints(ctx_rep):
    res = fingerprints.build(ctx_rep)
    _check_figures(ctx_rep, res, 2)
    assert {f.name for f in res.figures} == {"fingerprint", "twin_vs_reference"}
    e = _cols(ctx_rep, res, "effects")
    assert {"median_diff", "d", "p", "q", "flagged"} <= set(e.columns)
    row = e[(e["condition"] == "large") & (e["column"] == "diameter_median")].iloc[0]
    assert bool(row["flagged"])
    twin = e[e["condition"] == "control_twin"]
    assert not twin["flagged"].any() and (twin["d"].abs() < 1).all()


def test_fingerprint_excludes_reference(ctx_small):
    res = fingerprints.build(ctx_small)
    spec = json.loads((ctx_small.root / next(f for f in res.figures if f.name == "fingerprint").spec).read_text())
    data = next(iter(spec["datasets"].values()))
    assert ctx_small.inputs.reference not in {r["condition"] for r in data}
    assert max(abs(r["z_clamped"]) for r in data if r["z_clamped"] is not None) <= 10


def test_context_palette_and_columns(ctx_small):
    assert list(ctx_small.palette) == ctx_small.conditions
    assert ctx_small.label("diameter_median") == "Diameter median (px)"
    assert ctx_small.label("clark_evans") == "Clark-Evans index"


def _peak(df, col):
    return df.groupby("condition")[col].max()


def test_violin_kde_matches_intended_peak(ctx_rep):
    res = distributions.build(ctx_rep)
    d = _cols(ctx_rep, res, "densities")
    d = d[d["feature"] == "diameter"]
    obs, int = _peak(d, "density_observed"), _peak(d, "density_intended")
    for c in ctx_rep.conditions:
        assert abs(obs[c] - int[c]) / int[c] < 0.25, (c, obs[c], int[c])


def test_violin_specs_are_small(ctx_rep):
    res = distributions.build(ctx_rep)
    for f in res.figures:
        if f.name.startswith("dist_"):
            assert (ctx_rep.root / f.spec).stat().st_size < 100_000, f.name
            spec = json.loads((ctx_rep.root / f.spec).read_text())
            assert "transform_density" not in json.dumps(spec) and "density" not in json.dumps(spec.get("transform", []))


def test_rose_orientation_matches_image_frame(ctx_rep):
    df = distributions.rose_data(ctx_rep)
    a = df[(df["condition"] == "aligned") & (~df["mirror"])]
    top = a.loc[a["fraction"].idxmax()]
    mid = (top["start"] + 5) % 180
    assert abs(mid - 135) <= 10, mid
    sums = df[~df["mirror"]].groupby("condition")["fraction"].sum()
    assert (abs(sums - 1) < 1e-9).all()


# ---- part 2: sections 5-7 ----
def test_clustering(ctx_rep):
    from synthdata.report.sections import clustering

    res = clustering.build(ctx_rep)
    _check_figures(ctx_rep, res, 5)
    assert [f.name for f in res.figures] == ["pca", "umap_wells", "dendrogram", "similarity", "cluster_card"]
    cl = _cols(ctx_rep, res, "clusters")
    assert len(cl) == len(ctx_rep.profiles)
    maj = cl.groupby("condition")["cluster"].agg(lambda s: s.value_counts().idxmax())
    assert maj["control"] == maj["control_twin"]
    emb = _cols(ctx_rep, res, "embedding")
    assert {"well", "condition", "PC1", "PC2", "PC3", "UMAP1", "UMAP2", "cluster"} <= set(emb.columns)
    assert len(_cols(ctx_rep, res, "pca_loadings")) == len(ctx_rep.columns)
    assert len(_cols(ctx_rep, res, "similarity")) == len(ctx_rep.profiles) ** 2
    assert ctx_rep.clustering["ari"] > 0.5
    spec = json.loads((ctx_rep.root / res.figures[0].spec).read_text())
    assert "PC1 (" in json.dumps(spec) and "%)" in json.dumps(spec)


def test_clustering_skips_tiny_input(ctx_small):
    from synthdata.report.contract import ReportInputs
    from synthdata.report.sections import clustering

    i = ctx_small.inputs
    w = i.wells.iloc[:3].copy()
    w["condition"] = "control"
    w["condition"] = pd.Categorical(w["condition"])
    objs = i.objects.merge(w[["plate", "well"]], on=["plate", "well"])
    tiny = ReportContext.create(ReportInputs(objects=objs, wells=w, features=i.features, reference="control",
                                             n_sites=i.n_sites), ctx_small.root / "tiny")
    res = clustering.build(tiny)
    assert res.figures == [] and res.notes


def test_replicates(ctx_rep):
    from synthdata.report.sections import replicates

    res = replicates.build(ctx_rep)
    _check_figures(ctx_rep, res, 2)
    m = _cols(ctx_rep, res, "map")
    assert len(m) == len(ctx_rep.conditions)
    assert m["mAP"].between(0, 1).all()
    assert (m["n_wells"] == 12).all()
    assert {"condition", "column", "cv"} <= set(_cols(ctx_rep, res, "replicate_cv").columns)
    spec = json.loads((ctx_rep.root / next(f for f in res.figures if f.name == "map_bars").spec).read_text())
    assert spec["layer"][0]["encoding"]["x"]["scale"]["domain"] == [0, 1]


def test_objects(ctx_rep):
    from synthdata.report.sections import objects

    res = objects.build(ctx_rep)
    _check_figures(ctx_rep, res, 3)
    s = _cols(ctx_rep, res, "object_sample")
    assert 0 < len(s) <= 5000 and {"u1", "u2", "condition"} <= set(s.columns)
    # histogram is pre-binned in Python: no vega bin transform, bounded rows
    spec = json.loads((ctx_rep.root / next(f for f in res.figures if f.name == "two_peaks").spec).read_text())
    txt = json.dumps(spec)
    assert '"bin": true' not in txt and '"bin":true' not in txt
    assert len(ctx_rep.conditions) * 40 <= len(spec["datasets"][spec["data"]["name"]])
    assert len(spec["datasets"][spec["data"]["name"]]) == len(ctx_rep.conditions) * 40


def test_histogram_edges_are_honest(ctx_rep):
    import numpy as np

    from synthdata.report.sections import objects

    h = objects.hist_data(ctx_rep, "diameter")
    o = ctx_rep.inputs.objects.merge(ctx_rep.profiles.reset_index()[["plate", "well", "condition"]],
                                     on=["plate", "well"])
    o["condition"] = o["condition"].astype(str)
    assert h["bin_start"].min() == o["diameter"].min() and h["bin_end"].max() == o["diameter"].max()
    for c, g in h.groupby("condition"):
        v = o.loc[o["condition"] == c, "diameter"].to_numpy(float)
        assert g["count"].sum() == len(v)
        for _, r in g.iterrows():
            inside = ((v >= r["bin_start"]) & (v < r["bin_end"])).sum() + (
                (v == r["bin_end"]).sum() if r["bin_end"] == g["bin_end"].max() else 0)
            assert r["count"] <= inside


def test_figure_number_tables_exist(ctx_rep):
    from synthdata.report.sections import clustering, objects

    res = clustering.build(ctx_rep)
    assert list(_cols(ctx_rep, res, "pca_variance").columns) == ["PC", "ratio"]
    cs = _cols(ctx_rep, res, "cluster_scores")
    assert list(cs.columns) == ["ari", "nmi", "k", "n_wells"] and len(cs) == 1
    assert sorted(_cols(ctx_rep, res, "embedding")["leaf_order"]) == list(range(len(ctx_rep.profiles)))
    res = objects.build(ctx_rep)
    assert {"condition", "bin_start", "bin_end", "count", "share"} <= set(_cols(ctx_rep, res, "diameter_histogram").columns)


def test_condition_matrix_excludes_self_pairs():
    from synthdata.report.sections.clustering import condition_matrix

    names = ["w1", "w2"]
    sim = pd.DataFrame({"a": ["w1", "w1", "w2", "w2"], "b": ["w1", "w2", "w1", "w2"], "cosine": [1.0, 0.2, 0.2, 1.0]})
    agg, _ = condition_matrix(sim, {"w1": "c", "w2": "c"}, names)
    assert agg.loc[0, "cosine"] == 0.2


def test_large_plate_similarity_falls_back_with_note(tmp_path):
    import numpy as np

    from synthdata.report.contract import ReportInputs
    from synthdata.report.sections import clustering

    rng = np.random.default_rng(0)
    wells, objs = [], []
    for i in range(120):
        w = f"W{i:03d}"
        cond = ["control", "a", "b"][i % 3]
        wells.append({"plate": "p", "well": w, "row": "A", "col": i + 1, "condition": cond,
                      "role": "control" if cond == "control" else "treatment", "replicate": i // 3 + 1})
        for k in range(6):
            objs.append({"plate": "p", "well": w, "site": 1, "object_id": k, "x": 1.0, "y": 1.0,
                         "diameter": 10 + (i % 3) * 3 + rng.normal(), "eccentricity": rng.uniform(0, 1),
                         "orientation": rng.uniform(0, 180), "intensity": 100 + rng.normal()})
    wells = pd.DataFrame(wells)
    ctx = ReportContext.create(ReportInputs(objects=pd.DataFrame(objs), wells=wells, reference="control"), tmp_path)
    res = clustering.build(ctx)
    assert any("More than 100 wells" in n for n in res.notes)
    assert len(_cols(ctx, res, "similarity")) == 120 * 120


def test_objects_robust_without_features(tmp_path, ctx_small):
    from synthdata.report.contract import FeatureSpec, ReportInputs
    from synthdata.report.sections import objects

    i = ctx_small.inputs
    feats = (FeatureSpec("orientation", "axial", "Orientation", "deg"),)
    ctx = ReportContext.create(ReportInputs(objects=i.objects, wells=i.wells, features=feats, reference=i.reference,
                                            n_sites=i.n_sites), tmp_path)
    res = objects.build(ctx)
    assert res.notes and not any(f.name in ("two_peaks", "diameter_vs_eccentricity") for f in res.figures)


def test_skipped_clustering_sets_empty_state(ctx_small, tmp_path):
    from synthdata.report.contract import ReportInputs

    i = ctx_small.inputs
    w = i.wells.iloc[:3].copy()
    w["condition"] = pd.Categorical(["control"] * 3)
    tiny = ReportContext.create(ReportInputs(objects=i.objects.merge(w[["plate", "well"]], on=["plate", "well"]),
                                             wells=w, features=i.features, reference="control", n_sites=i.n_sites),
                                tmp_path)
    from synthdata.report.sections import clustering
    clustering.build(tiny)
    assert tiny.embedding == {} and tiny.clustering == {}


def test_bandwidth_widens_only_for_unimodal_conditions():
    import numpy as np
    rng = np.random.default_rng(0)
    broad = rng.normal(20, 2, 4000)
    mixture = np.r_[rng.normal(8, 1, 2000), rng.normal(20, 2, 2000)]
    base = 0.2
    assert distributions._condition_bandwidth(broad, distributions._scott(broad), base) > 1.5 * base
    assert distributions._condition_bandwidth(mixture, distributions._scott(mixture), base) == base
    # a broad unimodal condition is widened (capped at 2x); a multimodal one is exactly the base
    bw = distributions._condition_bandwidth(broad, distributions._scott(broad), base)
    assert base < bw <= 2 * base
