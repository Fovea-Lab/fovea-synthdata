import numpy as np
import pandas as pd
import pytest

from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report.contract import DEFAULT_FEATURES, validate
from synthdata.report.embed import (cluster_scores, dendrogram_segments, object_embedding, object_sample,
                                    pca, similarity, umap_embed, ward)
from synthdata.report.profiles import PROFILE_COLUMNS, robust_z, well_profiles


def _blobs(n=5, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0, 0], [20, 0, 0, 0], [0, 20, 20, 0]], float)
    x = np.vstack([c + rng.normal(0, 0.3, (n, 4)) for c in centres])
    labels = np.repeat(["a", "b", "c"], n)
    cols = list("wxyz")
    return pd.DataFrame(x, columns=cols, index=[f"w{i}" for i in range(3 * n)]), labels, cols


def test_pca_variance_and_stable_signs():
    z, _, cols = _blobs()
    z.iloc[0, 1] = np.nan
    s1, ev1, l1 = pca(z, cols)
    s2, ev2, l2 = pca(z, cols)
    assert list(s1.columns) == ["PC1", "PC2", "PC3"] and 0 < sum(ev1) <= 1 + 1e-9
    pd.testing.assert_frame_equal(s1, s2)
    for pc in l1.columns:
        col = l1[pc]
        assert col.loc[col.abs().idxmax()] > 0


def test_umap_deterministic_and_small_n():
    z, _, cols = _blobs()
    a, b = umap_embed(z[cols].to_numpy()), umap_embed(z[cols].to_numpy())
    assert a.shape == (15, 2) and np.array_equal(a, b)
    assert umap_embed(z[cols].to_numpy()[:3]) is None


def test_ward_recovers_blobs_and_segments():
    z, labels, cols = _blobs()
    Z, order = ward(z, cols)
    assert sorted(order) == list(range(15))
    assert cluster_scores(Z, labels, 3)["ari"] == pytest.approx(1.0)
    seg = dendrogram_segments(Z, list(z.index))
    assert list(seg.columns) == ["x", "y", "segment", "order"]
    assert len(seg) == 4 * (15 - 1) and seg["x"].min() == 0 and seg["x"].max() == 14
    sim = similarity(z, cols, order)
    assert list(sim.columns) == ["a", "b", "cosine"] and len(sim) == 15 * 15
    assert list(pd.unique(sim["a"])) == [z.index[i] for i in order]


def test_skip_rule():
    z, labels, cols = _blobs()
    assert pca(z.iloc[:2], cols) is None and ward(z.iloc[:2], cols) is None
    Z, _ = ward(z, cols)
    assert cluster_scores(Z, np.array(["a"] * 15), 1) is None


def test_object_sample_stratified_deterministic():
    rng = np.random.default_rng(1)
    wells = pd.DataFrame({"plate": "P", "well": [f"A{i}" for i in range(1, 4)], "condition": ["c1", "c2", "c3"]})
    objs = pd.concat([pd.DataFrame({"plate": "P", "well": w, "site": 1, "object_id": range(n),
                                    "diameter": rng.normal(size=n), "eccentricity": rng.random(n),
                                    "orientation": rng.uniform(0, 180, n), "intensity": rng.random(n)})
                      for w, n in [("A1", 500), ("A2", 300), ("A3", 40)]], ignore_index=True)
    a = object_sample(objs, wells, DEFAULT_FEATURES, n=300, seed=0)
    b = object_sample(objs, wells, DEFAULT_FEATURES, n=300, seed=0)
    pd.testing.assert_frame_equal(a, b)
    counts = a["condition"].value_counts()
    assert len(a) == 300 and counts["c3"] == 40 and abs(counts["c1"] - counts["c2"]) <= 1
    emb = object_embedding(a, DEFAULT_FEATURES)
    assert list(emb.columns) == ["u1", "u2"] and len(emb) == 300
    one = wells.assign(condition="c1")
    assert object_sample(objs, one, DEFAULT_FEATURES) is None


@pytest.mark.parametrize("fixture", ["small_truth_dataset", "replicate_truth_dataset"])
def test_truth_clustering(fixture, request, capsys):
    inputs = validate(truth_inputs(request.getfixturevalue(fixture)))
    prof = well_profiles(inputs)
    cols = PROFILE_COLUMNS(inputs.features)
    z, _ = robust_z(prof, inputs.reference, cols)
    Z, _ = ward(z, cols)
    conds = prof["condition"].astype(str).to_numpy()
    k = len(set(conds))
    res = cluster_scores(Z, conds, k)
    with capsys.disabled():
        print(f"\n[{fixture}] ARI={res['ari']:.4f} NMI={res['nmi']:.4f}")
    lab = pd.Series(res["labels"], index=prof.index)
    major = {c: lab[conds == c].mode().iloc[0] for c in ("control", "control_twin")}
    assert major["control"] == major["control_twin"]
    assert res["ari"] >= 0.6
