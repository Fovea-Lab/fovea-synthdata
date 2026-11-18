"""Aggregation defect rendering, assignment, and truth/qc.parquet."""

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from scipy.spatial import cKDTree

from synthdata.core.provenance import hash_tree
from synthdata.generators.particle_plate.design import bundled_designs, load_design
from synthdata.generators.particle_plate.generator import ParticlePlateGenerator
from synthdata.generators.particle_plate.layout import assign
from synthdata.generators.particle_plate.qc import NS_QC, assign_aggregation, assign_aggregation_detail
from synthdata.generators.particle_plate.render import NS_IMAGE
from synthdata.generators.particle_plate.layout import NS_LAYOUT

FIXTURES = Path(__file__).parent / "fixtures"
GEN = ParticlePlateGenerator()


def _raw(name="small"):
    return copy.deepcopy(yaml.safe_load(bundled_designs()[name].read_text()))


def _qc(agg, thresholds=None, base="small"):
    raw = _raw(base)
    raw["qc_defects"] = {"aggregation": agg, **({"thresholds": thresholds} if thresholds else {})}
    return load_design(raw)


def _gen(design, tmp_path, name="o", **kw):
    out = tmp_path / name
    GEN.generate(design, out, writer="tiff", **kw)
    return out


def _tbl(out, name):
    return pq.read_table(out / "truth" / f"{name}.parquet").to_pandas()


def test_namespace_distinct():
    assert len({NS_QC, NS_IMAGE, NS_LAYOUT}) == 3


def test_byte_identity_small(tmp_path):
    expected = json.loads((FIXTURES / "small_hashes.json").read_text())
    got = hash_tree(_gen(load_design(bundled_designs()["small"]), tmp_path))
    new = {k for k in got if k not in expected}
    assert new == {"truth/qc.parquet"}
    assert {k: v for k, v in got.items() if k in expected} == expected


def test_no_defects_qc_table_all_pass(tmp_path):
    out = _gen(load_design(bundled_designs()["small"]), tmp_path)
    qc = _tbl(out, "qc")
    assert len(qc) == 32 * 2
    assert (qc.degree == 0).all() and (qc.label == "pass").all() and (qc.well_label == "pass").all()
    assert (qc.n_clustered == 0).all()
    assert "qc_clustered" not in _tbl(out, "particles").columns


def test_random_exact_and_seed_stable():
    d = _qc({"level": "well", "assign": {"kind": "random", "fraction": 0.25}})
    slots = assign(d)
    a1, a2 = assign_aggregation(d, slots), assign_aggregation(d, slots)
    assert a1 == a2
    wells = {k[:3] for k in a1}
    assert len(wells) == round(0.25 * 32) and len(a1) == len(wells) * 2
    for w in wells:  # well level: both sites share the degree
        assert a1[(*w, 1)] == a1[(*w, 2)]
    raw = _raw()
    raw["seed"] = 8
    raw["qc_defects"] = d.qc_defects.model_dump(mode="json")
    assert assign_aggregation(load_design(raw), slots) != a1


def test_random_image_level_counts():
    d = _qc({"level": "image", "assign": {"kind": "random", "fraction": 0.5}})
    assert len(assign_aggregation(d, assign(d))) == round(0.5 * 64)


def test_by_condition_extremes():
    d1 = _qc({"level": "well", "assign": {"kind": "by_condition", "conditions": {"control": 1.0, "large": 0.0}}})
    a = assign_aggregation(d1, assign(d1))
    hit = {k[:3] for k in a}
    ctrl = {(s.plate, s.row, s.col) for s in assign(d1) if s.condition == "control"}
    assert hit == ctrl and len(ctrl) == 4
    d0 = _qc({"level": "well", "assign": {"kind": "by_condition", "conditions": {"control": 0.0}}})
    assert assign_aggregation(d0, assign(d0)) == {}


def test_by_condition_48_wells_per_condition():
    raw = _raw()
    raw["plate"]["format"] = {"rows": 8, "cols": 12}
    raw["layout"] = {"kind": "randomized", "order": ["control", "large"]}
    raw["conditions"] = {"control": raw["conditions"]["control"], "large": raw["conditions"]["large"]}
    raw["expectations"] = {"reference": "control", "differs": {"large": ["diameter"]}}
    raw["qc_defects"] = {"aggregation": {"level": "well", "assign": {"kind": "by_condition",
                                                                       "conditions": {"control": 1.0, "large": 0.0}}}}
    d = load_design(raw)
    slots = assign(d)
    a = assign_aggregation(d, slots)
    assert {k[:3] for k in a} == {(s.plate, s.row, s.col) for s in slots if s.condition == "control"}
    assert len({k[:3] for k in a}) == 48


def test_explicit_exact():
    d = _qc({"level": "image", "assign": {"kind": "explicit", "units": {"B03:s2": 0.7, "A01:s1": 0.1}}})
    assert assign_aggregation(d, assign(d)) == {(0, 1, 2, 2): 0.7, (0, 0, 0, 1): 0.1}
    dw = _qc({"level": "well", "assign": {"kind": "explicit", "units": {"B03": 0.6}}})
    assert assign_aggregation(dw, assign(dw)) == {(0, 1, 2, 1): 0.6, (0, 1, 2, 2): 0.6}


def test_clustered_count_and_labels(tmp_path):
    d = _qc({"level": "well", "assign": {"kind": "explicit", "units": {"A01": 0.8, "A02": 0.3, "A03": 0.1}}},
            {"warn": 0.2, "fail": 0.5})
    out = _gen(d, tmp_path)
    qc, imgs, parts = _tbl(out, "qc"), _tbl(out, "images"), _tbl(out, "particles")
    for well, deg, lab in [("A01", 0.8, "fail"), ("A02", 0.3, "warn"), ("A03", 0.1, "pass")]:
        for site in (1, 2):
            r = qc[(qc.well == well) & (qc.site == site)].iloc[0]
            n = int(imgs[(imgs.well == well) & (imgs.site == site)].n_particles.iloc[0])
            p = parts[(parts.well == well) & (parts.site == site)]
            assert r.degree == deg and r.label == lab and r.well_label == lab
            assert r.n_clustered == round(deg * n) == int(p.qc_clustered.sum())
            assert r.clusters >= 1 and r.spread > 0
    other = qc[~qc.well.isin(["A01", "A02", "A03"])]
    assert (other.degree == 0).all() and (other.n_clustered == 0).all()
    assert not parts[~parts.well.isin(["A01", "A02", "A03"])].qc_clustered.any()


def test_well_label_is_worst(tmp_path):
    d = _qc({"level": "image", "assign": {"kind": "explicit", "units": {"A01:s1": 0.05, "A01:s2": 0.9}}})
    qc = _tbl(_gen(d, tmp_path), "qc")
    a = qc[qc.well == "A01"]
    assert sorted(a.label) == ["fail", "pass"] and (a.well_label == "fail").all()


def _clark_evans(xy, area):
    tree = cKDTree(xy)
    dd, _ = tree.query(xy, k=2)
    return dd[:, 1].mean() / (0.5 / np.sqrt(len(xy) / area))


def test_clark_evans_decreases_with_degree(tmp_path):
    degs = [0.0, 0.3, 0.6, 0.9]
    raw = _raw()
    raw["base_particle"]["count"] = {"dist": "poisson", "mean": 60}
    raw["layout"] = {"kind": "by_row", "order": ["control"]}
    raw["conditions"] = {"control": raw["conditions"]["control"]}
    raw["expectations"] = {"reference": "control"}
    raw["plate"]["format"] = {"rows": 4, "cols": 4}
    raw["plate"]["sites_per_well"] = 1
    raw["qc_defects"] = {"aggregation": {"level": "well", "assign": {"kind": "explicit", "units": {
        "A01": 0.0, "A02": 0.3, "A03": 0.6, "A04": 0.9}}}}
    d = load_design(raw)
    out = _gen(d, tmp_path)
    p = _tbl(out, "particles")
    r = []
    for w in ("A01", "A02", "A03", "A04"):
        q = p[p.well == w]
        r.append(_clark_evans(q[["x", "y"]].to_numpy(), 384 * 384))
    print("CE", dict(zip(degs, r)))
    assert r[0] > r[1] > r[2] > r[3]


def test_workers_identical(tmp_path):
    d = _qc({"level": "well", "assign": {"kind": "random", "fraction": 0.3}})
    h1 = hash_tree(_gen(d, tmp_path, "w1", workers=1))
    h2 = hash_tree(_gen(d, tmp_path, "w2", workers=2))
    assert h1 == h2 and "truth/qc.parquet" in h1


def test_preview_unaffected_without_defects():
    from synthdata.generators.particle_plate.render import render_image
    d = load_design(bundled_designs()["small"])
    first = {}
    for sl in assign(d):
        first.setdefault(sl.condition, sl)
    tiles = GEN.preview(d, 2).tiles
    for cond, sl in first.items():
        for site, r in enumerate(tiles[cond], 1):
            assert np.array_equal(r.image, render_image(d, sl, site).image)
            assert r.qc_row["degree"] == 0 and r.qc_row["label"] == "pass"


def test_label_boundaries(tmp_path):
    d = _qc({"level": "well", "assign": {"kind": "explicit", "units": {"A01": 0.2, "A02": 0.5, "A03": 0.19}}},
            {"warn": 0.2, "fail": 0.5})
    qc = _tbl(_gen(d, tmp_path), "qc")
    lab = {w: qc[qc.well == w].label.iloc[0] for w in ("A01", "A02", "A03")}
    assert lab == {"A01": "warn", "A02": "fail", "A03": "pass"}


def test_unaffected_images_identical_to_no_defect_design(tmp_path):
    d = _qc({"level": "well", "assign": {"kind": "explicit", "units": {"B03": 0.8}}})
    base = hash_tree(_gen(load_design(bundled_designs()["small"]), tmp_path, "base"))
    out = hash_tree(_gen(d, tmp_path, "def"))
    imgs = [k for k in base if k.startswith("images/")]
    assert imgs
    diff = {k for k in imgs if base[k] != out[k]}
    assert diff and all("_B03_" in k for k in diff)


def test_degree_clipping_recorded(tmp_path):
    d = _qc({"level": "well", "assign": {"kind": "random", "fraction": 1.0},
             "degree": {"dist": "normal", "mean": 0.9, "sd": 0.3}})
    out = _gen(d, tmp_path)
    qc = _tbl(out, "qc")
    assert qc.degree.between(0, 1).all()
    n = int(qc.degree_clipped.sum())
    assert n > 0
    assert json.loads((out / "manifest.json").read_text())["extra"]["qc_degree_clipped"] == n
    assert set(qc[qc.degree_clipped].degree) <= {0.0, 1.0}
    _, clipped = assign_aggregation_detail(d, assign(d))
    assert len(clipped) == n
    qc0 = _tbl(_gen(load_design(bundled_designs()["small"]), tmp_path, "nd"), "qc")
    assert not qc0.degree_clipped.any()


def test_degrees_stable_when_other_units_change():
    def deg(p):
        d = _qc({"level": "well", "assign": {"kind": "by_condition", "conditions": {"control": 1.0, "large": p}}})
        return assign_aggregation(d, assign(d))
    a, b = deg(0.0), deg(1.0)
    assert set(a) < set(b)
    assert all(b[k] == v for k, v in a.items())


def test_zero_cluster_count_is_nan_and_unconsumed(tmp_path):
    raw = _raw()
    raw["base_particle"]["count"] = {"dist": "fixed", "value": 3}
    raw["qc_defects"] = {"aggregation": {"level": "well", "assign": {"kind": "explicit", "units": {"A01": 0.1}}}}
    out = _gen(load_design(raw), tmp_path)
    r = _tbl(out, "qc").query("well == 'A01'").iloc[0]
    assert r.degree == 0.1 and r.n_clustered == 0 and np.isnan(r.clusters) and np.isnan(r.spread)
    assert not _tbl(out, "particles").qc_clustered.any()
