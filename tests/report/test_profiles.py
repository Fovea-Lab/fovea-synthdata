from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from synthdata.report.contract import DEFAULT_FEATURES, ReportInputs, validate
from synthdata.report.profiles import (PROFILE_COLUMNS, axial_stats, clark_evans, n_sites, robust_z,
                                       sarle_bimodality, well_profiles)


def test_coherence_aligned():
    assert axial_stats(np.full(20, 30.0))[0] == pytest.approx(1.0)


def test_coherence_uniform():
    assert axial_stats(np.linspace(0, 180, 1000, endpoint=False))[0] < 0.01


def test_coherence_wraparound():
    r, axis = axial_stats(np.array([10.0, 170.0]))
    assert r == pytest.approx(np.cos(np.deg2rad(20)), abs=1e-3)
    assert 0 <= axis < 180
    assert axis == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("angles", [[175.0, 5.0], [160.0, 20.0, 170.0, 10.0], [10.0, 170.0]])
def test_mean_axis_strictly_below_180(angles):
    _, axis = axial_stats(np.array(angles))
    assert 0 <= axis < 180
    assert axis == pytest.approx(0.0, abs=1e-6)


def test_declared_site_count_divides_count_per_site():
    inp = _inputs(with_empty=False)
    inp.objects["site"] = 1  # data shows a single site
    inp = replace(inp, n_sites=4)
    assert n_sites(inp) == 4
    assert well_profiles(inp).loc[("P", "A01"), "count_per_site"] == 30 / 4
    assert n_sites(replace(inp, n_sites=None)) == 1


def _nn(pts):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=2)
    return [d[:, 1]]


def test_clark_evans_regular_and_clumped():
    g = np.array([(x, y) for x in np.arange(5, 100, 10) for y in np.arange(5, 100, 10)], float)
    assert clark_evans(_nn(g), 100 * 100) > 1.9
    clump = np.random.default_rng(0).normal(50, 2, size=(100, 2))
    assert clark_evans(_nn(clump), 100 * 100) < 0.3
    assert np.isnan(clark_evans(_nn(g), None))


def test_bimodality():
    rng = np.random.default_rng(0)
    bi = np.r_[rng.normal(8, 1, 500), rng.normal(20, 2, 500)]
    assert sarle_bimodality(bi) > 0.555
    assert sarle_bimodality(rng.normal(12, 1, 1000)) < 0.555
    assert np.isnan(sarle_bimodality(np.array([1.0, 2.0, 3.0])))


def _inputs(with_empty=True):
    rng = np.random.default_rng(1)
    rows = []
    for w, n in (("A01", 30), ("A02", 30), ("B01", 30), ("B02", 0 if with_empty else 30)):
        for i in range(n):
            rows.append(dict(plate="P", well=w, site=1 + i % 2, object_id=f"{w}-{i}",
                             x=rng.uniform(0, 100), y=rng.uniform(0, 100),
                             diameter=rng.normal(10, 1), eccentricity=rng.uniform(0, 1),
                             orientation=rng.uniform(0, 180), intensity=rng.normal(5, 1)))
    wells = pd.DataFrame(dict(plate="P", well=["A01", "A02", "B01", "B02"], row=[0, 0, 1, 1],
                              col=[0, 1, 0, 1], condition=["control", "control", "t", "t"],
                              role="x", replicate=[1, 2, 1, 2]))
    return validate(ReportInputs(pd.DataFrame(rows), wells, site_area_px=10000.0))


def test_profiles_handle_empty_well():
    inp = _inputs()
    p = well_profiles(inp)
    assert n_sites(inp) == 2
    e = p.loc[("P", "B02")]
    assert e["count_per_site"] == 0 and e["n_objects"] == 0
    for c in PROFILE_COLUMNS(DEFAULT_FEATURES):
        if c != "count_per_site":
            assert np.isnan(e[c]), c
    assert p.loc[("P", "A01"), "count_per_site"] == 15
    assert set(PROFILE_COLUMNS(DEFAULT_FEATURES)) <= set(p.columns)
    assert "orientation_mean_axis" in p.columns


def test_profile_columns_order():
    assert PROFILE_COLUMNS(DEFAULT_FEATURES) == [
        "count_per_site", "diameter_median", "diameter_iqr", "eccentricity_median", "eccentricity_iqr",
        "intensity_median", "intensity_iqr", "orientation_coherence", "nn_distance", "clark_evans",
        "diameter_bimodality"]


def test_robust_z_reference_median_zero():
    p = well_profiles(_inputs(with_empty=False))
    z, notes = robust_z(p, "control", ["diameter_median"])
    ref = p["condition"].astype(str) == "control"
    assert z.loc[ref, "diameter_median"].median() == pytest.approx(0.0, abs=1e-9)


def test_robust_z_zero_spread():
    p = pd.DataFrame({"condition": ["control"] * 3 + ["t"], "v": [2.0, 2.0, 2.0, 5.0]})
    z, notes = robust_z(p, "control", ["v"])
    assert z["v"].iloc[3] == pytest.approx(3.0)
    assert any("v" in n for n in notes)
