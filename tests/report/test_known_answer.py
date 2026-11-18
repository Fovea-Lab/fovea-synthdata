"""Spec §5 known answer on the full default plate (slow: ~1 min generate + ~30 s report).

Run with ``uv run pytest -m slow -q``.
"""
import pandas as pd
import pytest

from synthdata.generators.particle_plate import ParticlePlateGenerator
from synthdata.generators.particle_plate.design import bundled_designs, load_design
from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report.build import build_report

pytestmark = pytest.mark.slow

# Measured on the default design (seed 42) on 2026-09-30: ARI 0.827 → pinned at (0.827 − 0.05) rounded down.
ARI_FLOOR = 0.77

EXPECTED_FLAGS = {
    "large": {"diameter_median"},
    "elongated": {"eccentricity_median"},
    "aligned": {"eccentricity_median", "orientation_coherence"},
    "dim_peaked": {"intensity_median"},
    "two_sizes": {"diameter_bimodality"},
    "crowded": {"count_per_site", "nn_distance", "clark_evans"},
}


# Measured extras on the default plate (seed 42): flags beyond the designed feature that are real consequences of
# the design (an IQR moves with its median; a mixture also widens the IQR and changes the count). Anything else is new.
ALLOWED_EXTRA_FLAGS = {
    "large": {"diameter_iqr"},
    "dim_peaked": {"intensity_iqr"},
    "two_sizes": {"diameter_iqr", "count_per_site"},
}


@pytest.fixture(scope="module")
def default_report(tmp_path_factory):
    ds = tmp_path_factory.mktemp("known") / "default"
    ParticlePlateGenerator().generate(load_design(bundled_designs()["default"]), ds, writer="tiff", workers=8)
    out = ds / "report"
    res = build_report(truth_inputs(ds), out)
    return res, out


def test_twin_flagged_on_nothing(default_report):
    res, _ = default_report
    assert res.metrics["twin_flagged"] is False and res.metrics["twin_testable"] is True
    assert res.metrics["flagged"]["control_twin"] == []


@pytest.mark.parametrize("condition", sorted(EXPECTED_FLAGS))
def test_condition_flagged_on_its_feature(default_report, condition):
    res, _ = default_report
    flagged = set(res.metrics["flagged"][condition])
    missing = EXPECTED_FLAGS[condition] - flagged
    assert not missing, f"{condition} not flagged on {sorted(missing)} (flagged: {sorted(flagged)})"


@pytest.mark.parametrize("condition", sorted(EXPECTED_FLAGS))
def test_no_flags_outside_expected_and_allowed_extras(default_report, condition):
    res, _ = default_report
    allowed = EXPECTED_FLAGS[condition] | ALLOWED_EXTRA_FLAGS.get(condition, set())
    extra = set(res.metrics["flagged"][condition]) - allowed
    assert not extra, f"{condition} flagged on unexpected {sorted(extra)}"


def test_control_and_twin_share_a_cluster(default_report):
    _, out = default_report
    clusters = pd.read_parquet(out / "tables" / "clustering" / "clusters.parquet")
    majority = clusters.groupby("condition")["cluster"].agg(lambda s: s.mode().iloc[0])
    assert majority["control"] == majority["control_twin"]


def test_ari_at_or_above_pinned_floor(default_report):
    res, _ = default_report
    assert res.metrics["ari"] >= ARI_FLOOR


def test_map_high_for_distinct_low_for_control_and_twin(default_report):
    res, _ = default_report
    mp = res.metrics["map"]
    for cond, v in mp.items():
        if cond in ("control", "control_twin"):
            assert v < 0.5, f"{cond} mAP {v:.2f} should be noise-level"
        else:
            assert v >= 0.9, f"{cond} mAP {v:.2f} < 0.9"


@pytest.fixture(scope="module")
def default_qc_report(tmp_path_factory):
    ds = tmp_path_factory.mktemp("knownqc") / "default_qc"
    ParticlePlateGenerator().generate(load_design(bundled_designs()["default_qc"]), ds, writer="tiff", workers=8)
    res = build_report(truth_inputs(ds), ds / "report")
    return ds, res


def test_qc_heatmap_equals_truth_labels_and_crowded_passes(default_qc_report):
    ds, res = default_qc_report
    assert res.sections[-1].key == "qc"
    truth = pd.read_parquet(ds / "truth" / "qc.parquet").groupby(["plate", "well"])["well_label"].first()
    wells = pd.read_parquet(ds / "report" / "tables/qc/qc_wells.parquet")
    got = wells.set_index(["plate", "well"])["label"]
    assert len(got) == len(truth) and (got == truth.reindex(got.index)).all()
    crowded = wells[wells["condition"] == "crowded"]
    assert len(crowded) > 0
    clean = crowded[crowded["degree"] == 0]
    assert len(clean) >= 1 and (clean["label"] == "pass").all()
    assert (wells.loc[wells["degree"] == 0, "label"] == "pass").all()
    assert {"warn", "fail"} <= set(truth)
