import json

import numpy as np
import tifffile
import pytest

from synthdata.generators.particle_plate import ParticlePlateGenerator
from synthdata.generators.particle_plate.check import (
    angle_diff_deg, check_dataset, measure_objects, regionprops_orientation_deg,
)
from synthdata.generators.particle_plate.design import load_design

GEN = ParticlePlateGenerator()


@pytest.fixture(scope="module")
def small_ds(tmp_path_factory):
    import yaml
    from synthdata.generators.particle_plate.design import bundled_designs

    out = tmp_path_factory.mktemp("c") / "ds"
    GEN.generate(load_design(yaml.safe_load(bundled_designs()["small"].read_text())), out)
    return out


def test_helpers():
    assert regionprops_orientation_deg(np.pi / 2) == pytest.approx(0.0)
    assert regionprops_orientation_deg(np.pi / 4) == pytest.approx(45.0)
    assert angle_diff_deg(179.0, 1.0) == pytest.approx(2.0)


def test_small_default_design_passes(small_ds):
    rep = check_dataset(small_ds)
    failed = [c.name for c in rep.stage1 + rep.stage2 if not c.passed]
    assert rep.passed, failed
    assert rep.asserted
    body = json.loads((small_ds / "check" / "report.json").read_text())
    assert body["passed"] is True
    md = (small_ds / "check" / "report.md").read_text()
    assert "Stage 1" in md and "Stage 2" in md and "control_twin" in md
    assert set(rep.condition_medians) == {"control", "control_twin", "large", "elongated", "aligned",
                                          "dim_peaked", "two_sizes", "crowded"}


def test_check_is_not_vacuous(small_dict, tmp_path):
    small_dict["conditions"]["control_twin"]["diameter"] = {"dist": "normal", "mean": 20, "sd": 2}
    GEN.generate(load_design(small_dict), tmp_path / "ds", writer="tiff")
    rep = check_dataset(tmp_path / "ds")
    assert not rep.passed
    assert any(not c.passed and "control_twin" in c.name and "diameter" in c.name for c in rep.stage2)


def test_without_expectations_reports_only(small_dict, tmp_path):
    small_dict.pop("expectations")
    GEN.generate(load_design(small_dict), tmp_path / "ds", writer="tiff")
    rep = check_dataset(tmp_path / "ds")
    assert rep.asserted is False and rep.stage2 == [] and rep.passed


def test_check_survives_empty_condition(small_dict, tmp_path):
    small_dict["conditions"]["large"]["count"] = {"dist": "fixed", "value": 0}
    small_dict.pop("expectations")
    GEN.generate(load_design(small_dict), tmp_path / "ds", writer="tiff")
    rep = check_dataset(tmp_path / "ds")
    assert "large" in rep.condition_medians

    def _no_nan(c):
        raise AssertionError(f"non-finite constant {c} in report.json")

    json.loads((tmp_path / "ds" / "check" / "report.json").read_text(), parse_constant=_no_nan)


def test_measure_objects_on_blank_image():
    m = measure_objects(np.full((64, 64), 200, dtype=np.uint16))
    assert m["count"] == 0 and m["nn_distance"].size == 0


def test_stage1_fails_when_masks_are_blank(small_dict, tmp_path):
    small_dict.pop("expectations")
    GEN.generate(load_design(small_dict), tmp_path / "ds", writer="tiff")
    for f in (tmp_path / "ds" / "masks").glob("*.tif"):
        tifffile.imwrite(f, np.zeros_like(tifffile.imread(f)))
    rep = check_dataset(tmp_path / "ds")
    assert not rep.passed
    assert any(not c.passed and c.name == "stage1: no eligible truth particle found in the masks" for c in rep.stage1)


def test_stage1_reports_every_particle_found(small_ds):
    rep = check_dataset(small_ds, write=False)
    c = next(c for c in rep.stage1 if c.name == "stage1: every eligible particle found in its mask")
    assert c.passed and c.detail["matched"] == c.detail["eligible"] > 0


def test_insufficient_wells_fail():
    from synthdata.generators.particle_plate.check import _compare

    nan = float("nan")
    wells = {"a": [{"eccentricity": 0.5}, {"eccentricity": 0.6}, {"eccentricity": 0.55}],
             "b": [{"eccentricity": nan}, {"eccentricity": 0.9}, {"eccentricity": nan}]}
    for kind in ("same", "differs"):
        c = _compare(kind, "a", "b", "eccentricity", wells, {})
        assert not c.passed and c.detail == {"insufficient": True, "n_a": 3, "n_b": 1}


def test_blank_wells_still_fail_report(small_dict, tmp_path):
    # count 0 does not give an empty well to the naive pipeline (Otsu splits the noise floor), so the
    # expectation fails on the values instead; the report must fail either way.
    small_dict["conditions"]["aligned"]["count"] = {"dist": "fixed", "value": 0}
    GEN.generate(load_design(small_dict), tmp_path / "ds", writer="tiff")
    rep = check_dataset(tmp_path / "ds")
    c = next(c for c in rep.stage2 if c.name == "same: elongated vs aligned \u00b7 eccentricity")
    assert not c.passed and not rep.passed


def test_expectation_on_condition_missing_from_plate_fails(small_dict, tmp_path):
    small_dict["layout"]["order"] = [c for c in small_dict["layout"]["order"] if c != "crowded"]
    GEN.generate(load_design(small_dict), tmp_path / "ds", writer="tiff")
    rep = check_dataset(tmp_path / "ds")
    miss = [c for c in rep.stage2 if c.detail.get("missing") == "crowded"]
    assert miss and all(not c.passed and c.detail["reason"] == "not on the plate" for c in miss)
    assert not rep.passed
