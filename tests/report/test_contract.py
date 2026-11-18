import numpy as np
import pandas as pd
import pytest

from synthdata.report.contract import DEFAULT_FEATURES, FeatureSpec, ReportError, ReportInputs, validate


def make(**kw):
    wells = pd.DataFrame({
        "plate": "P1", "well": ["A01", "A02", "B01", "B02"], "row": ["A", "A", "B", "B"],
        "col": [1, 2, 1, 2], "condition": ["ctl", "ctl", "drug", "drug"],
        "role": ["control", "control", "test", "test"], "replicate": [1, 2, 1, 2],
    })
    rng = np.random.default_rng(0)
    objs = pd.DataFrame({
        "plate": "P1", "well": np.repeat(wells["well"], 3).to_numpy(), "site": 1,
        "object_id": np.arange(12), "x": rng.random(12), "y": rng.random(12),
        "diameter": rng.random(12), "eccentricity": rng.random(12),
        "orientation": rng.random(12) * 180, "intensity": rng.random(12),
    })
    return ReportInputs(objects=objs, wells=wells, **kw)


def test_valid_minimal():
    out = validate(make(reference="ctl"))
    assert list(out.wells["condition"].cat.categories) == ["ctl", "drug"]


def test_missing_column_named():
    inp = make(reference="ctl")
    inp.objects = inp.objects.drop(columns=["object_id"])
    with pytest.raises(ReportError, match="object_id"):
        validate(inp)


def test_contract_missing_feature():
    inp = make(reference="ctl", features=DEFAULT_FEATURES + (FeatureSpec("solidity", "linear", "Solidity"),))
    with pytest.raises(ReportError, match="solidity"):
        validate(inp)


def test_unknown_well_fails():
    inp = make(reference="ctl")
    inp.objects.loc[0, "well"] = "Z99"
    with pytest.raises(ReportError, match="Z99"):
        validate(inp)


def test_reference_one_well_fails():
    inp = make(reference="ctl")
    inp.wells = inp.wells.iloc[1:].reset_index(drop=True)
    inp.objects = inp.objects[inp.objects["well"].isin(inp.wells["well"])]
    with pytest.raises(ReportError, match="reference"):
        validate(inp)


def test_reference_unknown_fails():
    with pytest.raises(ReportError, match="nope"):
        validate(make(reference="nope"))


def test_empty_objects_fail():
    inp = make(reference="ctl")
    inp.objects = inp.objects.iloc[0:0]
    with pytest.raises(ReportError, match="empty"):
        validate(inp)


def test_condition_order_first_appearance():
    inp = make(reference="zeta")
    inp.wells["condition"] = ["zeta", "zeta", "alpha", "alpha"]
    assert list(validate(inp).wells["condition"].cat.categories) == ["zeta", "alpha"]
