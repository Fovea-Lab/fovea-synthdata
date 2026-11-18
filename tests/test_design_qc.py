"""qc_defects design block: schema, validation, round-trip, and no effect on designs without it."""

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from synthdata.generators.particle_plate.design import (
    QCDefects, bundled_designs, dump_design, load_design,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _raw(name="small_qc"):
    return yaml.safe_load(bundled_designs()[name].read_text())


def _with(**agg):
    raw = _raw()
    raw["qc_defects"]["aggregation"].update(agg)
    return raw


def test_bundled_qc_designs_load():
    for name in ("default_qc", "small_qc"):
        d = load_design(bundled_designs()[name])
        agg = d.qc_defects.aggregation
        assert agg.level == "well" and agg.assign.kind == "random" and agg.assign.fraction == 0.15
        assert (d.qc_defects.thresholds.warn, d.qc_defects.thresholds.fail) == (0.2, 0.5)


def test_qc_designs_share_seed_and_body_with_base():
    for base in ("default", "small"):
        b, q = _raw(base), _raw(base + "_qc")
        q.pop("qc_defects")
        assert b == q


@pytest.mark.parametrize("name", ["default", "small"])
def test_dump_without_block_unchanged(name):
    d = load_design(bundled_designs()[name])
    assert d.qc_defects is None
    text = dump_design(d)
    assert "qc_defects" not in text
    assert "qc_defects" not in dump_design(d.expanded())
    assert text == (FIXTURES / f"{name}_dump.yaml").read_text()


def test_round_trip():
    d = load_design(bundled_designs()["small_qc"])
    for dd in (d, d.expanded()):
        again = load_design(dump_design(dd))
        assert again.qc_defects == d.qc_defects
        assert dump_design(again) == dump_design(dd)
    assert isinstance(d.expanded().qc_defects, QCDefects)


def test_defaults_when_block_minimal():
    raw = _raw()
    raw["qc_defects"] = {"aggregation": {}}
    q = load_design(raw).qc_defects
    assert q.thresholds.warn == 0.2 and q.thresholds.fail == 0.5
    assert q.aggregation.assign.kind == "random"


@pytest.mark.parametrize("warn,fail", [(0.5, 0.5), (0.6, 0.5), (-0.1, 0.5), (0.2, 1.1)])
def test_bad_thresholds(warn, fail):
    raw = _raw()
    raw["qc_defects"]["thresholds"] = {"warn": warn, "fail": fail}
    with pytest.raises(ValidationError, match="warn"):
        load_design(raw)


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_bad_fraction(fraction):
    raw = _raw()
    raw["qc_defects"]["aggregation"]["assign"]["fraction"] = fraction
    with pytest.raises(ValidationError):
        load_design(raw)


def test_degree_outside_unit_interval():
    with pytest.raises(ValidationError, match="degree"):
        load_design(_with(degree={"dist": "fixed", "value": 3.0}))
    with pytest.raises(ValidationError, match="degree"):
        load_design(_with(degree={"dist": "fixed", "value": -2.0}))


def test_by_condition_unknown_key_and_range():
    with pytest.raises(ValidationError, match="unknown condition"):
        load_design(_with(assign={"kind": "by_condition", "conditions": {"nope": 0.5}}))
    known = next(iter(_raw()["conditions"]))
    with pytest.raises(ValidationError, match="in \\[0, 1\\]"):
        load_design(_with(assign={"kind": "by_condition", "conditions": {known: 1.5}}))
    assert load_design(_with(assign={"kind": "by_condition", "conditions": {known: 0.5}}))


def test_explicit_units_well_level():
    ok = load_design(_with(assign={"kind": "explicit", "units": {"B03": 0.7}}))
    assert ok.qc_defects.aggregation.assign.units == {"B03": 0.7}
    for bad in ({"B03:s1": 0.5}, {"Z99": 0.5}, {"nonsense": 0.5}, {"B03": 1.5}):
        with pytest.raises(ValidationError):
            load_design(_with(assign={"kind": "explicit", "units": bad}))


def test_explicit_units_image_level():
    raw = _raw()
    sites = raw["plate"]["sites_per_well"]
    ok = load_design(_with(level="image", assign={"kind": "explicit", "units": {f"B03:s{sites}": 0.7}}))
    assert ok.qc_defects.aggregation.level == "image"
    for bad in ({"B03": 0.5}, {f"B03:s{sites + 1}": 0.5}, {"B03:s0": 0.5}, {"B03:x1": 0.5}):
        with pytest.raises(ValidationError):
            load_design(_with(level="image", assign={"kind": "explicit", "units": bad}))


def test_unknown_field_forbidden():
    raw = copy.deepcopy(_raw())
    raw["qc_defects"]["bogus"] = 1
    with pytest.raises(ValidationError):
        load_design(raw)


def _units(units, **top):
    raw = _raw()
    raw.update(top)
    raw["qc_defects"]["aggregation"]["assign"] = {"kind": "explicit", "units": units}
    return raw


def test_unit_in_empty_row_of_by_row_layout_rejected():
    top = _raw()
    top["conditions"] = dict(list(top["conditions"].items())[:2])
    top["expectations"] = None
    top["layout"] = {"kind": "by_row"}
    load_design(_units({"A01": 0.5}, **top))
    with pytest.raises(ValidationError, match="H04.*empty"):
        load_design(_units({"H04": 0.5}, **top))


def test_unit_in_empty_well_of_randomized_layout_rejected():
    raw = _raw()
    raw["layout"] = {"kind": "randomized"}
    raw["plate"]["format"] = {"rows": 3, "cols": 3}
    raw["conditions"] = dict(list(raw["conditions"].items())[:2])
    raw.pop("expectations", None)
    # 9 wells / 2 conditions -> 8 populated, 1 empty
    empties = []
    from synthdata.generators.particle_plate.layout import assign
    d = load_design({**raw, "qc_defects": None})
    pop = {(s.row, s.col) for s in assign(d)}
    from synthdata.generators.particle_plate.wells import well_name
    empties = [well_name(r, c) for r in range(3) for c in range(3) if (r, c) not in pop]
    assert len(empties) == 1
    raw["qc_defects"] = _raw()["qc_defects"]
    raw["qc_defects"]["aggregation"]["assign"] = {"kind": "explicit", "units": {empties[0]: 0.5}}
    with pytest.raises(ValidationError, match="empty"):
        load_design(raw)
    raw["qc_defects"]["aggregation"]["assign"]["units"] = {well_name(*sorted(pop)[0]): 0.5}
    load_design(raw)


def test_unit_outside_partial_explicit_layout_rejected():
    raw = _raw()
    first = next(iter(raw["conditions"]))
    raw["layout"] = {"kind": "explicit", "wells": {"B03": first}}
    raw.pop("expectations", None)
    raw["qc_defects"]["aggregation"]["assign"] = {"kind": "explicit", "units": {"B03": 0.5}}
    load_design(raw)
    raw["qc_defects"]["aggregation"]["assign"]["units"] = {"B04": 0.5}
    with pytest.raises(ValidationError, match="B04"):
        load_design(raw)


@pytest.mark.parametrize("key", ["a01", "A1", " A01 ", "A01:s1"])
def test_noncanonical_well_keys_rejected(key):
    with pytest.raises(ValidationError, match="canonical"):
        load_design(_units({key: 0.5}))


@pytest.mark.parametrize("key", ["a01:s1", "A1:s1", "A01:s01", "A01:S1", "A01", "A01:s1 "])
def test_noncanonical_image_keys_rejected(key):
    with pytest.raises(ValidationError, match="canonical"):
        load_design(_image_units({key: 0.5}))


def _image_units(units):
    raw = _units(units)
    raw["qc_defects"]["aggregation"]["level"] = "image"
    return raw


def test_canonical_image_key_accepted():
    assert load_design(_image_units({"A01:s2": 0.5}))
