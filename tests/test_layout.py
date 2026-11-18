from collections import Counter

from synthdata.generators.particle_plate.design import bundled_designs, load_design
from synthdata.generators.particle_plate.layout import assign
from synthdata.generators.particle_plate.wells import parse_well, plate_name, well_name


def test_well_names():
    assert well_name(0, 0) == "A01" and well_name(15, 23) == "P24"
    assert parse_well("h12") == (7, 11)
    assert plate_name(0) == "plate01"


def test_default_by_row():
    d = load_design(bundled_designs()["default"])
    slots = assign(d)
    assert len(slots) == 96
    row_a = [s for s in slots if s.row == 0]
    assert {s.condition for s in row_a} == {"control"}
    assert [s.replicate for s in row_a] == list(range(1, 13))
    assert slots[0].well == "A01" and slots[-1].well == "H12"


def test_by_column(small_dict):
    small_dict["plate"]["format"] = {"rows": 4, "cols": 8}
    small_dict["layout"]["kind"] = "by_column"
    slots = assign(load_design(small_dict))
    assert {s.condition for s in slots if s.col == 2} == {"large"}


def test_explicit_leaves_other_wells_empty(small_dict):
    small_dict["layout"] = {"kind": "explicit", "wells": {"A01": "control", "B03": "large"}}
    slots = assign(load_design(small_dict))
    assert [(s.well, s.condition) for s in slots] == [("A01", "control"), ("B03", "large")]


def test_randomized_is_balanced_and_seed_stable(small_dict):
    small_dict["layout"] = {"kind": "randomized", "seed": 5}
    a = assign(load_design(small_dict))
    b = assign(load_design(small_dict))
    assert a == b
    assert set(Counter(s.condition for s in a).values()) == {4}
    small_dict["layout"]["seed"] = 6
    assert assign(load_design(small_dict)) != a


def test_multiple_plates_repeat_layout(small_dict):
    small_dict["plate"]["plates"] = 2
    slots = assign(load_design(small_dict))
    assert len(slots) == 64
    assert [s.condition for s in slots if s.plate == 0] == [s.condition for s in slots if s.plate == 1]
    assert slots[-1].plate_name == "plate02"


def test_is_outer():
    d = load_design(bundled_designs()["default"])
    by = {s.well: s for s in assign(d)}
    assert by["A05"].is_outer(8, 12) and by["D01"].is_outer(8, 12) and not by["D05"].is_outer(8, 12)
