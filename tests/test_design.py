import pytest
from pydantic import ValidationError

from synthdata.generators.particle_plate.design import (
    bundled_designs, dump_design, load_design,
)


def test_bundled_designs_load():
    for name, path in bundled_designs().items():
        d = load_design(path)
        assert d.generator == "particle_plate", name
    d = load_design(bundled_designs()["default"])
    assert d.plate.grid() == (8, 12)
    assert list(d.conditions) == ["control", "control_twin", "large", "elongated", "aligned",
                                  "dim_peaked", "two_sizes", "crowded"]


def test_condition_inherits_base_and_overrides(small_design):
    large = small_design.particle_for("large")
    assert large.diameter.mean == 20
    assert large.eccentricity == small_design.base_particle.eccentricity
    assert small_design.particle_for("control") == small_design.base_particle
    assert small_design.conditions["control"].role == "control"
    assert small_design.conditions["large"].role == "test"


def test_image_effect_override(small_dict):
    small_dict["conditions"]["large"]["image_effects"] = {"illumination": {"kind": "none"}}
    d = load_design(small_dict)
    assert d.effects_for("large").illumination.kind == "none"
    assert d.effects_for("large").background == d.image_effects.background
    assert d.effects_for("control").illumination.kind == "radial"


def test_dump_load_round_trip(small_design):
    assert load_design(dump_design(small_design)) == small_design


def test_expanded_is_equivalent_and_explicit(small_design):
    ex = small_design.expanded()
    for c in small_design.conditions:
        assert ex.particle_for(c) == small_design.particle_for(c)
        assert ex.effects_for(c) == small_design.effects_for(c)
        assert ex.conditions[c].diameter is not None
    assert load_design(dump_design(ex)).expanded() == ex


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d["conditions"]["large"].update(eccentricity={"dist": "fixed", "value": 1.2}),
     "conditions.large.eccentricity"),
    (lambda d: d["base_particle"].update(diameter={"dist": "normal", "mean": -1, "sd": 1}),
     "diameter"),
    (lambda d: d["base_particle"].update(count={"dist": "fixed", "value": -3}), "count"),
    (lambda d: d["layout"].update(order=["control", "ghost"]), "ghost"),
    (lambda d: d["plate"].update(channels=[{"name": "a"}, {"name": "b"}]), "one channel"),
    (lambda d: d["base_particle"].update(diamter={"dist": "fixed", "value": 3}), "diamter"),
    (lambda d: d["plate"].update(format={"rows": 4, "cols": 4}), "8 conditions > 4 rows"),
    (lambda d: d["layout"].update(kind="explicit", wells={"Z99": "control"}), "Z99"),
    (lambda d: d["expectations"].update(reference="ghost"), "ghost"),
    (lambda d: d["layout"].update(kind="explicit", wells={}), "places no wells"),
    (lambda d: d["layout"].update(kind="explicit", wells={"a01": "control", "A01": "large"}), "a01.*A01"),
])
def test_bad_designs_fail_with_named_field(small_dict, mutate, fragment):
    mutate(small_dict)
    with pytest.raises(ValidationError, match=fragment):
        load_design(small_dict)
