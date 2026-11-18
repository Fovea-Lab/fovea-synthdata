import copy
import re

import pytest
import yaml
from pydantic import ValidationError

from synthdata.core import registry
from synthdata.generators.cell_plate.design import Phenotype, bundled_designs, dump_design, load_design
from synthdata.generators.cell_plate.layout import assign


def test_bundled_designs_load():
    assert {"default", "small"} <= set(bundled_designs())
    d = load_design(bundled_designs()["default"])
    assert [c.name for c in d.plate.channels] == ["DNA", "ER", "RNA", "AGP", "Mito"]
    assert d.plate.image.height == 1024 and d.plate.sites_per_well == 2
    assert len([c for c in d.compounds.values() if c.role == "test"]) == 9
    s = load_design(bundled_designs()["small"])
    assert [c.name for c in s.plate.channels] == ["DNA", "AGP"]
    assert s.plate.image.height == 384 and s.plate.sites_per_well == 1


def test_dose_scaling():
    p = Phenotype(count_factor=0.35, multinucleate_add=0.6)
    low = p.at(0.4)
    assert low.count_factor == pytest.approx(0.74)
    assert low.multinucleate_add == pytest.approx(0.24)
    assert p.at(0.0).count_factor == 1.0
    assert all(type(getattr(low, f)) is float for f in Phenotype.model_fields)


@pytest.mark.parametrize("edit,field", [
    (lambda d: d["compounds"]["CPD-001"].update(phenotype="nope"), "phenotype"),
    (lambda d: d["layout"]["fixed"].update(A01="NOPE"), "layout"),
    (lambda d: d["plate"].update(plates=0), "plates"),
    (lambda d: d["plate"]["channels"].append(d["plate"]["channels"][0]), "channels"),
    (lambda d: d["expectations"].update(reference="NOPE"), "reference"),
])
def test_validation_names_the_field(edit, field):
    data = yaml.safe_load(bundled_designs()["default"].read_text())
    edit(data)
    with pytest.raises(ValidationError, match=field):
        load_design(copy.deepcopy(data))


@pytest.mark.parametrize("name", ["default", "small"])
def test_design_round_trips(name):
    d = load_design(bundled_designs()[name])
    assert load_design(dump_design(d)) == d


def test_registry_finds_both():
    assert {"cell_plate", "particle_plate"} <= set(registry.names())
    assert registry.get("cell_plate").name == "cell_plate"


def test_particle_bundled_designs_unchanged():
    from synthdata.generators.particle_plate.design import bundled_designs as pb
    assert "default_qc" in pb() and "default" in pb() and "cell_plate" not in str(pb()["default"])


def test_combined_multiplies_factors_and_adds_adds():
    p = Phenotype(count_factor=0.5, dna_foci_add=1.0).combined(Phenotype(count_factor=0.5, dna_foci_add=2.0))
    assert p.count_factor == 0.25 and p.dna_foci_add == 3.0


def test_absent_fields_stay_out_of_the_dumped_design():
    # Parsed keys, not a word search over the text: a phenotype or compound name must not decide this.
    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    text = dump_design(load_design(bundled_designs()["small"]))
    assert not {"potency", "side_effect", "mechanism", "plate_effects"} & set(keys(yaml.safe_load(text)))
    assert not {"potency", "side_effect", "mechanism", "plate_effects"} & set(re.findall(r"\w+", text))


def design_with(potency=None, side=None, mechanism=None, compound="CPD-001"):
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["phenotypes"]["x"] = {"count_factor": 0.5}
    c = data["compounds"][compound]
    if potency is not None:
        c["potency"] = potency
    if side is not None:
        c["side_effect"] = {"phenotype": side[0], "strength": side[1]}
    if mechanism is not None:
        c["mechanism"] = mechanism
    return load_design(data)


def slot_of(design, compound="CPD-001", dose_strength=1.0):
    return next(s for s in assign(design) if s.compound == compound and s.role == "test"
                and s.strength == pytest.approx(dose_strength * (design.compounds[compound].potency or 1.0)))


def test_potency_scales_main_strength_only():
    s = slot_of(design_with(potency=0.5, side=("x", 0.2)), dose_strength=1.0)
    assert s.strength == 0.5 and s.side_strength == 0.2
    assert s.side_effect == "x"


def test_side_strength_follows_the_dose_not_the_potency():
    d = design_with(potency=0.5, side=("x", 0.2))
    low = next(s for s in assign(d) if s.compound == "CPD-001" and s.dose_label == "1 uM")
    assert low.strength == pytest.approx(0.2) and low.side_strength == pytest.approx(0.08)


def test_mechanism_reaches_the_slot_and_controls_have_none():
    slots = assign(design_with(mechanism="tubulin"))
    assert {s.mechanism for s in slots if s.compound == "CPD-001"} == {"tubulin"}
    assert {s.mechanism for s in slots if s.compound != "CPD-001"} == {""}
    assert all(s.side_effect is None and s.side_strength == 0.0 for s in slots)


@pytest.mark.parametrize("edit,field", [
    (lambda c: c["CPD-001"].update(side_effect={"phenotype": "nope", "strength": 0.2}), "nope"),
    (lambda c: c["DMSO"].update(side_effect={"phenotype": "cytotoxic", "strength": 0.2}), "side_effect"),
    (lambda c: c["POS-CTRL"].update(mechanism="tubulin"), "mechanism"),
    (lambda c: c["DMSO"].update(mechanism="tubulin"), "mechanism"),
    (lambda c: c["POS-CTRL"].update(potency=0.5), "potency"),
    (lambda c: c["CPD-001"].update(potency=0), "potency"),
    (lambda c: c["CPD-001"].update(side_effect={"phenotype": "cytotoxic", "strength": 0}), "strength"),
    (lambda c: c["CPD-001"].update(side_effect={"phenotype": "cytotoxic", "strength": 1.5}), "strength"),
])
def test_unknown_side_effect_phenotype_is_refused(edit, field):
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    edit(data["compounds"])
    with pytest.raises(ValidationError, match=field):
        load_design(copy.deepcopy(data))


def _with_effects(effects):
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate_effects"] = effects
    return data


def test_wrong_number_of_plate_effects_is_refused():
    one_plate = yaml.safe_load(bundled_designs()["small"].read_text())["plate"]["plates"]
    assert one_plate == 1
    with pytest.raises(ValidationError, match="2 entries for 1 plate"):
        load_design(_with_effects([{"gain": {}, "background_shift": {}}] * 2))


def test_unknown_channel_in_plate_effect_is_refused():
    with pytest.raises(ValidationError, match=r"plate_effects\[0\].gain: unknown channel"):
        load_design(_with_effects([{"gain": {"NOPE": 2.0}, "background_shift": {}}]))
    with pytest.raises(ValidationError, match=r"background_shift: unknown channel"):
        load_design(_with_effects([{"gain": {}, "background_shift": {"NOPE": 2.0}}]))
