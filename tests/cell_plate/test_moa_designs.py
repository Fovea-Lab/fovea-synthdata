import warnings
from collections import Counter

import pytest
import yaml

from synthdata.generators.cell_plate.check import check_dataset
from synthdata.generators.cell_plate.design import bundled_designs, load_design
from synthdata.generators.cell_plate.generator import CellPlateGenerator
from synthdata.generators.cell_plate.layout import assign
from synthdata.generators.cell_plate.reader import read_platemap

NAMES = ["moa", "moa_tuning", "moa_small"]
MECHANISMS = {"A": "actin_disruptor", "B": "mitochondrial_toxin", "C": "microtubule_inhibitor",
              "D": "dna_damage", "E": "hdac_inhibitor"}
MOA_COMPOUNDS = [f"MOA-{m}{i}" for m in "ABCDE" for i in (1, 2, 3)]


def _design(name):
    return load_design(bundled_designs()[name].read_text())


def job_count(design):
    """5 x sites + wells + 4 x plates (stages 3, 5, 8, 10) + 7 run stages."""
    p = design.plate
    rows, cols = p.grid()
    wells = p.plates * rows * cols
    return 5 * wells * p.sites_per_well + wells + 4 * p.plates + 7


@pytest.mark.parametrize("name", NAMES)
def test_layout_counts(name):
    d = _design(name)
    slots = assign(d)
    for p in (0, 1):
        roles = Counter(s.role for s in slots if s.plate == p)
        assert roles == {"negative_control": 8, "positive_control": 8, "blank": 8, "test": 72}
    per_mechanism = Counter(c.mechanism for c in d.compounds.values() if c.mechanism)
    assert len(per_mechanism) == 5 and set(per_mechanism.values()) == {3}
    assert set(per_mechanism) == set(MECHANISMS.values())
    assert {n for n, c in d.compounds.items() if c.role == "test"} == set(MOA_COMPOUNDS) | {"NULL-1", "NULL-2", "LONE-1"}
    assert {s.dose_label for s in slots if s.role == "test"} == {"1 uM", "10 uM"}
    per_compound = Counter(s.compound for s in slots if s.role == "test")
    assert set(per_compound.values()) == {8} and len(per_compound) == 18  # 2 doses x 2 replicates x 2 plates


@pytest.mark.parametrize("name", NAMES)
def test_each_mechanism_has_a_strong_middle_and_weak_compound(name):
    d = _design(name)
    for letter, mech in MECHANISMS.items():
        potency = [d.compounds[f"MOA-{letter}{i}"].potency for i in (1, 2, 3)]
        assert all(c.mechanism == mech for c in (d.compounds[f"MOA-{letter}{i}"] for i in (1, 2, 3)))
        assert potency[0] > potency[1] > potency[2] > 0
        assert potency[0] <= 1.0
        phenos = {d.compounds[f"MOA-{letter}{i}"].phenotype for i in (1, 2, 3)}
        assert len(phenos) == 1  # same mechanism, same planted family
    for n in ("NULL-1", "NULL-2", "LONE-1"):
        assert d.compounds[n].mechanism is None


@pytest.mark.parametrize("name", NAMES)
def test_every_compound_has_a_different_side_effect(name):
    d = _design(name)
    side = {n: d.compounds[n].side_effect for n in MOA_COMPOUNDS}
    assert all(s is not None and s.strength == 0.6 for s in side.values())
    assert len({s.phenotype for s in side.values()}) == 15
    for n in ("NULL-1", "NULL-2", "LONE-1"):
        assert d.compounds[n].side_effect is None
    # a side effect never repeats a knob its own family plants
    for n in MOA_COMPOUNDS:
        own = d.phenotypes[d.compounds[n].phenotype].model_dump()
        extra = d.phenotypes[side[n].phenotype].model_dump()
        neutral = type(d.phenotypes[d.compounds[n].phenotype])().model_dump()
        planted = {k for k, v in own.items() if v != neutral[k]}
        touched = {k for k, v in extra.items() if v != neutral[k]}
        assert not planted & touched, (n, planted & touched)


@pytest.mark.parametrize("name", NAMES)
def test_expectations_cover_every_planted_family(name):
    d = _design(name)
    assert d.expectations.reference == "DMSO"
    assert d.expectations.same == ["NULL-1", "NULL-2"]
    assert set(d.expectations.differs) == set(MOA_COMPOUNDS) | {"LONE-1"}
    assert d.compounds["NULL-1"].phenotype == d.compounds["NULL-2"].phenotype == "no_effect"
    assert d.compounds["LONE-1"].phenotype == "cell_elongate"


def _without_seed(name):
    raw = yaml.safe_load(bundled_designs()[name].read_text())
    raw.pop("seed")
    return raw


def test_tuning_and_moa_differ_only_in_seed():
    assert _without_seed("moa") == _without_seed("moa_tuning")
    assert _design("moa").seed == 2031 and _design("moa_tuning").seed == 4243
    assert _design("moa_small").seed == 2032


ROUND5_FRACTIONS = {1.0: 0.8, 0.7: 0.65, 0.45: 0.5}


def _before_round5(raw):
    """moa's values as they were before round 5: no well effects, no partial response, MOA-B3 on side_fibre_up."""
    raw = {**raw, "phenotypes": dict(raw["phenotypes"]), "compounds": {k: dict(v) for k, v in raw["compounds"].items()}}
    raw.pop("well_effects")
    for c in raw["compounds"].values():
        c.pop("responder_fraction", None)
    raw["phenotypes"].pop("side_foci")
    raw["phenotypes"]["side_fibre_up"] = {"agp_fibre_factor": 1.5}
    raw["compounds"]["MOA-B3"]["side_effect"] = {"phenotype": "side_fibre_up", "strength": 0.6}
    return raw


@pytest.mark.parametrize("name", ["moa", "moa_tuning"])
def test_round5_values(name):
    d = _design(name)
    fx = d.well_effects
    assert (fx.gain_sd, fx.count_sd, fx.size_sd, fx.response_sd) == (0.05, 0.10, 0.03, 0.25)
    for n in MOA_COMPOUNDS:
        assert d.compounds[n].responder_fraction == ROUND5_FRACTIONS[d.compounds[n].potency]
    assert d.compounds["LONE-1"].responder_fraction == 0.7
    assert all(d.compounds[n].responder_fraction is None for n in ("NULL-1", "NULL-2"))
    assert d.compounds["MOA-B3"].side_effect.phenotype == "side_foci"
    assert d.phenotypes["side_foci"].dna_foci_add == 5 and "side_fibre_up" not in d.phenotypes


def test_moa_small_keeps_v4_values_with_the_new_knobs_off():
    small = _design("moa_small")
    assert small.well_effects is None and all(c.responder_fraction is None for c in small.compounds.values())
    assert small.compounds["MOA-B3"].side_effect.phenotype == "side_fibre_up"


def test_moa_small_differs_from_moa_only_in_size_and_round5():
    big, small = _before_round5(_without_seed("moa")), _without_seed("moa_small")
    assert big["plate"]["sites_per_well"] == 2 and small["plate"]["sites_per_well"] == 1
    assert big["plate"]["image"]["height"] == 1024 and small["plate"]["image"]["height"] == 512
    assert big["base_cell"]["count"]["mean"] == 120 and small["base_cell"]["count"]["mean"] == 30
    for d in (big, small):
        d["plate"]["image"].update(height=0, width=0)
        d["plate"]["sites_per_well"] = 0
        d["base_cell"]["count"]["mean"] = 0
    assert big == small


@pytest.mark.parametrize("name", NAMES)
def test_plates_and_defects(name):
    d = _design(name)
    assert d.plate.plates == 2 and len(d.plate_effects) == 2
    assert d.plate_effects[0].gain == {} and d.plate_effects[0].background_shift == {}
    assert d.plate_effects[0].count_factor == 1.0
    p2 = d.plate_effects[1]
    assert any(g > 1 for g in p2.gain.values()) and any(g < 1 for g in p2.gain.values())
    assert max(p2.background_shift.values()) <= 30 and p2.count_factor == 0.9
    assert d.qc_defects.blurred_sites.n == 3 and d.qc_defects.saturated_debris.n == 2


def test_job_counts_match_section_7():
    # 5 x sites + wells + 4 x plates (stages 3, 5, 8, 10) + 7 run stages
    assert job_count(_design("moa")) == 5 * 384 + 192 + 8 + 7 == 2127
    assert job_count(_design("moa_small")) == 5 * 192 + 192 + 8 + 7 == 1167


def test_tiny_moa_small_passes_stage_one(tmp_path):
    raw = yaml.safe_load(bundled_designs()["moa_small"].read_text())
    raw["plate"]["image"].update(height=128, width=128)
    raw["base_cell"]["count"]["mean"] = 4
    d = load_design(yaml.safe_dump(raw))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="cell_plate", category=UserWarning)  # dense or crowded sites
        CellPlateGenerator().generate(d, tmp_path, writer="tiff")
    rep = check_dataset(tmp_path, write=False)
    assert all(c.passed for c in rep.stage1), [c.name for c in rep.stage1 if not c.passed]
    pm = read_platemap(tmp_path)
    assert set(pm.plate) == {"plate01", "plate02"}
    assert "moa" in pm.columns
