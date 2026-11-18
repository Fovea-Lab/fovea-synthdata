"""Per-well random variation (`well_effects`) and partial response (`responder_fraction`). Both are off by default."""

from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from synthdata.core.provenance import read_manifest
from synthdata.core.rng import stream
from synthdata.generators.cell_plate.cells import sample_cells
from synthdata.generators.cell_plate.design import Phenotype, bundled_designs, dump_design, load_design
from synthdata.generators.cell_plate.generator import CellPlateGenerator
from synthdata.generators.cell_plate.layout import WellSlot, assign
from synthdata.generators.cell_plate.render import NS_CELLS, render_site
from synthdata.generators.cell_plate.well_effects import draw_well

pytestmark = pytest.mark.filterwarnings("ignore:cell_plate:UserWarning")  # small sites are dense

gen = CellPlateGenerator()
FX = {"gain_sd": 0.05, "count_sd": 0.1, "size_sd": 0.03, "response_sd": 0.25}


def _raw(size=160, count=10):
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate"]["image"] = {"height": size, "width": size, "dtype": "uint16", "pixel_size_um": 0.65}
    data["base_cell"]["count"] = {"dist": "poisson", "mean": count}
    return data


def _design(well_effects=None, fractions=None, **kw):
    data = _raw(**kw)
    if well_effects is not None:
        data["well_effects"] = well_effects
    for name, f in (fractions or {}).items():
        data["compounds"][name]["responder_fraction"] = f
    return load_design(data)


def _slot(design, compound):
    return next(s for s in assign(design) if s.compound == compound and s.dose_label in ("", "10 uM"))


def _plan(design, slot, site=1):
    """The site's cells as `render_site` draws them, with the well's draws."""
    key = (slot.plate, slot.row, slot.col, site)
    return sample_cells(design, slot, site, stream(design.seed, NS_CELLS, *key), False, draw_well(design, slot))


# ---- defaults -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(set(bundled_designs()) - {"moa", "moa_tuning"}))  # round 5 sets them there
def test_bundled_designs_without_the_knobs_dump_without_them(name):
    d = load_design(bundled_designs()[name].read_text())
    assert d.well_effects is None
    assert all(c.responder_fraction is None for c in d.compounds.values())
    text = dump_design(d)
    assert "well_effects" not in text and "responder_fraction" not in text


def test_off_by_default_writes_no_new_truth(tmp_path):
    gen.generate(_design(), tmp_path, writer="tiff")
    assert not (tmp_path / "truth" / "well_effects.parquet").exists()
    assert "responder" not in pq.read_table(tmp_path / "truth" / "cells.parquet").column_names
    assert "responder_fraction" not in pq.read_table(tmp_path / "truth" / "wells.parquet").column_names


def test_responder_fraction_only_on_test_compounds():
    data = _raw()
    data["compounds"]["POS-CTRL"]["responder_fraction"] = 0.5
    with pytest.raises(ValueError, match="responder_fraction"):
        load_design(data)
    data = _raw()
    data["compounds"]["CPD-002"]["responder_fraction"] = 0.0
    with pytest.raises(ValueError):
        load_design(data)


# ---- the well draws -------------------------------------------------------------------------------------------------

def test_draws_are_reproducible_and_differ_between_wells():
    d = _design(FX)
    slots = [s for s in assign(d) if s.role == "test"]
    a, b = draw_well(d, slots[0]), draw_well(d, slots[0])
    assert a == b
    assert draw_well(d, slots[1]) != a
    assert set(a.gain) == {c.name for c in d.plate.channels}
    assert all(v > 0 for v in a.gain.values()) and a.count_factor > 0 and a.size_factor > 0
    assert a.response_factor is not None and a.response_factor >= 0


def test_a_wells_draws_depend_on_its_position_not_on_other_wells():
    d = _design(FX)
    s = _slot(d, "CPD-003")
    moved = replace(s, compound="CPD-004", phenotype="cell_enlarge")  # same well, another compound
    assert draw_well(d, s).gain == draw_well(d, moved).gain


def test_blank_has_no_draws_and_dmso_has_no_response():
    d = _design(FX)
    slots = assign(d)
    assert draw_well(d, next(s for s in slots if s.role == "blank")) is None
    dmso = draw_well(d, next(s for s in slots if s.role == "negative_control"))
    assert dmso is not None and dmso.response_factor is None
    assert draw_well(_design(None), slots[0]) is None


def test_sd_zero_draws_exactly_one():
    d = _design({"gain_sd": 0, "count_sd": 0, "size_sd": 0, "response_sd": 0})
    w = draw_well(d, _slot(d, "CPD-003"))
    assert set(w.gain.values()) == {1.0} and (w.count_factor, w.size_factor, w.response_factor) == (1.0, 1.0, 1.0)


def test_response_is_clipped_at_zero():
    d = _design({"response_sd": 50.0})
    factors = [draw_well(d, s).response_factor for s in assign(d) if s.role == "test"]
    assert min(factors) == 0.0 and max(factors) > 1.0


def test_response_scales_the_main_strength_not_the_side_effect():
    data = _raw()
    data["phenotypes"]["side"] = {"rna_intensity_factor": 1.5}
    data["compounds"]["CPD-003"]["side_effect"] = {"phenotype": "side", "strength": 0.5}
    data["well_effects"] = {"response_sd": 0.5}
    d = load_design(data)
    s = _slot(d, "CPD-003")
    r = draw_well(d, s).response_factor
    cell = _plan(d, s).cells[0].params
    main = d.phenotypes["nuclear_shrink"].at(min(s.strength * r, 1.0))
    assert r != 1.0
    assert cell["nucleus_diameter_factor"] == pytest.approx(main.nucleus_diameter_factor)
    assert cell["rna_intensity_factor"] == pytest.approx(main.rna_intensity_factor * d.phenotypes["side"].at(s.side_strength).rna_intensity_factor)


def test_size_factor_scales_body_and_nucleus():
    big = _design({"size_sd": 0.0}, size=384, count=4)
    s = _slot(big, "CPD-001")
    plain = _plan(big, s)
    w = draw_well(_design({"size_sd": 0.3}, size=384, count=4), s)
    sized = _plan(_design({"size_sd": 0.3}, size=384, count=4), s)
    assert len(sized.cells) > 0 and w.size_factor != 1.0
    first = sized.cells[0]
    assert first.body_diameter == pytest.approx(plain.cells[0].body_diameter * w.size_factor)


def test_gain_multiplies_the_signal():
    fx = {"gain_sd": 0.4}
    d0 = _design({"gain_sd": 0.0})
    d1 = _design(fx)
    s = _slot(d0, "DMSO")
    d0q = load_design({**yaml.safe_load(dump_design(d0)), "image_effects": {"noise": {"poisson": False, "gaussian_sd": 0},
                       "background": {"dist": "fixed", "value": 0.0}}})
    d1q = load_design({**yaml.safe_load(dump_design(d1)), "image_effects": {"noise": {"poisson": False, "gaussian_sd": 0},
                       "background": {"dist": "fixed", "value": 0.0}}})
    g = draw_well(d1q, s).gain["DNA"]
    a = render_site(d0q, s, 1).channels["DNA"].astype(float)
    b = render_site(d1q, s, 1).channels["DNA"].astype(float)
    assert b.sum() == pytest.approx(a.sum() * g, rel=0.01)


# ---- partial response -----------------------------------------------------------------------------------------------

def test_non_responders_carry_no_main_effect_and_side_reaches_all():
    data = _raw(size=256, count=40)
    data["phenotypes"]["side"] = {"rna_intensity_factor": 1.5}
    data["compounds"]["CPD-003"]["side_effect"] = {"phenotype": "side", "strength": 1.0}
    data["compounds"]["CPD-003"]["responder_fraction"] = 0.5
    d = load_design(data)
    s = _slot(d, "CPD-003")
    cells = _plan(d, s).cells
    resp = [c for c in cells if c.params["responder"]]
    non = [c for c in cells if not c.params["responder"]]
    assert resp and non
    side = d.phenotypes["side"].at(s.side_strength).rna_intensity_factor
    for c in non:
        assert c.params["nucleus_diameter_factor"] == 1.0 and c.params["dna_intensity_factor"] == 1.0
        assert c.params["rna_intensity_factor"] == side
    main = d.phenotypes["nuclear_shrink"].at(s.strength)
    for c in resp:
        assert c.params["nucleus_diameter_factor"] == main.nucleus_diameter_factor
        assert c.params["rna_intensity_factor"] == pytest.approx(main.rna_intensity_factor * side)


def test_responder_share_follows_the_fraction():
    d = _design(fractions={"CPD-003": 0.3}, size=512, count=200)
    cells = _plan(d, _slot(d, "CPD-003")).cells
    share = np.mean([c.params["responder"] for c in cells])
    assert 0.2 < share < 0.4


def test_fraction_one_and_sd_zero_equal_the_old_behaviour():
    old = _design()
    new = _design({"gain_sd": 0, "count_sd": 0, "size_sd": 0, "response_sd": 0},
                  {n: 1.0 for n, c in old.compounds.items() if c.role == "test"})
    for compound in ("DMSO", "POS-CTRL", "CPD-003", "CPD-008"):
        s_old, s_new = _slot(old, compound), _slot(new, compound)
        a, b = render_site(old, s_old, 1), render_site(new, s_new, 1)
        for ch in a.channels:
            assert np.array_equal(a.channels[ch], b.channels[ch])
        assert np.array_equal(a.nuclei_mask, b.nuclei_mask) and np.array_equal(a.cell_mask, b.cell_mask)
        strip = [{k: v for k, v in r.items() if k != "responder"} for r in b.cell_rows]
        assert a.cell_rows == strip
        assert all(r.get("responder") is (None if compound == "DMSO" else True) for r in b.cell_rows)


# ---- generation -----------------------------------------------------------------------------------------------------

def test_truth_records_draws_and_responders_and_is_worker_independent(tmp_path):
    data = _raw(size=128, count=8)
    data["plate"]["plates"] = 2
    data["well_effects"] = FX
    data["compounds"]["CPD-003"]["responder_fraction"] = 0.5
    data.pop("expectations")
    d = load_design(data)
    m1 = gen.generate(d, tmp_path / "w1", writer="tiff", workers=1)
    m2 = gen.generate(d, tmp_path / "w2", writer="tiff", workers=2)
    assert m1.files == m2.files
    rows = pq.read_table(tmp_path / "w1" / "truth" / "well_effects.parquet").to_pylist()
    slots = assign(d)
    assert len(rows) == sum(s.role != "blank" for s in slots)
    by = {(r["plate"], r["well"]): r for r in rows}
    s = _slot(d, "CPD-003")
    w = draw_well(d, s)
    r = by[(s.plate_name, s.well)]
    assert r["gain_DNA"] == w.gain["DNA"] and r["count_factor"] == w.count_factor
    assert r["size_factor"] == w.size_factor and r["response_factor"] == w.response_factor
    dmso = next(x for x in slots if x.role == "negative_control")
    assert by[(dmso.plate_name, dmso.well)]["response_factor"] is None
    cells = pq.read_table(tmp_path / "w1" / "truth" / "cells.parquet").to_pandas()
    cpd = cells[cells.compound == "CPD-003"]
    assert set(cpd.responder) == {True, False}
    assert cells[cells.compound == "DMSO"].responder.isna().all()
    wells = pq.read_table(tmp_path / "w1" / "truth" / "wells.parquet").to_pandas()
    assert set(wells[wells.condition.str.startswith("CPD-003")].responder_fraction) == {0.5}
    assert "well_effects" in (tmp_path / "w1" / "design.yaml").read_text()
    assert read_manifest(tmp_path / "w1").files
    assert isinstance(_slot(d, "CPD-003"), WellSlot)


# ---- the cap on strength and the floor on factors ------------------------------------------------------------------

def test_a_factor_never_goes_below_zero():
    p = Phenotype(agp_fibre_factor=0.1, mito_length_factor=0.35, body_eccentricity_add=-0.2).at(1.47)
    assert p.agp_fibre_factor == 0.0 and p.mito_length_factor == pytest.approx(1 - 1.47 * 0.65)
    assert p.body_eccentricity_add == pytest.approx(-0.294)  # adds are offsets, not clipped
    for s in (0.0, 0.18, 0.45, 0.7, 1.0):  # at or below strength 1, unchanged from 1 + s(f - 1)
        assert Phenotype(agp_fibre_factor=0.1).at(s).agp_fibre_factor == 1.0 + s * (0.1 - 1.0)


def _response_design(response):
    data = _raw()
    data["phenotypes"]["fibre"] = {"agp_fibre_factor": 0.1}
    data["compounds"]["CPD-009"]["phenotype"] = "fibre"
    data["well_effects"] = {"response_sd": 0.0}
    d = load_design(data)
    return d, _slot(d, "CPD-009")


def _cells_at(d, s, response):
    w = replace(draw_well(d, s), response_factor=response)
    key = (s.plate, s.row, s.col, 1)
    return sample_cells(d, s, 1, stream(d.seed, NS_CELLS, *key), False, w).cells


def test_response_is_capped_at_full_strength():
    d, s = _response_design(1.47)
    assert s.strength == 1.0
    cells = _cells_at(d, s, 1.47)
    assert cells and all(c.params["agp_fibre_factor"] == pytest.approx(0.1) for c in cells)  # not 1 - 1.47 x 0.9 < 0


def test_the_cap_leaves_strength_at_or_below_one_unchanged():
    d, s = _response_design(0.8)
    for r in (0.0, 0.5, 0.8, 1.0):
        want = d.phenotypes["fibre"].at(s.strength * r).agp_fibre_factor
        assert all(c.params["agp_fibre_factor"] == want for c in _cells_at(d, s, r))
    low = replace(s, strength=0.4)  # 1 uM: 0.4 x 2.0 = 0.8 stays as it is
    w = replace(draw_well(d, low), response_factor=2.0)
    cells = sample_cells(d, low, 1, stream(d.seed, NS_CELLS, low.plate, low.row, low.col, 1), False, w).cells
    assert all(c.params["agp_fibre_factor"] == pytest.approx(1 - 0.8 * 0.9) for c in cells)
