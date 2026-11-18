from collections import Counter

import numpy as np
import pytest
import yaml
from scipy import ndimage as ndi

from synthdata.generators.cell_plate.design import bundled_designs, load_design
from synthdata.generators.cell_plate.layout import assign
from synthdata.generators.cell_plate.qc import NS_DEFECT, assign_defects
from synthdata.generators.cell_plate.render import render_site

H = W = 256


def _design(qc="keep", **qc_edits):
    data = yaml.safe_load(bundled_designs()["default"].read_text())
    data["plate"]["image"] = {"height": H, "width": W, "dtype": "uint16", "pixel_size_um": 0.65}
    data["base_cell"]["count"] = {"dist": "poisson", "mean": 6}
    if qc is None:
        data.pop("qc_defects")
    for k, v in qc_edits.items():
        data["qc_defects"][k] = v
    return load_design(data)


@pytest.fixture(scope="module")
def design():
    return _design()


@pytest.fixture(scope="module")
def slots(design):
    return assign(design)


@pytest.fixture(scope="module")
def defects(design, slots):
    return assign_defects(design, slots)


def _slot_of(slots, key):
    return next(s for s in slots if (s.plate, s.row, s.col) == key[:3])


def _first(defects, kind):
    return next((k, v) for k, v in sorted(defects.items()) if v.kind == kind)


def _clean_key(design, slots, defects, role="test"):
    s = next(s for s in slots if s.role == role and not any(k[:3] == (s.plate, s.row, s.col) for k in defects))
    return (s.plate, s.row, s.col, 1)


def test_namespace():
    assert NS_DEFECT == 13


def test_defect_counts_and_targets(design, slots, defects):
    assert Counter(x.kind for x in defects.values()) == {"blur": 6, "saturated_debris": 4}
    wells = [k[:3] for k in defects]
    assert len(set(wells)) == len(wells)  # at most one per well
    for k in defects:
        assert _slot_of(slots, k).role not in ("blank", "negative_control")
        assert 1 <= k[3] <= design.plate.sites_per_well
    assert all(d.psf_sigma_px == 3.5 for d in defects.values() if d.kind == "blur")
    assert all(d.psf_sigma_px is None for d in defects.values() if d.kind == "saturated_debris")


def test_assignment_is_deterministic(design, slots, defects):
    assert assign_defects(design, slots) == defects


def test_no_block_means_no_defects(slots):
    assert assign_defects(_design(qc=None), slots) == {}


def test_too_many_defects_names_qc_defects(slots):
    d = _design(blurred_sites={"n": 500, "psf_sigma_px": 3.5})
    with pytest.raises(ValueError, match="qc_defects"):
        assign_defects(d, slots)


def _hf(img):
    return float(np.abs(ndi.laplace(img.astype(float))).mean())


def test_blurred_site_is_blurrier_in_every_channel(design, slots, defects):
    key, dfct = _first(defects, "blur")
    slot = _slot_of(slots, key)
    blurred = render_site(design, slot, key[3], dfct)
    plain = render_site(design, slot, key[3], None)
    for ch in design.plate.channels:
        assert _hf(blurred.channels[ch.name]) < _hf(plain.channels[ch.name])
    assert blurred.qc_rows == [{"plate": slot.plate_name, "well": slot.well, "site": key[3], "defect": "blur",
                                "degree": 1.0, "label": "fail"}]


def test_saturated_site_has_clipped_pixels(design, slots, defects):
    key, dfct = _first(defects, "saturated_debris")
    slot = _slot_of(slots, key)
    res = render_site(design, slot, key[3], dfct)
    maxv = np.iinfo(design.plate.image.dtype).max
    assert sum(int((img == maxv).sum()) for img in res.channels.values()) > 0
    assert res.qc_rows[0]["defect"] == "saturated_debris" and res.qc_rows[0]["label"] == "fail"
    clean = render_site(design, slot, key[3], None)
    assert sum(int((img == maxv).sum()) for img in clean.channels.values()) == 0


def test_clean_site_gets_a_pass_row(design, slots, defects):
    key = _clean_key(design, slots, defects)
    slot = _slot_of(slots, key)
    res = render_site(design, slot, key[3], None)
    assert res.qc_rows == [{"plate": slot.plate_name, "well": slot.well, "site": key[3], "defect": "none",
                            "degree": 0.0, "label": "pass"}]


def test_no_defect_block_renders_identically(design, slots, defects):
    bare = _design(qc=None)
    bare_slots = assign(bare)
    assert bare_slots == slots
    for key in (_clean_key(design, slots, defects), _clean_key(design, slots, defects, "positive_control")):
        slot = _slot_of(slots, key)
        a = render_site(design, slot, key[3], defects.get(key))
        b = render_site(bare, slot, key[3], assign_defects(bare, bare_slots).get(key))
        for name in a.channels:
            assert a.channels[name].tobytes() == b.channels[name].tobytes()


def _two_plate_design():
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate"]["plates"] = 2
    data["qc_defects"] = {"blurred_sites": {"n": 3, "psf_sigma_px": 3.5}, "saturated_debris": {"n": 2}}
    return load_design(data)


def test_defects_are_per_plate():
    d2 = _two_plate_design()
    d = assign_defects(d2, assign(d2))
    for p in (0, 1):
        kinds = [v.kind for k, v in d.items() if k[0] == p]
        assert kinds.count("blur") == 3 and kinds.count("saturated_debris") == 2


def test_plate_zero_defects_match_the_one_plate_design():
    d2 = _two_plate_design()
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate"]["plates"] = 1
    data["qc_defects"] = {"blurred_sites": {"n": 3, "psf_sigma_px": 3.5}, "saturated_debris": {"n": 2}}
    d1 = load_design(data)
    two = assign_defects(d2, assign(d2))
    assert {k: v for k, v in two.items() if k[0] == 0} == assign_defects(d1, assign(d1))
