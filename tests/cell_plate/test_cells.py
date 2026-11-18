import pickle
import time
import warnings
from dataclasses import replace

import numpy as np
import pytest
import yaml
from skimage.draw import ellipse as draw_ellipse
from skimage.measure import regionprops

from synthdata.core.rng import stream
from synthdata.generators.cell_plate.cells import SitePlan, rasterise, sample_cells
from synthdata.generators.cell_plate.design import bundled_designs, load_design
from synthdata.generators.cell_plate.layout import WellSlot

SHAPE = (384, 384)


def _design(**edits):
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    for dotted, value in edits.items():
        node = data
        *path, last = dotted.split("__")
        for p in path:
            node = node[p]
        node[last] = value
    return load_design(data)


def _slot(phenotype=None, strength=0.0, role="negative_control", row=3, col=5):
    return WellSlot(0, row, col, "X", "", "X", role, 1, phenotype, strength)


def _plan(design, slot, seed=1, outer=False):
    return sample_cells(design, slot, 0, stream(seed, 99), outer)


@pytest.fixture(scope="module")
def plan_labels():
    plan = _plan(_design(), _slot())
    return plan, rasterise(plan, SHAPE)


def _alone(plan, cell, nucleus=None):
    """Label image of one cell's body alone, or of one nucleus alone."""
    solo = replace(cell, nuclei=[] if nucleus is None else [nucleus])
    return rasterise(SitePlan([solo], 1, 1), SHAPE)


def test_nuclei_inside_their_bodies_and_never_overlap(plan_labels):
    plan, (nuclei, cells) = plan_labels
    assert nuclei.dtype == cells.dtype == np.uint32
    assert len(plan.cells) > 5
    masks = []
    for cell in plan.cells:
        body = _alone(plan, cell)[1] > 0
        for nu in cell.nuclei:
            mask = _alone(plan, cell, nu)[0] > 0
            assert mask.any()
            assert body[mask].all()  # nucleus lies inside its own body, painted without the nucleus
            assert (nuclei == nu.nucleus_id).sum() == mask.sum()  # nothing else overwrote it
            masks.append(mask)
    assert len(masks) == len(regionprops(nuclei))
    assert np.sum(masks) == np.any(masks, axis=0).sum()  # pairwise disjoint
    assert (cells[nuclei > 0] > 0).all()


@pytest.mark.filterwarnings("ignore:cell_plate. placed")
def test_dense_overlap_keeps_every_nucleus_pixel_in_its_own_cell():
    design = _design(base_cell__count={"dist": "fixed", "value": 60}, base_cell__overlap_fraction=0.6)
    for seed in range(8):
        plan = _plan(design, _slot(), seed=seed)
        nuclei, cells = rasterise(plan, SHAPE)
        for c in plan.cells:
            for nu in c.nuclei:
                assert (cells[nuclei == nu.nucleus_id] == c.cell_id).all()


@pytest.mark.filterwarnings("ignore:cell_plate. placed")
def test_supersampled_labels_align_with_native_labels():
    plan = _plan(_design(), _slot())
    for native, fine in zip(rasterise(plan, SHAPE), rasterise(plan, SHAPE, supersample=2)):
        fg = (native > 0) | (fine > 0)
        assert (native[fg] == fine[fg]).mean() >= 0.95


def test_blank_well_has_no_cells():
    slot = _slot(role="blank")
    plan = _plan(_design(), slot)
    assert (plan.n_requested, plan.n_placed, plan.cells) == (0, 0, [])
    nuclei, cells = rasterise(plan, SHAPE)
    assert not nuclei.any() and not cells.any()


def test_shapes_are_lumpy_not_ellipses():
    design = _design(base_cell__count={"dist": "fixed", "value": 4},
                     plate__image={"height": 384, "width": 384, "dtype": "uint16", "pixel_size_um": 0.65})
    solidity, diff = [], []
    for seed in range(5):
        plan = _plan(design, _slot(), seed=seed)
        _, cells = rasterise(plan, SHAPE)
        for r in regionprops(cells):
            if r.bbox[0] == 0 or r.bbox[1] == 0 or r.bbox[2] == SHAPE[0] or r.bbox[3] == SHAPE[1]:
                continue
            fit = np.zeros(SHAPE, bool)
            rr, cc = draw_ellipse(*r.centroid, r.axis_major_length / 2, r.axis_minor_length / 2,
                                  shape=SHAPE, rotation=r.orientation)
            fit[rr, cc] = True
            mask = cells == r.label
            solidity.append(r.solidity)
            diff.append((fit ^ mask).sum() / r.area)
    assert len(solidity) >= 8
    assert np.mean(solidity) < 1
    assert np.mean(diff) > 0.01


def _mean_diameter(strength, n_seeds=6):
    design = _design(base_cell__count={"dist": "fixed", "value": 8})
    vals = [c.body_diameter for s in range(n_seeds)
            for c in _plan(design, _slot("cell_enlarge", strength, role="test"), seed=s).cells]
    return float(np.mean(vals))


def test_dose_scales_effect():
    none, low, high = _mean_diameter(0.0), _mean_diameter(0.4), _mean_diameter(1.0)
    assert high > low > none
    assert high / none == pytest.approx(1.4, rel=0.15)


def test_multinucleate_has_two_to_three_nuclei():
    design = _design(base_cell__count={"dist": "fixed", "value": 12})
    counts = [len(c.nuclei) for s in range(20)
              for c in _plan(design, _slot("multinucleate", 1.0, role="test"), seed=s).cells]
    counts = np.array(counts)
    assert counts.max() <= 3
    frac = (counts >= 2).mean()
    assert 0.45 < frac < 0.75  # multinucleate_add = 0.6
    assert set(counts) <= {1, 2, 3}
    plan = _plan(design, _slot("multinucleate", 1.0, role="test"))
    nuclei, cells = rasterise(plan, SHAPE)
    n_nuc = sum(len(c.nuclei) for c in plan.cells)
    assert len(np.unique(nuclei)) - 1 == n_nuc
    assert (cells[nuclei > 0] > 0).all()
    neutral = [len(c.nuclei) for s in range(5) for c in _plan(design, _slot(), seed=s).cells]
    assert max(neutral) == 1


@pytest.mark.filterwarnings("ignore:cell_plate. placed")
def test_cytotoxic_reduces_count():
    design = _design()
    base = np.mean([_plan(design, _slot(), seed=s).n_requested for s in range(30)])
    tox = np.mean([_plan(design, _slot("cytotoxic", 1.0, role="test"), seed=s).n_requested for s in range(30)])
    assert tox < 0.6 * base


@pytest.mark.filterwarnings("ignore:cell_plate. placed")
def test_outer_well_reduces_count_when_edge_effect_on():
    on = _design()
    off = _design(image_effects__edge_effect={"enabled": False})
    n = lambda d, outer: np.mean([_plan(d, _slot(), seed=s, outer=outer).n_requested for s in range(40)])
    assert n(on, True) < 0.9 * n(on, False)
    assert n(off, True) == pytest.approx(n(off, False))


def test_too_dense_design_records_shortfall_and_warns():
    design = _design(base_cell__count={"dist": "fixed", "value": 2000})
    t0 = time.time()
    with pytest.warns(UserWarning, match="placed"):
        plan = _plan(design, _slot())
    assert time.time() - t0 < 10
    assert plan.n_requested == 2000
    assert 0 < plan.n_placed < plan.n_requested
    assert plan.n_placed == len(plan.cells)


def test_sparse_design_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _plan(_design(), _slot())


@pytest.mark.filterwarnings("ignore:cell_plate. placed")
def test_deterministic_and_picklable():
    a = _plan(_design(), _slot("cell_enlarge", 1.0, role="test"), seed=3)
    b = _plan(_design(), _slot("cell_enlarge", 1.0, role="test"), seed=3)
    assert all((x.x, x.y, x.body_diameter) == (y.x, y.y, y.body_diameter) for x, y in zip(a.cells, b.cells))
    c = pickle.loads(pickle.dumps(a))
    th = np.linspace(0, 6.28, 20)
    assert np.allclose(c.cells[0].body_radius_fn(th), a.cells[0].body_radius_fn(th))
    assert a.cells[0].params["body_diameter_factor"] == pytest.approx(1.4)


def test_side_effect_reaches_the_rendered_cell_params():
    design = _design(phenotypes__x={"count_factor": 0.5, "dna_foci_add": 4.0})
    main = design.phenotypes["cytotoxic"].at(0.6)
    side = design.phenotypes["x"].at(0.3)
    both = replace(_slot("cytotoxic", 0.6, "test"), side_effect="x", side_strength=0.3)
    plan = _plan(design, both)
    assert plan.cells
    want = main.combined(side)
    for cell in plan.cells:
        assert cell.params["count_factor"] == pytest.approx(want.count_factor)
        assert cell.params["count_factor"] == pytest.approx(main.count_factor * side.count_factor)
        assert cell.params["dna_foci_add"] == pytest.approx(want.dna_foci_add) and want.dna_foci_add > 0
    plain = _plan(design, _slot("cytotoxic", 0.6, "test"))
    assert all(c.params["dna_foci_add"] == 0.0 for c in plain.cells)
