import pickle
from dataclasses import replace

import numpy as np
import pytest
import yaml
from scipy import ndimage as ndi

from synthdata.core.rng import stream
from synthdata.generators.cell_plate.cells import rasterise, sample_cells
from synthdata.generators.cell_plate.design import bundled_designs, load_design
from synthdata.generators.cell_plate.layout import WellSlot
from synthdata.generators.cell_plate.render import (
    NS_CELLS, NS_CHANNEL, NS_NOISE, SiteResult, downsample, illumination_field, render_ideal, render_site)

H = W = 384
SHAPE = (H, W)
SS = 2


def _design(name="default", **edits):
    data = yaml.safe_load(bundled_designs()[name].read_text())
    data["plate"]["image"] = {"height": H, "width": W, "dtype": "uint16", "pixel_size_um": 0.65}
    data["base_cell"]["count"] = {"dist": "poisson", "mean": 18}
    for dotted, value in edits.items():
        node = data
        *path, last = dotted.split("__")
        for p in path:
            node = node[p]
        node[last] = value
    return load_design(data)


def _quiet(name="default", **edits):
    """No noise, no illumination, no edge dim, zero background unless edited: pixel values are the signal."""
    quiet = {"image_effects__noise": {"poisson": False, "gaussian_sd": 0.0},
             "image_effects__illumination": {"kind": "none", "strength": 0.0},
             "image_effects__edge_effect": {"enabled": False},
             "image_effects__background": {"dist": "fixed", "value": 0.0}}
    return _design(name, **{**quiet, **edits})


def _slot(phenotype=None, strength=0.0, role="negative_control", row=3, col=5):
    return WellSlot(0, row, col, "X", "", "X", role, 1, phenotype, strength)


def _plan(design, slot=None, site=1):
    slot = slot or _slot()
    return sample_cells(design, slot, site, stream(design.seed, NS_CELLS, 0, slot.row, slot.col, site), False)


def _ideal(plan, design, seed=5):
    return render_ideal(plan, design, stream(seed, NS_CHANNEL), SHAPE)


def _with(plan, **params):
    return replace(plan, cells=[replace(c, params={**c.params, **params}) for c in plan.cells])


def _native(a):
    return downsample(a, SS)


@pytest.fixture(scope="module")
def design():
    return _quiet()


@pytest.fixture(scope="module")
def plan(design):
    return _plan(design)


@pytest.fixture(scope="module")
def labels(plan):
    return rasterise(plan, SHAPE)


@pytest.fixture(scope="module")
def ideal(plan, design):
    return _ideal(plan, design)


def test_render_is_deterministic(design):
    a, b = render_site(design, _slot(), 1), render_site(design, _slot(), 1)
    assert a.channels.keys() == b.channels.keys()
    for ch in a.channels:
        assert a.channels[ch].tobytes() == b.channels[ch].tobytes()
    assert a.cell_rows == b.cell_rows
    other = render_site(design, _slot(), 2)
    assert a.channels["DNA"].tobytes() != other.channels["DNA"].tobytes()


def test_channels_have_the_right_dtype_and_shape():
    design = _design()
    res = render_site(design, _slot(), 1)
    assert list(res.channels) == ["DNA", "ER", "RNA", "AGP", "Mito"]
    for img in res.channels.values():
        assert img.dtype == np.uint16 and img.shape == SHAPE
    assert res.nuclei_mask.shape == res.cell_mask.shape == SHAPE
    assert res.nuclei_mask.dtype == res.cell_mask.dtype == np.uint32
    ideal = render_ideal(_plan(design), design, stream(1, NS_CHANNEL), SHAPE)
    assert all(a.dtype.kind == "f" and a.shape == (H * SS, W * SS) for a in ideal.values())
    assert [r["label"] for r in res.qc_rows] == ["pass"]
    assert pickle.loads(pickle.dumps(res)).cell_rows == res.cell_rows


def test_dna_signal_sits_in_nuclei(ideal, labels):
    nuclei, cells = labels
    dna = _native(ideal["DNA"])
    assert dna[nuclei > 0].mean() > 5 * dna[cells == 0].mean() + 1e-9
    assert dna[cells == 0].mean() == 0.0


def test_er_brighter_near_nucleus_than_at_edge(ideal, labels):
    nuclei, cells = labels
    er = _native(ideal["ER"])
    cyto = (cells > 0) & (nuclei == 0)
    d_nuc = ndi.distance_transform_edt(nuclei == 0)
    d_edge = ndi.distance_transform_edt(cells > 0)
    near = cyto & (d_nuc <= 4)
    far = cyto & (d_edge <= 3)
    assert er[near].mean() > 1.5 * er[far].mean()
    assert er[nuclei > 0].mean() < er[near].mean()


def test_rna_nucleoli_inside_nuclei(ideal, labels):
    nuclei, _ = labels
    rna = _native(ideal["RNA"])
    top = rna >= np.quantile(rna, 0.997)
    assert (nuclei[top] > 0).mean() > 0.95


def test_nuclear_shrink_lowers_nucleoli_and_rna(design, plan, labels):
    nuclei, _ = labels
    base = _with(plan, nucleoli_count_factor=1.0, rna_intensity_factor=1.0)
    shrunk = _with(plan, nucleoli_count_factor=0.5, rna_intensity_factor=0.7)
    rb, rs = _ideal(base, design), _ideal(shrunk, design)
    n_b = np.mean([c.params["n_nucleoli"] for c in base.cells])
    n_s = np.mean([c.params["n_nucleoli"] for c in shrunk.cells])
    assert n_s < n_b
    assert _native(rs["RNA"])[nuclei > 0].mean() < 0.9 * _native(rb["RNA"])[nuclei > 0].mean()


def test_agp_cortex_and_fibres(design, plan, labels):
    nuclei, cells = labels
    inner = ndi.binary_erosion((cells > 0) & (nuclei == 0), iterations=5) & (ndi.distance_transform_edt(nuclei == 0) > 4)
    assert inner.sum() > 500

    def contrast(p):
        agp = _native(_ideal(p, design)["AGP"])
        high = agp - ndi.gaussian_filter(agp, 4.0)
        return high[inner].std() / agp[inner].mean()

    assert contrast(_with(plan, agp_fibre_factor=0.1)) < 0.6 * contrast(_with(plan, agp_fibre_factor=1.0))
    agp = _native(_ideal(plan, design)["AGP"])
    edge = (cells > 0) & (ndi.distance_transform_edt(cells > 0) <= 2)
    assert agp[edge].mean() > 1.3 * agp[inner].mean()


def _components(img, cells):
    """Threshold Mito inside cells at half the cell's 90th percentile; count components and their mean length."""
    from skimage.measure import label, regionprops
    inside = cells > 0
    binary = (img > 0.5 * np.quantile(img[inside], 0.9)) & inside
    props = regionprops(label(binary, connectivity=2))
    return len(props), float(np.mean([p.axis_major_length for p in props]))


def test_mito_fragment_makes_more_shorter_objects(design, plan, labels):
    _, cells = labels
    n_b, len_b = _components(_native(_ideal(_with(plan, mito_length_factor=1.0, mito_count_factor=1.0), design)["Mito"]), cells)
    n_f, len_f = _components(_native(_ideal(_with(plan, mito_length_factor=0.35, mito_count_factor=2.0), design)["Mito"]), cells)
    assert n_f > n_b
    assert len_f < len_b


def test_chromatin_foci_raise_dna_contrast(design, plan, labels):
    nuclei, _ = labels
    inside = ndi.binary_erosion(nuclei > 0, iterations=2)

    def cv(p):
        dna = _native(_ideal(p, design)["DNA"])
        return dna[inside].std() / dna[inside].mean()

    assert cv(_with(plan, dna_foci_add=8.0)) > 1.4 * cv(_with(plan, dna_foci_add=0.0))


def test_psf_blur_applied_before_noise():
    design = _quiet()
    sigma = 1.6
    channels = [{"name": c.name, "stain": c.stain, "psf_sigma_px": sigma} for c in design.plate.channels]
    design = _quiet(image_effects__background={"dist": "fixed", "value": 100.0}, plate__channels=channels)
    res = render_site(design, _slot(), 1)
    slot = _slot()
    plan = sample_cells(design, slot, 1, stream(design.seed, NS_CELLS, 0, slot.row, slot.col, 1), False)
    ideal = render_ideal(plan, design, stream(design.seed, NS_CHANNEL, 0, slot.row, slot.col, 1), SHAPE)
    assert plan.cells
    for ch, img in res.channels.items():
        expected = ndi.gaussian_filter(downsample(ideal[ch], SS), sigma, mode="nearest") + 100.0
        assert np.array_equal(img, np.clip(np.rint(expected), 0, 65535).astype(np.uint16))


def test_wider_psf_lowers_high_frequency_power():
    def power(sigma):
        channels = [{"name": n, "stain": "s", "psf_sigma_px": sigma} for n in ("DNA", "AGP")]
        d = _quiet("small", plate__channels=channels)
        img = render_site(d, _slot(), 1).channels["DNA"].astype(float)
        img = (img - img.mean()) * np.outer(np.hanning(H), np.hanning(W))  # no wrap-around edges
        spec = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
        yy, xx = np.mgrid[:H, :W]
        r = np.hypot(yy - H // 2, xx - W // 2)
        return spec[r > 0.25 * H].sum() / spec.sum()

    assert power(2.5) < 0.25 * power(0.7)


def test_illumination_darkens_corners():
    from synthdata.generators.particle_plate.design import Illumination
    f = illumination_field(H, W, Illumination(kind="radial", strength=0.2))
    assert f[0, 0] / f[H // 2, W // 2] == pytest.approx(0.8, abs=0.01)
    design = _design("small", base_cell__count={"dist": "fixed", "value": 0},
                     image_effects__noise={"poisson": False, "gaussian_sd": 0.0},
                     image_effects__edge_effect={"enabled": False},
                     image_effects__illumination={"kind": "radial", "strength": 0.2},
                     image_effects__background={"dist": "fixed", "value": 1000.0})
    img = render_site(design, _slot(), 1).channels["DNA"].astype(float)
    assert img[:8, :8].mean() / img[H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4].mean() == pytest.approx(0.8, abs=0.03)


def test_blank_has_no_structure():
    design = _design("small", image_effects__illumination={"kind": "none", "strength": 0.0},
                     image_effects__edge_effect={"enabled": False})
    res = render_site(design, _slot(role="blank"), 1)
    assert not res.nuclei_mask.any() and not res.cell_mask.any()
    assert res.cell_rows == [] and res.nucleus_rows == []
    img = res.channels["DNA"].astype(float)
    assert abs(img.mean() - 200) < 20  # the background level is drawn per image (sd 5)
    assert img.std() < 25  # background sd 5 plus Poisson (sd ~14) plus read noise 8
    assert abs(np.corrcoef(img[:, :-1].ravel(), img[:, 1:].ravel())[0, 1]) < 0.05


def test_two_channel_design_renders_only_its_channels():
    design = _design("small")
    res = render_site(design, _slot(), 1)
    assert list(res.channels) == ["DNA", "AGP"]
    assert list(render_ideal(_plan(design), design, stream(1, NS_CHANNEL), SHAPE)) == ["DNA", "AGP"]
    assert {r["channel"] for r in res.image_rows} == {"DNA", "AGP"}
    assert "intensity_ER" not in res.cell_rows[0]


def test_truth_rows_carry_the_resolved_parameters():
    design = _design("small")
    slot = _slot("chromatin_condense", 0.4, role="test")
    res = render_site(design, slot, 1)
    assert res.cell_rows and res.nucleus_rows
    row = res.cell_rows[0]
    for key in ("plate", "well", "site", "cell_id", "nucleus_ids", "x", "y", "body_diameter", "body_eccentricity",
                "body_orientation", "intensity_DNA", "intensity_AGP", "phenotype", "dose_strength",
                "delta_dna_foci_add"):
        assert key in row
    assert row["delta_dna_foci_add"] == pytest.approx(0.4 * 8)
    assert row["delta_dna_intensity_factor"] == pytest.approx(0.4 * 0.15)
    assert len(res.nucleus_rows) == sum(len(r["nucleus_ids"]) for r in res.cell_rows)
    assert len(res.image_rows) == 2


def test_masks_align_with_the_image():
    design = _quiet("small")
    res = render_site(design, _slot(), 1)
    dna = res.channels["DNA"].astype(float)
    assert dna[res.nuclei_mask > 0].mean() > 5 * dna[res.cell_mask == 0].mean()


def _effect(**kw):
    return [{"gain": {}, "background_shift": {}, **kw}]


def test_gain_scales_signal_not_background():
    bg = {"image_effects__background": {"dist": "fixed", "value": 50.0}}
    plain = render_site(_quiet(**bg), _slot(), 1)
    gained = render_site(_quiet(**bg, plate_effects=_effect(gain={"DNA": 2.0})), _slot(), 1)
    sig0 = plain.channels["DNA"].astype(float) - 50
    sig1 = gained.channels["DNA"].astype(float) - 50
    assert sig0.max() > 100  # there is signal to scale
    assert np.abs(sig1 - 2 * sig0).max() <= 1.0  # doubled up to rounding, background untouched
    assert gained.channels["ER"].tobytes() == plain.channels["ER"].tobytes()  # a channel without a gain is unchanged
    row = {r["channel"]: r for r in gained.image_rows}
    assert row["DNA"]["background"] == 50.0 and row["DNA"]["plate_gain"] == 2.0
    assert row["ER"]["plate_gain"] == 1.0 and row["ER"]["plate_background_shift"] == 0.0
    assert "plate_gain" not in plain.image_rows[0]


def test_shift_raises_the_blank_well_median():
    bg = {"image_effects__background": {"dist": "fixed", "value": 100.0}}
    blank = _slot(role="blank")
    plain = render_site(_quiet(**bg), blank, 1)
    shifted = render_site(_quiet(**bg, plate_effects=_effect(background_shift={"DNA": 40.0})), blank, 1)
    assert np.median(plain.channels["DNA"]) == 100
    assert np.median(shifted.channels["DNA"]) == 140
    assert np.median(shifted.channels["ER"]) == 100
    row = shifted.image_rows[0]
    assert row["background"] == 100.0 and row["plate_background_shift"] == 40.0


def test_count_factor_changes_cells_per_site():
    plain, halved = _quiet(), _quiet(plate_effects=_effect(count_factor=0.5))
    n0 = [len(_plan(plain, site=s).cells) for s in range(1, 5)]
    n1 = [len(_plan(halved, site=s).cells) for s in range(1, 5)]
    assert sum(n1) < 0.7 * sum(n0)
    assert render_site(halved, _slot(), 1).image_rows[0]["n_cells_requested"] < 18
