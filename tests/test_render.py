import math

import numpy as np
import pytest
from skimage.measure import regionprops

from synthdata.generators.particle_plate.design import Illumination, Profile, load_design
from synthdata.generators.particle_plate.layout import assign
from synthdata.generators.particle_plate.render import (
    ParticleSet, illumination_field, particle_flags, rasterise, render_image,
)


def one(y, x, d, e, th, inten=1000.0, soft=1.0):
    f = lambda v: np.array([float(v)])
    return ParticleSet(f(y), f(x), f(d), f(e), f(th), f(inten), f(soft))


def cat(*sets):
    return ParticleSet(*[np.concatenate([getattr(s, k) for s in sets]) for k in
                         ("y", "x", "diameter", "eccentricity", "orientation", "intensity", "edge_softness")])


@pytest.mark.parametrize("d", [8.0, 14.0, 24.0])
@pytest.mark.parametrize("e", [0.0, 0.6, 0.85])
@pytest.mark.parametrize("th", [0.0, 40.0, 120.0])
def test_single_particle_geometry_is_recoverable(d, e, th):
    r = rasterise(128, 128, one(63.3, 64.7, d, e, th), Profile())
    p = regionprops(r.mask)[0]
    assert abs(p.equivalent_diameter_area - d) <= 0.35
    if e >= 0.5:
        assert abs(p.eccentricity - e) <= 0.06
        measured = (90 - math.degrees(p.orientation)) % 180
        # d=8: pixelisation of an exact ellipse gives 6.06 deg at (e=0.6, th=120), independent of the renderer
        assert abs((measured - th + 90) % 180 - 90) <= (7 if d == 8 else 6)


def test_signal_integral_matches_area_times_intensity():
    for soft in (0.0, 0.3):
        r = rasterise(96, 96, one(48, 48, 16, 0.5, 30, inten=1000, soft=soft), Profile())
        assert r.signal.sum() == pytest.approx(1000 * math.pi * 8**2, rel=0.03)


def test_half_max_contour_matches_truth_edge():
    r = rasterise(96, 96, one(48, 48, 16, 0.5, 30, inten=1000, soft=1.0), Profile())
    assert (r.signal >= 500).sum() == pytest.approx((r.mask == 1).sum(), rel=0.03)


def test_centre_on_supersample_point_is_full_intensity():
    r = rasterise(40, 40, one(20.125, 20.125, 12, 0.0, 0, soft=1.0), Profile())
    assert r.signal[20, 20] >= 0.97 * 1000


def test_gaussian_profile_peaks_at_centre():
    r = rasterise(64, 64, one(32, 32, 20, 0.0, 0), Profile(kind="gaussian", sigma_frac=0.35))
    assert r.signal[32, 32] > 3 * r.signal[32, 32 + 8]


def test_overlap_assigns_nearer_centre_and_flags_both():
    ps = cat(one(40, 40, 16, 0, 0), one(40, 50, 16, 0, 0))
    r = rasterise(80, 96, ps, Profile())
    assert r.mask[40, 40] == 1 and r.mask[40, 50] == 2
    flags = particle_flags(ps, r, 80, 96)
    assert all(f["touching"] for f in flags)
    assert all(0 < f["overlap_frac"] < 1 for f in flags)


def test_isolated_particle_flags():
    r = rasterise(64, 64, one(32, 32, 10, 0, 0), Profile())
    f = particle_flags(one(32, 32, 10, 0, 0), r, 64, 64)[0]
    assert f == {"clipped": False, "touching": False, "overlap_frac": 0.0, "visible_px": int((r.mask == 1).sum())}


def test_particle_on_border_is_clipped():
    ps = one(30, 1.0, 12, 0, 0)
    r = rasterise(64, 64, ps, Profile())
    assert particle_flags(ps, r, 64, 64)[0]["clipped"] is True


def test_particle_larger_than_image():
    ps = one(16, 16, 60, 0.5, 10)
    r = rasterise(32, 32, ps, Profile())
    f = particle_flags(ps, r, 32, 32)[0]
    assert f["clipped"] is True and f["visible_px"] > 0


def test_illumination_field():
    radial = illumination_field(101, 101, Illumination(kind="radial", strength=0.2))
    assert radial[50, 50] == pytest.approx(1.0) and radial[0, 0] == pytest.approx(0.8)
    assert np.all(illumination_field(10, 10, Illumination(kind="none")) == 1)
    lin = illumination_field(10, 20, Illumination(kind="linear", strength=0.5, angle=0))
    assert lin[:, 0].mean() == pytest.approx(1.0) and lin[:, -1].mean() == pytest.approx(0.5)


def test_render_image_is_deterministic_and_site_dependent(small_design):
    slot = assign(small_design)[0]
    a, b = render_image(small_design, slot, 1), render_image(small_design, slot, 1)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.mask, b.mask) and a.particles == b.particles
    assert not np.array_equal(a.image, render_image(small_design, slot, 2).image)
    assert a.image.dtype == np.uint16 and a.mask.dtype == np.uint32
    assert a.image_row["n_particles"] == len(a.particles)
    assert {p["particle_id"] for p in a.particles} == set(range(1, len(a.particles) + 1))


def test_zero_particles(small_dict):
    small_dict["conditions"]["large"]["count"] = {"dist": "fixed", "value": 0}
    d = load_design(small_dict)
    slot = next(s for s in assign(d) if s.condition == "large")
    res = render_image(d, slot, 1)
    assert res.particles == [] and res.mask.max() == 0 and res.image_row["n_particles"] == 0


def test_saturation_is_clipped_and_counted(small_dict):
    small_dict["base_particle"]["intensity"] = {"dist": "fixed", "value": 1e6}
    d = load_design(small_dict)
    res = render_image(d, assign(d)[0], 1)
    assert res.image.max() == 65535 and res.image_row["clipped_px"] > 0


def test_edge_effect_only_on_outer_wells(small_dict):
    small_dict["image_effects"]["edge_effect"] = {"enabled": True, "dim": 0.5}
    d = load_design(small_dict)
    slots = assign(d)
    outer = next(s for s in slots if s.is_outer(8, 4))
    inner = next(s for s in slots if not s.is_outer(8, 4))
    assert render_image(d, outer, 1).image_row["edge_effect"] is True
    assert render_image(d, inner, 1).image_row["edge_effect"] is False


def test_clumped_is_tighter_than_uniform(small_dict):
    from scipy.spatial import cKDTree

    def mean_nn(arrangement):
        small_dict["base_particle"]["arrangement"] = arrangement
        small_dict["base_particle"]["count"] = {"dist": "fixed", "value": 60}
        d = load_design(small_dict)
        res = render_image(d, assign(d)[0], 1)
        pts = np.array([[p["y"], p["x"]] for p in res.particles])
        return cKDTree(pts).query(pts, k=2)[0][:, 1].mean()

    assert mean_nn({"kind": "clumped", "clusters": 2, "spread": 15}) < 0.6 * mean_nn({"kind": "uniform"})
