import numpy as np
import pandas as pd
import pytest
import tifffile

from synthdata.core import registry
from synthdata.core.generator import GenerationError
from synthdata.core.provenance import is_complete, read_manifest
from synthdata.generators.particle_plate import ParticlePlateGenerator
from synthdata.generators.particle_plate import generator as gen_module
from synthdata.generators.particle_plate.design import load_design
from synthdata.generators.particle_plate.layout import assign
from synthdata.generators.particle_plate.render import render_image

GEN = ParticlePlateGenerator()


def test_registry_resolves_generator():
    assert registry.get("particle_plate").name == "particle_plate"


def test_generate_tiff_layout_and_truth(small_design, tmp_path):
    calls = []
    m = GEN.generate(small_design, tmp_path / "ds", writer="tiff", progress=lambda i, n: calls.append((i, n)))
    out = tmp_path / "ds"
    assert is_complete(out) and read_manifest(out) == m
    assert calls[-1] == (64, 64) and len(calls) == 64
    assert m.extra == {"n_images": 64, "n_wells": 32, "n_particles": m.extra["n_particles"]}
    platemap = pd.read_csv(out / "platemap.csv")
    assert list(platemap.columns) == ["plate", "well", "row", "col", "condition", "role", "replicate"]
    assert len(platemap) == 32
    assert len(list((out / "images").glob("*.tif"))) == 64 and len(list((out / "masks").glob("*.tif"))) == 64
    slot = assign(small_design)[5]
    res = render_image(small_design, slot, 2)
    stem = f"{slot.plate_name}_{slot.well}_s2"
    assert np.array_equal(tifffile.imread(out / "images" / f"{stem}.tif"), res.image)
    assert np.array_equal(tifffile.imread(out / "masks" / f"{stem}_mask.tif"), res.mask)
    particles = pd.read_parquet(out / "truth" / "particles.parquet")
    images = pd.read_parquet(out / "truth" / "images.parquet")
    wells = pd.read_parquet(out / "truth" / "wells.parquet")
    assert len(particles) == m.extra["n_particles"] == images.n_particles.sum()
    assert len(images) == 64 and len(wells) == 32
    assert (images.illumination_angle == small_design.image_effects.illumination.angle).all()
    w = wells[wells.well == slot.well].iloc[0]
    got = particles[particles.well == slot.well].diameter
    assert w.diameter_mean == pytest.approx(got.mean()) and w.n_particles == len(got)
    assert '"dist": "normal"' in w.dist_diameter or '"dist":"normal"' in w.dist_diameter
    assert (out / "design.yaml").read_text().count("diameter:") == 1 + 8  # base + every condition expanded


def test_same_design_same_bytes(small_design, tmp_path):
    a = GEN.generate(small_design, tmp_path / "a", writer="tiff")
    b = GEN.generate(small_design, tmp_path / "b", writer="tiff")
    assert a.files == b.files


def test_parallel_equals_serial(small_design, tmp_path):
    a = GEN.generate(small_design, tmp_path / "a", writer="tiff", workers=1)
    b = GEN.generate(small_design, tmp_path / "b", writer="tiff", workers=2)
    assert a.files == b.files


def test_changing_one_condition_leaves_other_wells_untouched(small_dict, tmp_path):
    a = GEN.generate(load_design(small_dict), tmp_path / "a", writer="tiff")
    small_dict["conditions"]["large"]["diameter"] = {"dist": "normal", "mean": 30, "sd": 1}
    b = GEN.generate(load_design(small_dict), tmp_path / "b", writer="tiff")
    control = [k for k in a.files if k.startswith("images/plate01_A")]
    large = [k for k in a.files if k.startswith("images/plate01_C")]
    assert control and all(a.files[k] == b.files[k] for k in control)
    assert all(a.files[k] != b.files[k] for k in large)


def test_refuses_non_empty_output(small_design, tmp_path):
    (tmp_path / "x.txt").write_text("keep")
    with pytest.raises(GenerationError, match="not empty"):
        GEN.generate(small_design, tmp_path, writer="tiff")


def test_unknown_writer(small_design, tmp_path):
    with pytest.raises(GenerationError, match="unknown writer 'png'"):
        GEN.generate(small_design, tmp_path / "o", writer="png")


def test_failure_names_the_image_and_writes_no_manifest(small_design, tmp_path, monkeypatch):
    real = gen_module.render_image

    def boom(design, slot, site):
        if slot.well == "C02" and site == 2:
            raise RuntimeError("kaboom")
        return real(design, slot, site)

    monkeypatch.setattr(gen_module, "render_image", boom)
    with pytest.raises(GenerationError, match=r"plate01 C02 site 2: RuntimeError: kaboom"):
        GEN.generate(small_design, tmp_path / "o", writer="tiff")
    assert not is_complete(tmp_path / "o")


def test_preview_and_contact_sheet(small_design):
    pv = GEN.preview(small_design, n_per_condition=2)
    assert list(pv.tiles) == list(small_design.conditions)
    assert all(len(v) == 2 for v in pv.tiles.values())
    sheet = GEN.contact_sheet(pv, tile=128)
    assert sheet.size == (2 * 128, 8 * (128 + 18))


def test_resizing_the_plate_leaves_another_well_unchanged(small_dict):
    def render(d):
        design = load_design(d)
        slot = next(s for s in assign(design) if s.well == "A02")
        return render_image(design, slot, 1)

    a = render(small_dict)
    small_dict["plate"]["format"] = {"rows": 8, "cols": 6}
    b = render(small_dict)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.mask, b.mask)
