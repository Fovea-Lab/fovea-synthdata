import json

import numpy as np
import pytest
import zarr

from synthdata.generators.particle_plate import ParticlePlateGenerator
from synthdata.generators.particle_plate.layout import assign
from synthdata.generators.particle_plate.reader import (
    dataset_writer, iter_images, list_images, read_image, validate_ngff,
)
from synthdata.generators.particle_plate.render import render_image
from synthdata.generators.particle_plate.writers.ome_zarr_hcs import scale_factors

GEN = ParticlePlateGenerator()


@pytest.fixture(scope="module")
def zarr_ds(tmp_path_factory):
    import yaml
    from synthdata.generators.particle_plate.design import bundled_designs, load_design

    design = load_design(yaml.safe_load(bundled_designs()["small"].read_text()))
    out = tmp_path_factory.mktemp("z") / "ds"
    GEN.generate(design, out)  # default writer = ome_zarr
    return design, out


def test_scale_factors():
    assert scale_factors(1024, 1024) == (2, 4)
    assert scale_factors(384, 384) == (2,)
    assert scale_factors(256, 256) == ()


def test_default_writer_is_ome_zarr(zarr_ds):
    _, out = zarr_ds
    assert GEN.default_writer == "ome_zarr" and dataset_writer(out) == "ome_zarr"
    assert (out / "plate01.ome.zarr").is_dir()


def test_plate_validates_against_ngff_0_5(zarr_ds):
    _, out = zarr_ds
    validate_ngff(out / "plate01.ome.zarr")
    attrs = dict(zarr.open_group(str(out / "plate01.ome.zarr"), mode="r").attrs)["ome"]
    assert attrs["version"] == "0.5"
    assert [r["name"] for r in attrs["plate"]["rows"]] == list("ABCDEFGH")
    assert [c["name"] for c in attrs["plate"]["columns"]] == ["1", "2", "3", "4"]
    assert len(attrs["plate"]["wells"]) == 32 and attrs["plate"]["field_count"] == 2


def test_well_metadata_and_pyramid(zarr_ds):
    _, out = zarr_ds
    well = zarr.open_group(str(out / "plate01.ome.zarr" / "C" / "1"), mode="r")
    assert dict(well.attrs)["synthdata"] == {"condition": "large", "role": "test", "replicate": 1}
    img = zarr.open_group(str(out / "plate01.ome.zarr" / "C" / "1" / "0"), mode="r")
    ms = dict(img.attrs)["ome"]["multiscales"][0]
    assert [d["coordinateTransformations"][0]["scale"] for d in ms["datasets"]] == [[0.65, 0.65], [1.3, 1.3]]
    assert dict(img["labels"].attrs)["ome"] == {"version": "0.5", "labels": ["particles"]}


def test_pixels_and_labels_equal_the_render(zarr_ds):
    design, out = zarr_ds
    slot = assign(design)[9]
    res = render_image(design, slot, 2)
    image, mask = read_image(out, slot.plate_name, slot.well, 2)
    assert np.array_equal(image, res.image) and np.array_equal(mask, res.mask)


def test_iter_images_covers_dataset(zarr_ds):
    _, out = zarr_ds
    keys = list_images(out)
    assert len(keys) == 64 and keys[0] == ("plate01", "A01", 1)
    first = next(iter_images(out))
    assert first.image.shape == (384, 384) and first.mask.dtype == np.uint32


def test_zarr_is_array_deterministic(zarr_ds, tmp_path):
    design, out = zarr_ds
    GEN.generate(design, tmp_path / "again")
    for key in list_images(out)[:6]:
        a, b = read_image(out, *key), read_image(tmp_path / "again", *key)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_reader_reads_tiff_datasets(small_design, tmp_path):
    GEN.generate(small_design, tmp_path / "t", writer="tiff")
    assert dataset_writer(tmp_path / "t") == "tiff"
    img, mask = read_image(tmp_path / "t", "plate01", "A01", 1)
    assert img.shape == (384, 384)
