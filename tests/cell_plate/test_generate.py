import re

import numpy as np
import tifffile
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from synthdata.cli import app
from synthdata.core.provenance import read_manifest
from synthdata.generators.cell_plate.design import bundled_designs, load_design
from synthdata.generators.cell_plate.generator import CellPlateGenerator
from synthdata.generators.cell_plate.reader import (
    list_sites, read_acquisition, read_platemap, read_site, read_truth_masks)
from synthdata.generators.particle_plate.reader import validate_ngff

pytestmark = pytest.mark.filterwarnings("ignore:cell_plate:UserWarning")  # the small design is dense at a few sites

gen = CellPlateGenerator()
small = load_design(bundled_designs()["small"].read_text())
TRUTH = ["truth/cells.parquet", "truth/nuclei.parquet", "truth/images.parquet", "truth/wells.parquet",
         "truth/qc.parquet"]


@pytest.fixture(scope="module")
def tiff_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiff")
    gen.generate(small, out, writer="tiff")
    return out


@pytest.fixture(scope="module")
def zarr_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("zarr")
    gen.generate(small, out, writer="ome_zarr")
    return out


def test_small_plate_outputs(tiff_out):
    for f in ["platemap.csv", "images.csv", "acquisition.json", "design.yaml", "manifest.json", *TRUTH]:
        assert (tiff_out / f).is_file()
    pm = read_platemap(tiff_out)
    assert list(pm.columns) == ["plate", "well_position", "broad_sample", "solvent", "concentration", "control_type"]
    assert len(pm) == 96
    dmso = pm.broad_sample == "DMSO"
    assert dmso.sum() == 8
    assert pm.loc[dmso, "control_type"].eq("negcon").all()
    assert pm.loc[dmso, "concentration"].eq("").all()
    assert pm.loc[pm.broad_sample == "POS-CTRL", "concentration"].eq("10 uM").all()
    assert pm.loc[pm.broad_sample == "POS-CTRL", "control_type"].eq("poscon").all()
    blank = pm.broad_sample == "BLANK"
    assert pm.loc[blank, "solvent"].eq("").all() and pm.loc[blank, "control_type"].eq("blank").all()
    cpd = pm.broad_sample.str.startswith("CPD-")
    assert pm.loc[cpd, "control_type"].eq("").all() and pm.loc[cpd, "solvent"].eq("DMSO").all()
    assert pm.loc[cpd, "concentration"].isin(["1 uM", "10 uM"]).all()
    assert len(list_sites(tiff_out)) == 96
    assert read_acquisition(tiff_out)["channels"][0]["name"] == "DNA"


def test_truth_and_pixels_are_separate_files(tiff_out):
    assert len(list((tiff_out / "images").glob("*.tif"))) == 96 * 2
    assert len(list((tiff_out / "truth" / "labels").glob("*.tif"))) == 96 * 2
    assert not any("truth" in p.name or "nuclei" in p.name for p in (tiff_out / "images").iterdir())
    for p in ["platemap.csv", "images.csv", "acquisition.json"]:
        assert str(tiff_out) not in (tiff_out / p).read_text()
    paths = [l.split(",")[-1] for l in (tiff_out / "images.csv").read_text().splitlines()[1:]]
    assert all(not p.startswith("/") for p in paths)
    assert (tiff_out / paths[0]).is_file()


def test_truth_wells_and_images_columns(tiff_out):
    w = pq.read_table(tiff_out / "truth" / "wells.parquet").to_pandas()
    for c in ["plate", "well", "row", "col", "condition", "role", "replicate", "Metadata_Plate", "Metadata_Well",
              "Metadata_broad_sample", "Metadata_control_type", "Metadata_concentration", "phenotype", "strength"]:
        assert c in w.columns
    assert "DMSO" in set(w.condition) and "BLANK" in set(w.condition) and len(set(w.condition)) == 21
    im = pq.read_table(tiff_out / "truth" / "images.parquet").to_pandas()
    assert {"path", "illumination_factor", "channel", "psf_sigma_px", "background"} <= set(im.columns)
    assert len(im) == 96 * 2
    assert (im.illumination_factor.between(0.5, 1.0)).all()
    assert (tiff_out / im.path.iloc[0]).is_file()


def test_phenotype_is_blinded(tiff_out, zarr_out):
    names = list(small.phenotypes)
    for root in (tiff_out, zarr_out):
        texts = [(root / f).read_text() for f in ["platemap.csv", "images.csv", "acquisition.json"]]
        texts += [p.read_text() for p in root.rglob("*") if p.is_file() and p.name in ("zarr.json", ".zattrs")]
        assert len(texts) > 3 or root == tiff_out
        for t in texts:
            for n in names:
                assert not re.search(rf"\b{n}\b", t), n
            assert "strength" not in t and "phenotype" not in t


def test_byte_reproducible_and_worker_independent(tmp_path, zarr_out):
    m2 = gen.generate(small, tmp_path / "w2", writer="ome_zarr", workers=2)
    m1 = read_manifest(zarr_out)
    assert m1.files == m2.files


def test_writers_agree(tiff_out, zarr_out):
    sites = list_sites(tiff_out)
    assert sites == list_sites(zarr_out)
    for plate, well, site in [sites[0], sites[40], sites[-1]]:
        a, b = read_site(tiff_out, plate, well, site), read_site(zarr_out, plate, well, site)
        assert list(a) == list(b) == ["DNA", "AGP"]
        for ch in a:
            assert a[ch].dtype == np.uint16 and a[ch].shape == (384, 384)
            assert np.array_equal(a[ch], b[ch])
        for x, y in zip(read_truth_masks(tiff_out, plate, well, site), read_truth_masks(zarr_out, plate, well, site)):
            assert np.array_equal(x, y)


def test_ngff_valid(zarr_out):
    validate_ngff(zarr_out / "plate01.ome.zarr")
    import zarr
    ig = zarr.open_group(str(zarr_out / "plate01.ome.zarr" / "A" / "1" / "0"), mode="r")
    ome = dict(ig.attrs)["ome"]
    assert [c["label"] for c in ome["omero"]["channels"]] == ["DNA", "AGP"]
    assert [a["name"] for a in ome["multiscales"][0]["axes"]] == ["c", "y", "x"]
    assert {"nuclei", "cells"} <= set(dict(ig["labels"].attrs)["ome"]["labels"])


def test_truth_consistent_with_masks(tiff_out):
    cells = pq.read_table(tiff_out / "truth" / "cells.parquet").to_pandas()
    nuclei = pq.read_table(tiff_out / "truth" / "nuclei.parquet").to_pandas()
    assert len(cells) > 500
    for (plate, well, site), c in list(cells.groupby(["plate", "well", "site"]))[:6]:
        nuc, cel = read_truth_masks(tiff_out, plate, well, site)
        assert set(np.unique(cel)) - {0} == set(c.cell_id)
        n = nuclei[(nuclei.plate == plate) & (nuclei.well == well) & (nuclei.site == site)]
        assert set(np.unique(nuc)) - {0} == set(n.nucleus_id)
        for _, row in c.iterrows():
            assert sorted(row.nucleus_ids) == sorted(n[n.cell_id == row.cell_id].nucleus_id)
    qc = pq.read_table(tiff_out / "truth" / "qc.parquet").to_pandas()
    assert (qc.defect != "none").sum() == 10 and len(qc) == 96


def test_blank_wells_have_no_cells(tiff_out):
    pm = read_platemap(tiff_out)
    w = pq.read_table(tiff_out / "truth" / "cells.parquet").to_pandas()
    blanks = set(pm.loc[pm.broad_sample == "BLANK", "well_position"])
    assert blanks and not (set(w.well) & blanks)


def test_report_shortfall_in_manifest(tmp_path, tiff_out):
    m = read_manifest(tiff_out)
    assert m.extra["n_cells"] > 500 and m.extra["n_cells_not_placed"] >= 0
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate"]["format"] = {"rows": 1, "cols": 2}
    data["layout"] = {"kind": "controlled_random", "fixed": {"A01": "DMSO", "A02": "POS-CTRL"}}
    data["doses"] = {}
    data["compounds"] = {k: v for k, v in data["compounds"].items() if k in ("DMSO", "POS-CTRL")}
    data["doses"] = {"high": {"label": "10 uM", "strength": 1.0}}
    data["base_cell"]["count"] = {"dist": "fixed", "value": 300}
    data["plate"]["image"] = {"height": 128, "width": 128, "dtype": "uint16", "pixel_size_um": 0.65}
    data["qc_defects"] = None
    data.pop("expectations")
    with pytest.warns(UserWarning):
        m = gen.generate(load_design(data), tmp_path, writer="tiff")
    assert m.extra["n_cells_not_placed"] > 0
    assert m.extra["n_cells"] + m.extra["n_cells_not_placed"] > 300


def test_cli_generate_and_preview(tmp_path):
    r = CliRunner()
    design = str(bundled_designs()["small"])
    res = r.invoke(app, ["generate", design, "-o", str(tmp_path / "d"), "--writer", "tiff", "--workers", "2"])
    assert res.exit_code == 0, res.output
    assert "cells" in res.output and "KeyError" not in res.output
    assert (tmp_path / "d" / "manifest.json").is_file()
    res = r.invoke(app, ["preview", design, "-o", str(tmp_path / "p.png"), "--tile", "64"])
    assert res.exit_code == 0, res.output
    from PIL import Image
    w, h = Image.open(tmp_path / "p.png").size
    assert w == 2 * 64 and h > 21 * 64


def test_two_plate_dataset_reads_back_and_checks(tmp_path):
    from synthdata.generators.cell_plate.check import check_dataset

    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate"]["plates"] = 2
    data["plate"]["image"] = {"height": 192, "width": 192, "dtype": "uint16", "pixel_size_um": 0.65}
    data["base_cell"]["count"] = {"dist": "poisson", "mean": 6}
    data["qc_defects"] = {"blurred_sites": {"n": 3, "psf_sigma_px": 3.5}, "saturated_debris": {"n": 2}}
    data.pop("expectations")
    out = tmp_path / "two"
    gen.generate(load_design(data), out, writer="tiff")
    sites = list_sites(out)
    assert {p for p, _, _ in sites} == {"plate01", "plate02"} and len(sites) == 192
    for plate in ("plate01", "plate02"):
        well, site = next((w, s) for p, w, s in sites if p == plate)
        assert set(read_site(out, plate, well, site)) == {"DNA", "AGP"}
    report = check_dataset(out)
    # Stage 1 only: the helper has no expectations and tiny images, so stage 2 is not asserted here.
    assert report.stage1 and all(c.passed for c in report.stage1), [c for c in report.stage1 if not c.passed]


def _mech_design(mechanism):
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    if mechanism:
        data["compounds"]["CPD-002"]["mechanism"] = mechanism
        data["compounds"]["CPD-002"]["potency"] = 0.5
    return load_design(data)


def test_platemap_has_moa_only_when_a_mechanism_exists(tmp_path):
    import csv

    from synthdata.generators.cell_plate.files import PLATEMAP_COLUMNS, write_platemap
    from synthdata.generators.cell_plate.layout import assign

    plain = tmp_path / "plain"
    plain.mkdir()
    write_platemap(plain, assign(_mech_design(None)))
    assert (plain / "platemap.csv").read_text().splitlines()[0] == ",".join(PLATEMAP_COLUMNS)
    assert "moa" not in (plain / "platemap.csv").read_text()

    withm = tmp_path / "with"
    withm.mkdir()
    write_platemap(withm, assign(_mech_design("tubulin")))
    rows = list(csv.DictReader((withm / "platemap.csv").open()))
    assert list(rows[0]) == [*PLATEMAP_COLUMNS, "moa"]
    by = {}
    for r in rows:
        by.setdefault(r["broad_sample"], set()).add(r["moa"])
    assert by["CPD-002"] == {"tubulin"}
    assert by["DMSO"] == {""} and by["CPD-001"] == {""} and by["BLANK"] == {""}


def test_truth_wells_gain_mechanism_columns_only_with_a_mechanism(tmp_path):
    from synthdata.generators.cell_plate.layout import assign
    from synthdata.generators.cell_plate.truth import wells_rows

    assert not {"mechanism", "potency", "side_effect", "side_strength"} & set(wells_rows(assign(_mech_design(None)))[0])
    rows = wells_rows(assign(_mech_design("tubulin")))
    assert all({"mechanism", "potency", "side_effect", "side_strength"} <= set(r) for r in rows)
    got = {r["mechanism"] for r in rows if r["Metadata_broad_sample"] == "CPD-002"}
    assert got == {"tubulin"}
    assert {r["potency"] for r in rows if r["Metadata_broad_sample"] == "CPD-002"} == {0.5}
    assert {r["mechanism"] for r in rows if r["Metadata_broad_sample"] != "CPD-002"} == {""}


def _effects_design(effects):
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate"]["plates"] = 2
    data["plate"]["image"] = {"height": 96, "width": 96, "dtype": "uint16", "pixel_size_um": 0.65}
    data["base_cell"]["count"] = {"dist": "poisson", "mean": 3}
    data.pop("expectations")
    if effects is not None:
        data["plate_effects"] = effects
    return load_design(data)


def test_plate_effects_truth_is_written_only_when_effects_are_set(tmp_path):
    effects = [{"gain": {"DNA": 1.5}, "background_shift": {}, "count_factor": 1.0},
               {"gain": {}, "background_shift": {"AGP": 20.0}, "count_factor": 0.5}]
    with_fx, without = tmp_path / "with", tmp_path / "without"
    gen.generate(_effects_design(effects), with_fx, writer="tiff")
    gen.generate(_effects_design(None), without, writer="tiff")
    rows = pq.read_table(with_fx / "truth" / "plate_effects.parquet").to_pylist()
    by = {(r["plate"], r["channel"]): r for r in rows}
    assert len(rows) == 4 and by[("plate01", "DNA")]["gain"] == 1.5
    assert by[("plate02", "AGP")]["background_shift"] == 20.0 and by[("plate02", "DNA")]["count_factor"] == 0.5
    assert not (without / "truth" / "plate_effects.parquet").exists()
    assert {"plate_gain", "plate_background_shift"} <= set(pq.read_table(with_fx / "truth" / "images.parquet").column_names)
    assert not {"plate_gain"} & set(pq.read_table(without / "truth" / "images.parquet").column_names)


def _five_channel_design():
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate"]["image"] = {"height": 96, "width": 96, "dtype": "uint16", "pixel_size_um": 0.65}
    data["plate"]["channels"] = yaml.safe_load(bundled_designs()["default"].read_text())["plate"]["channels"]
    data["base_cell"]["channels"] = yaml.safe_load(bundled_designs()["default"].read_text())["base_cell"]["channels"]
    data["base_cell"]["count"] = {"dist": "poisson", "mean": 3}
    data.pop("expectations")
    return load_design(data)


def _gen_five(out, writer):
    gen.generate(_five_channel_design(), out, writer=writer)
    return out


def test_stack_holds_the_same_pixels_as_the_zarr_writer(tmp_path):
    z, t = _gen_five(tmp_path / "z", "ome_zarr"), _gen_five(tmp_path / "t", "tiff_stack")
    sites = list_sites(t)
    assert sites == list_sites(z)
    for plate, well, site in [sites[0], sites[len(sites) // 2], sites[-1]]:
        a, b = read_site(z, plate, well, site), read_site(t, plate, well, site)
        assert list(a) == list(b) == ["DNA", "ER", "RNA", "AGP", "Mito"]
        for ch in a:
            assert np.array_equal(a[ch], b[ch]) and b[ch].dtype == np.uint16
        for x, y in zip(read_truth_masks(z, plate, well, site), read_truth_masks(t, plate, well, site)):
            assert np.array_equal(x, y)


def test_stack_shape_is_channels_first(tmp_path):
    out = _gen_five(tmp_path, "tiff_stack")
    files = sorted((out / "images").glob("*.tif"))
    assert files and len(files) == len(list_sites(out))
    arr = tifffile.imread(files[0])
    assert arr.shape[0] == 5 and arr.ndim == 3
    with tifffile.TiffFile(files[0]) as tf:
        assert tf.series[0].axes == "CYX"


def test_stack_images_csv_lists_one_row_per_channel_with_the_shared_path(tmp_path):
    import csv
    out = _gen_five(tmp_path, "tiff_stack")
    rows = list(csv.DictReader((out / "images.csv").open()))
    by_site = {}
    for r in rows:
        by_site.setdefault((r["plate"], r["well"], r["site"]), []).append(r)
    for rs in by_site.values():
        assert [r["channel"] for r in rs] == ["DNA", "ER", "RNA", "AGP", "Mito"]
        assert len({r["path"] for r in rs}) == 1 and rs[0]["path"].endswith(".tif")
        assert (out / rs[0]["path"]).is_file()
