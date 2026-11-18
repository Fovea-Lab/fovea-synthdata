from PIL import Image
from typer.testing import CliRunner

from synthdata.cli import app
from synthdata.core.provenance import is_complete
from synthdata.generators.particle_plate.design import bundled_designs

runner = CliRunner()
SMALL = str(bundled_designs()["small"])


def test_list():
    r = runner.invoke(app, ["list"])
    assert r.exit_code == 0 and "particle_plate" in r.output


def test_generate_check_and_refuse(tmp_path):
    out = tmp_path / "ds"
    r = runner.invoke(app, ["generate", SMALL, "-o", str(out), "--writer", "tiff", "--workers", "1", "--seed", "1"])
    assert r.exit_code == 0, r.output
    assert is_complete(out) and "64 images" in r.output
    assert "seed: 1" in (out / "design.yaml").read_text()
    r = runner.invoke(app, ["check", str(out)])
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output
    r = runner.invoke(app, ["generate", SMALL, "-o", str(out), "--writer", "tiff", "--workers", "1"])
    assert r.exit_code == 1 and "not empty" in r.output


def test_bad_design_names_the_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(bundled_designs()["small"].read_text().replace("sites_per_well", "sites_per_wel"))
    r = runner.invoke(app, ["generate", str(bad), "-o", str(tmp_path / "o")])
    assert r.exit_code == 2 and "sites_per_wel" in r.output


def test_preview(tmp_path):
    png = tmp_path / "p.png"
    r = runner.invoke(app, ["preview", SMALL, "-o", str(png), "--per-condition", "2", "--tile", "64"])
    assert r.exit_code == 0, r.output
    assert Image.open(png).size == (128, 8 * (64 + 18))


def test_check_on_incomplete_folder(tmp_path):
    r = runner.invoke(app, ["check", str(tmp_path)])
    assert r.exit_code == 1 and "not a complete synthdata dataset" in r.output


def test_negative_seed_is_rejected_before_any_output(tmp_path):
    out = tmp_path / "o"
    r = runner.invoke(app, ["generate", SMALL, "-o", str(out), "--seed", "-1"])
    assert r.exit_code != 0 and not out.exists()
