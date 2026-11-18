import json

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner
from jsonschema import Draft202012Validator

from synthdata import sdk
from synthdata.cli import app


@pytest.mark.parametrize("name", ["particle_plate", "cell_plate"])
def test_public_catalog_and_editable_preset(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    catalog = sdk.describe_generator(name)
    assert catalog["default_writer"] in catalog["writers"]
    assert "plate" in catalog["settings_schema"]["properties"]
    assert "check" in catalog["capabilities"]
    assert ("report" in catalog["capabilities"]) == (name == "particle_plate")
    preset = f"{name}/small"
    assert preset in catalog["presets"]

    cli = CliRunner()
    assert json.loads(cli.invoke(app, ["describe", name]).stdout) == catalog
    assert cli.invoke(app, ["preset", "list", name]).stdout.splitlines() == catalog["presets"]
    out = tmp_path / "mine.yaml"
    result = cli.invoke(app, ["preset", "export", preset, "-o", str(out)])
    assert result.exit_code == 0, result.output
    settings = sdk.load_preset(preset)
    assert yaml.safe_load(out.read_text()) == settings
    settings["seed"] += 21
    assert sdk.validate_design(settings).seed == settings["seed"]
    assert sdk.load_preset(preset)["seed"] != settings["seed"]
    out.write_text("my private changes")
    assert cli.invoke(app, ["preset", "export", preset, "-o", str(out)]).exit_code != 0
    assert out.read_text() == "my private changes"


def test_validation_precedes_writes(tmp_path):
    design = sdk.load_preset("particle_plate/small")
    design["plate"]["sites_per_well"] = 0
    out = tmp_path / "bad"
    with pytest.raises(ValidationError, match="sites_per_well"):
        sdk.generate(design, out)
    assert not out.exists()


def test_bundled_designs_conform_to_the_published_settings_schema():
    for name in sdk.list_generators():
        validator = Draft202012Validator(sdk.describe_generator(name)["settings_schema"])
        for preset in sdk.list_presets(name):
            validator.validate(sdk.load_preset(preset))


@pytest.mark.parametrize("preset", ["../docs/research", "particle_plate/../default", "small", "cell_plate/nope"])
def test_only_bundled_presets_are_read(preset):
    with pytest.raises((KeyError, ValueError)):
        sdk.load_preset(preset)


def test_tiff_channel_names_must_be_filename_components():
    design = sdk.load_preset("cell_plate/small")
    design["plate"]["channels"][0]["name"] = "a/b"
    with pytest.raises(ValidationError, match="channels.0.name"):
        sdk.validate_design(design)


def test_cell_preview_shows_each_requested_site():
    design = sdk.load_preset("cell_plate/small")
    design.update(compounds={"DMSO": {"role": "negative_control"}}, doses={}, expectations=None,
                  layout={"fixed": {"A01": "DMSO"}}, qc_defects=None)
    design["plate"].update(format={"rows": 1, "cols": 1}, sites_per_well=2)
    image = sdk.preview(design, per_condition=2, tile=32)
    assert image.size == (2 * len(design["plate"]["channels"]) * 32, 32 + 18)
