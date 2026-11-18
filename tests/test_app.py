import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from synthdata.cli import APP_MAIN
from synthdata.generators.particle_plate.design import bundled_designs

SMALL_YAML = bundled_designs()["small"].read_text()
T = 180


@pytest.fixture(autouse=True)
def _keep_main_module():
    """AppTest swaps ``sys.modules['__main__']`` for a temp script; spawned worker pools in later tests re-import it."""
    saved = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = saved


def _design():
    from synthdata.app.pages import design_page
    design_page()


def _preview():
    from synthdata.app.pages import preview_page
    preview_page()


def _generate():
    from synthdata.app.pages import generate_page
    generate_page()


def _inspect():
    from synthdata.app.pages import inspect_page
    inspect_page()


def test_main_navigation_loads():
    at = AppTest.from_file(str(APP_MAIN), default_timeout=T).run()
    assert not at.exception


def test_design_page_valid_default():
    at = AppTest.from_function(_design, default_timeout=T).run()
    assert not at.exception
    assert any("valid" in s.value for s in at.success)


def test_design_page_reports_invalid_yaml():
    at = AppTest.from_function(_design, default_timeout=T)
    at.session_state["design_yaml"] = SMALL_YAML.replace("sites_per_well", "sites_per_wel")
    at.run()
    assert not at.exception and any("sites_per_wel" in e.value for e in at.error)


def test_preview_page_renders():
    at = AppTest.from_function(_preview, default_timeout=T)
    at.session_state["design_yaml"] = SMALL_YAML
    at.run()
    assert not at.exception and len(at.get("image")) >= 1


def test_generate_then_inspect(tmp_path):
    out = tmp_path / "ds"
    at = AppTest.from_function(_generate, default_timeout=T)
    at.session_state["design_yaml"] = SMALL_YAML
    at.run()
    at.text_input(key="gen_out").set_value(str(out))
    at.number_input(key="gen_workers").set_value(1)
    at.selectbox(key="gen_writer").set_value("tiff")
    at.run()
    at.button(key="gen_go").click().run()
    assert not at.exception
    assert (out / "manifest.json").exists()
    at2 = AppTest.from_function(_inspect, default_timeout=T)
    at2.session_state["dataset_path"] = str(out)
    at2.run()
    assert not at2.exception


def _yaml_text_area(at):
    return next(t for t in at.text_area if t.label == "design.yaml")


def test_yaml_tab_follows_form_edit_and_apply_keeps_it():
    at = AppTest.from_function(_design, default_timeout=T)
    at.session_state["design_yaml"] = SMALL_YAML
    at.run()
    at.number_input(key="r0_seed").set_value(4242).run()
    assert not at.exception
    assert "seed: 4242" in _yaml_text_area(at).value
    at.button(key="yaml_apply").click().run()
    assert not at.exception
    assert "seed: 4242" in at.session_state["design_yaml"]


def test_duplicate_twice_keeps_both_copies():
    at = AppTest.from_function(_design, default_timeout=T)
    at.session_state["design_yaml"] = SMALL_YAML.replace("rows: 8, cols: 4", "rows: 16, cols: 4")  # room for extra rows
    at.run()
    at.button(key="r0_control_dup").click().run()
    rev = at.session_state["design_rev"]
    at.button(key=f"r{rev}_control_dup").click().run()
    y = at.session_state["design_yaml"]
    assert "control_copy:" in y and "control_copy2:" in y


def test_inspect_incomplete_dataset(tmp_path):
    from synthdata.core.provenance import write_manifest

    write_manifest(tmp_path, generator="particle_plate", generator_version="0", seed=1,
                   design_yaml="x", writer="tiff")
    at = AppTest.from_function(_inspect, default_timeout=T)
    at.session_state["dataset_path"] = str(tmp_path)
    at.run()
    assert not at.exception and any("is missing" in e.value for e in at.error)
