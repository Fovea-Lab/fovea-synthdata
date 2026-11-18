import sys

import pytest
from streamlit.testing.v1 import AppTest

T = 180


@pytest.fixture(autouse=True)
def _keep_main_module():
    saved = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = saved


def _page():
    from synthdata.app.pages import report_page
    report_page()


def test_no_path_shows_info():
    at = AppTest.from_function(_page, default_timeout=T).run()
    assert not at.exception and len(at.info) >= 1


def test_build_button_creates_report(replicate_truth_dataset, tmp_path):
    import shutil
    ds = tmp_path / "ds"
    shutil.copytree(replicate_truth_dataset, ds)
    at = AppTest.from_function(_page, default_timeout=T)
    at.session_state["dataset_path"] = str(ds)
    at.run()
    at.button(key="report_build").click().run()
    assert not at.exception
    assert (ds / "report" / "report.json").exists()


def test_existing_report_clustering_renders_chart(small_truth_dataset, tmp_path):
    import shutil

    from synthdata.generators.particle_plate.truth_report import truth_inputs
    from synthdata.report import build_report
    ds = tmp_path / "ds"
    shutil.copytree(small_truth_dataset, ds)
    build_report(truth_inputs(ds), ds / "report", overwrite=True)
    at = AppTest.from_function(_page, default_timeout=T)
    at.session_state["dataset_path"] = str(ds)
    at.run()
    assert not at.exception
    at.selectbox(key="report_section").select("clustering").run()
    assert not at.exception and len(at.get("vega_lite_chart")) >= 1


def test_corrupt_report_json_shows_error_with_rebuild(tmp_path):
    ds = tmp_path / "ds"
    (ds / "report").mkdir(parents=True)
    (ds / "report" / "report.json").write_text("{not json")
    at = AppTest.from_function(_page, default_timeout=T)
    at.session_state["dataset_path"] = str(ds)
    at.run()
    assert not at.exception and at.error and "--overwrite" in at.error[0].value
    assert not [b for b in at.button if b.key == "report_rebuild"]  # hidden until confirmed
    at.checkbox(key="report_rebuild_confirm").check().run()
    assert [b for b in at.button if b.key == "report_rebuild"]


def test_paths_outside_report_are_ignored_and_twin_shown(tmp_path):
    import json
    ds = tmp_path / "ds"
    (ds / "report").mkdir(parents=True)
    secret = tmp_path / "secret.vl.json"
    secret.write_text(json.dumps({"mark": "point"}))
    rep = {"metrics": {"ari": 0.9, "twin_flagged": False, "n_wells": 4},
           "sections": [{"key": "s", "title": "S", "caption": "", "tables": [],
                         "figures": [{"name": "evil", "title": "Evil", "spec": "../../secret.vl.json"}]}]}
    (ds / "report" / "report.json").write_text(json.dumps(rep))
    at = AppTest.from_function(_page, default_timeout=T)
    at.session_state["dataset_path"] = str(ds)
    at.run()
    assert not at.exception and not at.get("vega_lite_chart")
    assert any("outside the report folder" in c.value for c in at.caption)
    assert any(m.label == "Twin" and m.value == "indistinguishable" for m in at.metric)
