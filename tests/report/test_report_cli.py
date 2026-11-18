import json

from typer.testing import CliRunner

from synthdata.cli import app

runner = CliRunner()


def test_cli_report_builds(small_truth_dataset, tmp_path):
    r = runner.invoke(app, ["report", str(small_truth_dataset), "-o", str(tmp_path / "r")])
    assert r.exit_code == 0, r.output
    assert "ARI" in r.output and str(tmp_path / "r") in r.output
    assert (tmp_path / "r" / "index.html").exists()
    r = runner.invoke(app, ["report", str(small_truth_dataset), "-o", str(tmp_path / "r")])
    assert r.exit_code == 1
    r = runner.invoke(app, ["report", str(small_truth_dataset), "-o", str(tmp_path / "r"), "--overwrite"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["report", str(small_truth_dataset), "-o", str(tmp_path / "r"), "--sections", "2,replicates"])
    assert r.exit_code == 0, r.output
    assert len(json.loads((tmp_path / "r" / "report.json").read_text())["sections"]) == 7


def test_cli_report_refuses_nonempty(small_truth_dataset, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    r = runner.invoke(app, ["report", str(small_truth_dataset), "-o", str(out), "--overwrite"])
    assert r.exit_code == 1
    assert "not empty and not a report" in r.output
    assert (out / "keep.txt").read_text() == "x"


def test_cli_report_not_a_dataset(tmp_path):
    r = runner.invoke(app, ["report", str(tmp_path)])
    assert r.exit_code == 1


def test_cli_incomplete_dataset(small_truth_dataset, tmp_path):
    import shutil

    d = tmp_path / "ds"
    shutil.copytree(small_truth_dataset, d)
    shutil.rmtree(d / "truth")
    r = runner.invoke(app, ["report", str(d)])
    assert r.exit_code == 1 and "Traceback" not in r.output
    assert not isinstance(r.exception, (FileNotFoundError, KeyError))
