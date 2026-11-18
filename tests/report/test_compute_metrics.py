import json

import pytest

from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report import build_report, compute_metrics


def test_compute_metrics_equals_report_json(replicate_truth_dataset, tmp_path):
    inp = truth_inputs(replicate_truth_dataset)
    res = build_report(inp, tmp_path / "rep")
    on_disk = json.loads((tmp_path / "rep" / "report.json").read_text())["metrics"]
    cm = compute_metrics(inp)
    assert cm.metrics == on_disk
    assert cm.metrics["ari"] == pytest.approx(res.metrics["ari"], abs=0)
    assert {"condition", "column", "flagged"} <= set(cm.effects.columns)
