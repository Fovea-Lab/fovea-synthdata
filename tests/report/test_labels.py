"""ReportInputs.labels: an optional override of reader-facing names, absent by default."""
from dataclasses import replace

from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report.build import build_report
from synthdata.report.context import ReportContext, column_label
from synthdata.report.contract import inputs_sha256

# Taken from the code before the field existed, on the bundled `small` design in a folder named `small`.
GOLDEN_INPUTS_SHA = "88c945e23f6643e639c4754ae8e5ad18ae3c2b9aefc3cebd327629d7a1c64cdf"


def test_labels_override_and_default_is_unchanged(small_truth_dataset, tmp_path):
    inp = truth_inputs(small_truth_dataset)
    assert inp.labels is None
    assert inputs_sha256(inp) == GOLDEN_INPUTS_SHA
    build_report(inp, tmp_path / "r")
    original = (tmp_path / "r" / "index.html").read_bytes()
    assert b"Objects per site" in original
    # Compare the entire report on the same renderer and numerical stack. The
    # pre-override reader-facing labels below must reproduce the default bytes.
    defaults = {"count_per_site": "Objects per site", "diameter": "Diameter"}
    build_report(replace(inp, labels=defaults), tmp_path / "defaults")
    assert (tmp_path / "defaults" / "index.html").read_bytes() == original

    over = replace(inp, labels={"count_per_site": "Cells per site", "diameter": "Width"})
    assert inputs_sha256(over) == GOLDEN_INPUTS_SHA  # a label is presentation, not data
    ctx = ReportContext.create(over, tmp_path / "x")
    assert ctx.label("count_per_site") == "Cells per site"
    assert ctx.label("diameter") == "Width (px)"        # the unit stays
    assert ctx.label("diameter_median") == "Diameter median (px)"  # derived columns are left alone
    plain = ReportContext.create(inp, tmp_path / "y")
    assert plain.label("count_per_site") == column_label("count_per_site", inp.features) == "Objects per site"
