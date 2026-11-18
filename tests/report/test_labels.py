"""ReportInputs.labels: an optional override of reader-facing names, absent by default."""
import hashlib
from dataclasses import replace

from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report.build import build_report
from synthdata.report.context import ReportContext, column_label
from synthdata.report.contract import inputs_sha256

# Taken from the code before the field existed, on the bundled `small` design in a folder named `small`.
GOLDEN_INPUTS_SHA = "88c945e23f6643e639c4754ae8e5ad18ae3c2b9aefc3cebd327629d7a1c64cdf"
GOLDEN_INDEX_SHA = "8f53872a786e8dec4e18bdf07888ef302c95d055db6119f890c3737338d5636f"


def test_labels_override_and_default_is_unchanged(small_truth_dataset, tmp_path):
    inp = truth_inputs(small_truth_dataset)
    assert inp.labels is None
    assert inputs_sha256(inp) == GOLDEN_INPUTS_SHA
    build_report(inp, tmp_path / "r")
    assert hashlib.sha256((tmp_path / "r" / "index.html").read_bytes()).hexdigest() == GOLDEN_INDEX_SHA

    over = replace(inp, labels={"count_per_site": "Cells per site", "diameter": "Width"})
    assert inputs_sha256(over) == GOLDEN_INPUTS_SHA  # a label is presentation, not data
    ctx = ReportContext.create(over, tmp_path / "x")
    assert ctx.label("count_per_site") == "Cells per site"
    assert ctx.label("diameter") == "Width (px)"        # the unit stays
    assert ctx.label("diameter_median") == "Diameter median (px)"  # derived columns are left alone
    plain = ReportContext.create(inp, tmp_path / "y")
    assert plain.label("count_per_site") == column_label("count_per_site", inp.features) == "Objects per site"
