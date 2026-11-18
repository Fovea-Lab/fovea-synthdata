"""The generic QC section: contract, presence, heatmap labels, not-assessed wells, truth wiring."""
import json
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from synthdata.generators.particle_plate import ParticlePlateGenerator
from synthdata.generators.particle_plate.design import bundled_designs, load_design
from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report.build import build_report
from synthdata.report.context import ReportContext
from synthdata.report.contract import ReportError, inputs_sha256, validate
from synthdata.report.sections import QC_SECTION, SECTIONS, qc as qc_section
from synthdata.report.theme import QC_COLOURS


@pytest.fixture(scope="module")
def qc_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("qcds") / "small_qc"
    ParticlePlateGenerator().generate(load_design(bundled_designs()["small_qc"]), out, writer="tiff")
    return out


@pytest.fixture(scope="module")
def qc_report(qc_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("qcr") / "r"
    return build_report(truth_inputs(qc_dataset), out), out


def test_truth_inputs_carry_qc(qc_dataset):
    inp = truth_inputs(qc_dataset)
    truth = pd.read_parquet(qc_dataset / "truth" / "qc.parquet")
    assert inp.qc_title == "Quality control (true defect labels)"
    assert len(inp.qc) == len(inp.wells) and set(inp.qc["label"]) <= {"pass", "warn", "fail"}
    mx = truth.groupby(["plate", "well"])["degree"].max()
    got = inp.qc.set_index(["plate", "well"])["degree"]
    assert (got == mx.reindex(got.index)).all()
    assert set(inp.qc["label"]) != {"pass"}  # small_qc has defects


def test_section_last_and_present_with_qc(qc_report):
    res, out = qc_report
    assert [s.key for s in res.sections] == [k for k, *_ in [*SECTIONS, QC_SECTION]] and res.sections[-1].key == "qc"
    s = res.sections[-1]
    assert s.title == "Quality control (true defect labels)"
    assert [f.name for f in s.figures] == ["qc_plate", "qc_counts"]
    assert [t.name for t in s.tables] == ["qc_wells"]
    assert "amber and red wells need review" in s.caption
    for f in s.figures:
        assert (out / f.png).stat().st_size > 0


def test_heatmap_labels_equal_truth(qc_report, qc_dataset):
    _, out = qc_report
    truth = pd.read_parquet(qc_dataset / "truth" / "qc.parquet").groupby(["plate", "well"])["well_label"].first()
    wells = pd.read_parquet(out / "tables/qc/qc_wells.parquet").set_index(["plate", "well"])["label"]
    assert (wells == truth.reindex(wells.index)).all() and len(wells) == len(truth)
    spec = (out / "figures/qc/qc_plate.vl.json").read_text()
    for lab, col in QC_COLOURS.items():
        assert col in spec and lab in spec


def test_absent_without_qc_and_hash_unchanged(qc_dataset, tmp_path):
    inp = truth_inputs(qc_dataset)
    with_qc = inputs_sha256(inp)
    qc = inp.qc
    inp.qc = None
    res = build_report(inp, tmp_path / "r")
    assert "qc" not in [s.key for s in res.sections] and len(res.sections) == 7
    assert "Quality control" not in (tmp_path / "r" / "index.html").read_text()
    assert inputs_sha256(inp) != with_qc
    inp.qc = qc
    assert inputs_sha256(inp) == with_qc
    with pytest.raises(ReportError, match="needs a qc table"):
        build_report(replace(inp, qc=None), tmp_path / "x", sections=["qc"])


def test_design_without_defect_layer_has_no_qc_section(small_truth_dataset, tmp_path):
    """small has an all-pass truth/qc.parquet but no qc_defects: 7 sections, bytes equal to a forced qc=None."""
    assert (small_truth_dataset / "truth" / "qc.parquet").exists()
    inp = truth_inputs(small_truth_dataset)
    assert inp.qc is None and inp.qc_title is None
    a = build_report(inp, tmp_path / "a")
    b = build_report(replace(truth_inputs(small_truth_dataset), qc=None, qc_title=None), tmp_path / "b")
    assert len(a.sections) == 7 and [s.key for s in a.sections] == [k for k, *_ in SECTIONS]
    fa = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    fb = sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*") if p.is_file())
    assert fa == fb
    for rel in fa:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes(), rel


def test_missing_well_is_not_assessed(qc_dataset, tmp_path):
    inp = truth_inputs(qc_dataset)
    gone = inp.qc.iloc[:3][["plate", "well"]]
    inp.qc = inp.qc.iloc[3:].reset_index(drop=True)
    ctx = ReportContext.create(inp, tmp_path)
    df = qc_section.well_table(ctx)
    assert len(df) == 32
    na = df[df["label"] == "not assessed"]
    assert set(zip(na["plate"], na["well"])) == set(zip(gone["plate"], gone["well"]))
    res = qc_section.build(ctx)
    assert res.figures


@pytest.mark.parametrize("mutate,msg", [
    (lambda q: q.assign(label="bad"), "qc labels"),
    (lambda q: q.drop(columns="label"), "missing required"),
    (lambda q: pd.concat([q.iloc[1:], q.iloc[:1].assign(well="Z99")]), "absent from the wells"),
    (lambda q: pd.concat([q, q.iloc[:1]]), "more than one row"),
    (lambda q: q.assign(degree="x"), "numeric"),
])
def test_validation_errors(qc_dataset, mutate, msg):
    inp = truth_inputs(qc_dataset)
    inp.qc = mutate(inp.qc)
    with pytest.raises(ReportError, match=msg):
        validate(inp)


def test_generic_report_has_no_defect_vocabulary():
    import synthdata.report as pkg
    root = Path(pkg.__file__).parent
    hits = [f"{f.relative_to(root)}:{n}" for f in sorted(root.rglob("*.py"))
            for n, line in enumerate(f.read_text().splitlines(), 1) if re.search(r"aggregat|clump", line, re.I)]
    assert hits == []
