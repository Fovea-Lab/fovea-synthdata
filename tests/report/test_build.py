import json
import os
import re
from pathlib import Path

import pytest

from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report import ReportError
from synthdata.report.build import build_report
from synthdata.report.sections import SECTIONS


@pytest.fixture(scope="module")
def built(small_truth_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("b") / "r"
    return build_report(truth_inputs(small_truth_dataset), out), out


@pytest.fixture(scope="module")
def built_rep(replicate_truth_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("br") / "r"
    return build_report(truth_inputs(replicate_truth_dataset), out), out


def test_full_build_outputs(built):
    res, out = built
    assert (out / "index.html").is_file() and (out / "report.json").is_file()
    assert sum(len(s.figures) for s in res.sections) >= 15
    rj = json.loads((out / "report.json").read_text())
    assert [s["key"] for s in rj["sections"]] == [k for k, *_ in SECTIONS]
    assert {"ari", "nmi", "k", "map", "flagged", "twin_flagged", "n_wells", "n_objects"} <= set(rj["metrics"])
    assert rj["synthdata_version"] and rj["source_manifest_sha256"]


def test_index_self_contained(built):
    _, out = built
    html = (out / "index.html").read_text()
    assert not re.search(r"<script[^>]*\ssrc=", html)
    assert 'src="http' not in html and 'href="http' not in html
    # the only <link> is the inline favicon; nothing is fetched
    assert all(h.startswith("data:") for h in re.findall(r'<link[^>]*href="([^"]*)"', html))
    for key, *_ in SECTIONS:
        assert f'id="{key}"' in html
    assert "data:image/png;base64," in html
    assert 'rel="icon" href="data:' in html  # no favicon request when served
    assert "<h3>" not in html  # a chart's own title is its only heading
    assert 'class="grid multiples"' in html  # heatmaps laid out as small multiples
    assert "vegaEmbed" in html and "twin:" in html


def test_report_json_valid_and_significance(built_rep):
    res, out = built_rep
    text = (out / "report.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    m = json.loads(text)["metrics"]
    assert "diameter_median" in m["flagged"]["large"]
    assert m["twin_flagged"] is False
    assert m["ari"] >= 0.6
    assert res.metrics["ari"] == m["ari"]


def test_determinism(small_truth_dataset, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build_report(truth_inputs(small_truth_dataset), a)
    build_report(truth_inputs(small_truth_dataset), b)
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert fa == fb and len(fa) > 30
    for f in fa:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_sections_rebuild_only_that_section(small_truth_dataset, tmp_path):
    out = tmp_path / "r"
    build_report(truth_inputs(small_truth_dataset), out)
    files = [p for p in out.rglob("*") if p.is_file()]
    for p in files:
        os.utime(p, (1_000_000, 1_000_000))
    build_report(truth_inputs(small_truth_dataset), out, sections=["replicates"])
    for p in (q for q in out.rglob("*") if q.is_file()):
        rel = p.relative_to(out)
        changed = p.stat().st_mtime > 2_000_000
        expected = rel.name in ("index.html", "report.json") or rel.parts[1] == "replicates"
        assert changed == expected, rel
    rj = json.loads((out / "report.json").read_text())
    assert all(s["figures"] for s in rj["sections"] if s["key"] == "overview")
    assert len(rj["sections"]) == 7


def test_partial_rebuild_removes_stale_files(small_truth_dataset, tmp_path):
    out = tmp_path / "r"
    build_report(truth_inputs(small_truth_dataset), out)
    stale = out / "figures" / "replicates" / "stale.svg"
    stale.write_text("x")
    build_report(truth_inputs(small_truth_dataset), out, sections=["replicates"])
    assert not stale.exists()


def test_partial_rebuild_refused_for_other_dataset(small_truth_dataset, replicate_truth_dataset, tmp_path):
    out = tmp_path / "r"
    build_report(truth_inputs(small_truth_dataset), out, sections=["overview"])
    before = (out / "report.json").read_bytes()
    with pytest.raises(ReportError, match="different dataset"):
        build_report(truth_inputs(replicate_truth_dataset), out, sections=["overview"])
    assert (out / "report.json").read_bytes() == before


def test_overwrite_keeps_foreign_files(small_truth_dataset, tmp_path):
    out = tmp_path / "r"
    (out / "data").mkdir(parents=True)
    (out / "report.json").write_text("{}")
    (out / "notes.txt").write_text("keep")
    (out / "data" / "x.csv").write_text("1")
    build_report(truth_inputs(small_truth_dataset), out, overwrite=True)
    assert (out / "notes.txt").read_text() == "keep" and (out / "data" / "x.csv").exists()
    assert (out / "index.html").exists() and "sections" in json.loads((out / "report.json").read_text())


def test_failed_build_leaves_old_report(small_truth_dataset, tmp_path, monkeypatch):
    out = tmp_path / "r"
    build_report(truth_inputs(small_truth_dataset), out)
    before = (out / "index.html").read_bytes()
    inp = truth_inputs(small_truth_dataset)
    inp.reference = "nope"
    with pytest.raises(ReportError):
        build_report(inp, out, overwrite=True)
    assert (out / "index.html").read_bytes() == before


def test_out_is_a_file_and_corrupt_report(small_truth_dataset, tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(ReportError, match="not a folder"):
        build_report(truth_inputs(small_truth_dataset), f)
    out = tmp_path / "c"
    out.mkdir()
    (out / "report.json").write_text("not json")
    with pytest.raises(ReportError, match="not a readable report"):
        build_report(truth_inputs(small_truth_dataset), out, sections=["overview"])
    (out / "report.json").write_text('{"a": 1}')
    with pytest.raises(ReportError, match="not a readable report"):
        build_report(truth_inputs(small_truth_dataset), out, sections=["overview"])


def test_source_is_path_independent(small_truth_dataset, tmp_path):
    import shutil

    outs = []
    for i in (1, 2):
        d = tmp_path / f"p{i}" / "ds"
        shutil.copytree(small_truth_dataset, d)
        outs.append(tmp_path / f"o{i}")
        build_report(truth_inputs(d), outs[-1], sections=["overview", "heatmaps"])
    for name in ("report.json", "index.html"):
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()
    assert str(tmp_path) not in (outs[0] / "report.json").read_text()


def test_section_exception_becomes_note(small_truth_dataset, tmp_path, monkeypatch):
    from synthdata.report import build as b

    def boom(ctx):
        raise RuntimeError("kaput")

    monkeypatch.setattr(b, "SECTIONS", [(k, t, c, boom if k == "heatmaps" else fn) for k, t, c, fn in SECTIONS])
    res = build_report(truth_inputs(small_truth_dataset), tmp_path / "r")
    bad = next(s for s in res.sections if s.key == "heatmaps")
    assert bad.notes == ["heatmaps failed: RuntimeError: kaput"]
    assert all(s.figures for s in res.sections if s.key != "heatmaps")
    assert len(res.sections) == 7


def test_metrics_failure_is_guarded(small_truth_dataset, tmp_path, monkeypatch):
    from synthdata.report import build as b

    monkeypatch.setattr(b, "_metrics", lambda ctx: 1 / 0)
    res = build_report(truth_inputs(small_truth_dataset), tmp_path / "r")
    assert res.metrics["ari"] is None and any("metrics failed" in n for n in res.notes)
    assert (tmp_path / "r" / "index.html").exists() and (tmp_path / "r" / "report.json").exists()


def test_script_close_tag_escaped():
    from synthdata.report.html import _safe_json

    out = _safe_json({"title": "a </script><script>alert(1)</script> <!-- x"})
    assert "</" not in out and "<!--" not in out
    import json as j
    assert j.loads(out)["title"].startswith("a </script>")


def test_refuses_foreign_and_overwrite(small_truth_dataset, tmp_path):
    out = tmp_path / "r"
    out.mkdir()
    (out / "mine.txt").write_text("x")
    with pytest.raises(ReportError, match="not empty and not a report"):
        build_report(truth_inputs(small_truth_dataset), out)
    assert (out / "mine.txt").exists()
    out2 = tmp_path / "r2"
    build_report(truth_inputs(small_truth_dataset), out2, sections=["overview"])
    assert (out2 / "report.json").exists()
    with pytest.raises(ReportError, match="already contains a report"):
        build_report(truth_inputs(small_truth_dataset), out2)
    build_report(truth_inputs(small_truth_dataset), out2, overwrite=True)


@pytest.mark.parametrize("kind", ["three_wells", "one_condition"])
def test_report_builds_on_tiny_input(small_truth_dataset, tmp_path, kind):
    inp = truth_inputs(small_truth_dataset)
    w = inp.wells
    if kind == "three_wells":
        keep = list(w[w["condition"] == inp.reference].index[:2]) + list(w[w["condition"] != inp.reference].index[:1])
    else:
        keep = list(w[w["condition"] == inp.reference].index)
    w = w.loc[keep]
    inp.wells = w.reset_index(drop=True)
    pw = set(zip(w["plate"], w["well"]))
    o = inp.objects
    inp.objects = o[[(a, b) in pw for a, b in zip(o["plate"], o["well"])]].reset_index(drop=True)
    inp.images = {c: v for c, v in (inp.images or {}).items() if c in set(w["condition"])}
    res = build_report(inp, tmp_path / "t")
    assert (tmp_path / "t" / "index.html").exists()
    assert len(res.sections) == 7
    assert any(s.notes for s in res.sections) or res.notes
    m = json.loads((tmp_path / "t" / "report.json").read_text())["metrics"]
    if kind == "one_condition":
        assert m["ari"] is None
