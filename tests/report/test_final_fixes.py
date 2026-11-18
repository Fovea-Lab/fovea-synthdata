"""Final-review fix wave: generic contract, honest twin claim, safe filenames, stamps, captions."""
import copy
import json
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report import ReportError
from synthdata.report.build import build_report
from synthdata.report.context import ReportContext
from synthdata.report.sections import distributions, fingerprints

PP_NAMES = ["control_twin", "two_sizes", "dim_peaked", "elongated", "aligned", "crowded", "large"]


@pytest.fixture(scope="module")
def built_small(small_truth_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("fs") / "r"
    return build_report(truth_inputs(small_truth_dataset), out), out


@pytest.fixture(scope="module")
def built_rep(replicate_truth_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("fr") / "r"
    return build_report(truth_inputs(replicate_truth_dataset), out), out


def _rename(inp, mapping):
    f = lambda s: s.map(lambda c: mapping.get(str(c), str(c)))  # noqa: E731
    w = inp.wells.copy()
    w["condition"] = f(w["condition"])
    return replace(inp, wells=w, reference=mapping.get(inp.reference, inp.reference),
                   intended=None, images={mapping.get(c, c): a for c, a in (inp.images or {}).items()})


# ---- I1 ----
def test_twin_claim_not_testable_on_small(built_small):
    res, out = built_small
    m = res.metrics
    assert m["twin_testable"] is False
    html = (out / "index.html").read_text()
    assert "✓" not in html and "not testable" in html


def test_twin_claim_ok_when_testable(built_rep):
    res, out = built_rep
    assert res.metrics["twin_testable"] is True and res.metrics["twin_flagged"] is False
    assert "twin indistinguishable ✓" in (out / "index.html").read_text()


def test_twin_none_without_twin(small_truth_dataset, tmp_path):
    inp = truth_inputs(small_truth_dataset)
    keep = inp.wells["condition"] != "control_twin"
    w = inp.wells[keep].reset_index(drop=True)
    pw = set(zip(w["plate"], w["well"]))
    o = inp.objects
    inp = replace(inp, wells=w, objects=o[[(a, b) in pw for a, b in zip(o["plate"], o["well"])]].reset_index(drop=True))
    res = build_report(inp, tmp_path / "r")
    assert res.metrics["twin_flagged"] is None
    assert "twin: not applicable" in (tmp_path / "r" / "index.html").read_text()


def test_twin_status_wording():
    from synthdata.report.summary import twin_status
    assert twin_status({"twin_flagged": None, "twin_testable": True})[1] == "twin: not applicable"
    assert "not testable" in twin_status({"twin_flagged": False, "twin_testable": False})[1]
    assert twin_status({"twin_flagged": False, "twin_testable": True})[1] == "twin indistinguishable ✓"
    assert "✗" in twin_status({"twin_flagged": True, "twin_testable": True})[1]


# ---- I2 ----
def test_generic_captions_have_no_design_names(small_truth_dataset, tmp_path):
    inp = truth_inputs(small_truth_dataset)
    mapping = {c: f"cond{i}" for i, c in enumerate(inp.wells["condition"].drop_duplicates())}
    inp = replace(_rename(inp, mapping), notes_for_reader=None)
    res = build_report(inp, tmp_path / "r")
    text = " ".join(s.caption for s in res.sections)
    for n in PP_NAMES:
        assert n not in text, n
    assert "intended distribution" not in text and "dark line" not in text


def test_adapter_notes_and_intended_caption(built_rep):
    res, _ = built_rep
    caps = {s.key: s.caption for s in res.sections}
    assert "two_sizes" in caps["objects"] and "dark line" in caps["distributions"]


def test_notes_for_reader_default_none():
    from synthdata.report.contract import ReportInputs
    assert ReportInputs.__dataclass_fields__["notes_for_reader"].default is None


# ---- I3 ----
def test_unsafe_condition_name_builds(small_truth_dataset, tmp_path):
    inp = truth_inputs(small_truth_dataset)
    inp = _rename(inp, {"large": "cpd 3/10uM", "elongated": "../evil"})
    out = tmp_path / "sub" / "r"
    res = build_report(inp, out)
    assert (out / "figures" / "overview" / "example_00.png").exists()
    inside = {p.resolve() for p in out.rglob("*")}
    assert not (tmp_path / "evil").exists() and not (tmp_path / "sub" / "evil").exists()
    assert all(str(p).startswith(str(out.resolve())) for p in inside)
    for name in ("report.json", "index.html"):
        t = (out / name).read_text()
        assert str(tmp_path) not in t
    titles = [f.title for s in res.sections for f in s.figures]
    assert any("cpd 3/10uM" in t for t in titles)
    for s in res.sections:
        assert all(str(tmp_path) not in n for n in s.notes)


def test_exception_notes_have_no_absolute_path(small_truth_dataset, tmp_path, monkeypatch):
    from synthdata.report import build as b
    from synthdata.report.sections import SECTIONS

    def boom(ctx):
        raise FileNotFoundError(2, "No such file", str(ctx.root / "figures" / "x.png"))

    monkeypatch.setattr(b, "SECTIONS", [(k, t, c, boom if k == "heatmaps" else fn) for k, t, c, fn in SECTIONS])
    out = tmp_path / "r"
    res = build_report(truth_inputs(small_truth_dataset), out)
    note = next(s for s in res.sections if s.key == "heatmaps").notes[0]
    assert str(tmp_path) not in note and "failed" in note
    assert str(tmp_path) not in (out / "report.json").read_text()


# ---- I4 ----
def test_partial_rebuild_refused_on_different_objects_without_manifest(small_truth_dataset, tmp_path):
    a = replace(truth_inputs(small_truth_dataset), source_manifest_sha256=None)
    out = tmp_path / "r"
    build_report(a, out, sections=["overview"])
    b = copy.deepcopy(a)
    b.objects = b.objects.copy()
    b.objects.loc[b.objects.index[0], "diameter"] += 0.5
    with pytest.raises(ReportError, match="different dataset"):
        build_report(b, out, sections=["overview"])
    build_report(a, out, sections=["replicates"])  # same data is accepted


def test_inputs_sha_stamped(built_small):
    _, out = built_small
    rj = json.loads((out / "report.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{64}", rj["inputs_sha256"])


# ---- A ----
def test_ari_control_merged_exceeds_ari(built_rep):
    res, out = built_rep
    m = res.metrics
    assert m["ari_control_merged"] > m["ari"]
    html = (out / "index.html").read_text()
    assert "controls merged" in html
    cap = next(s.caption for s in res.sections if s.key == "clustering")
    assert "cannot reach" in cap and "ARI 1" in cap


# ---- B ----
def test_clark_evans_no_values_note(small_truth_dataset):
    inp = replace(truth_inputs(small_truth_dataset), site_area_px=None)
    ctx = ReportContext.create(inp, "unused")
    note = next(n for n in ctx.z_notes if n.startswith("clark_evans"))
    assert "no values" in note and "MAD" not in note


# ---- C ----
def test_thumbnails_embedded_small_page(built_small):
    _, out = built_small
    html = (out / "index.html").read_text()
    # the vega bundle (0.9 MB) and the figure specs alone are ~1.5 MB, so the 1.5 MB target is unreachable on a
    # page that carries them; bound what this fix controls: the tiles (was 8 full 1024^2 PNGs, ~8 MB on the default plate)
    assert len(html.encode()) < 2_600_000
    assert sum(len(u) for u in re.findall(r'data:image/png;base64,([^"]+)', html)) < 1_000_000
    import base64
    uris = re.findall(r'<img[^>]*src="data:image/png;base64,([^"]+)"', html)
    assert uris
    from io import BytesIO
    for u in uris:
        im = Image.open(BytesIO(base64.b64decode(u)))
        assert max(im.size) <= 384
    full = list((out / "figures" / "overview").glob("example_*.png"))
    assert len(full) == 8
    assert 'href="figures/overview/example_00.png"' in html


def test_thumb_uri_downsizes_big_tile(tmp_path):
    import base64
    from io import BytesIO
    from synthdata.report.html import _thumb_uri
    p = tmp_path / "big.png"
    Image.fromarray((np.random.default_rng(0).random((1024, 1024)) * 255).astype(np.uint8)).save(p)
    im = Image.open(BytesIO(base64.b64decode(_thumb_uri(p).split(",", 1)[1])))
    assert max(im.size) == 384 and max(Image.open(p).size) == 1024


# ---- D ----
def test_no_abs_bar_glyph_in_fingerprint_specs(built_rep):
    _, out = built_rep
    for p in (out / "figures" / "fingerprints").glob("*.vl.json"):
        assert "|d|" not in p.read_text(), p


# ---- E ----
def test_violin_area_breaks_at_gap(ctx_rep_fix):
    from synthdata.report.sections import distributions as d
    chart = d._violin(ctx_rep_fix, "diameter", d.densities(ctx_rep_fix, "diameter"))
    spec = json.dumps(chart.to_dict())
    layers = chart.to_dict()["spec"]["layer"]
    assert "ga" in json.dumps(layers[0]["encoding"].get("detail", ""))


@pytest.fixture(scope="module")
def ctx_rep_fix(replicate_truth_dataset, tmp_path_factory):
    return ReportContext.create(truth_inputs(replicate_truth_dataset), tmp_path_factory.mktemp("fx"))


# ---- H ----
def test_twin_caption_mentions_chance(built_rep):
    res, _ = built_rep
    cap = next(s.caption for s in res.sections if s.key == "fingerprints")
    assert "by chance" in cap and "q" in cap
