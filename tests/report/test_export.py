import json

import altair as alt
import pandas as pd

from synthdata.report import theme
from synthdata.report.export import canonical_svg, save_figure, save_table


def heat():
    df = pd.DataFrame({"r": list("AABB"), "c": ["1", "2", "1", "2"], "v": [1.0, 2.0, 3.0, 4.0]})
    return alt.Chart(df).mark_rect().encode(x="c:N", y="r:N", color="v:Q")


def test_canonical_svg_renumbers():
    svg = '<defs><linearGradient id="gradient_7"/></defs><rect fill="url(#gradient_7)"/><use href="#clip_3"/><g id="clip_3"/>'
    out = canonical_svg(svg)
    assert 'id="gradient_0"' in out and "url(#gradient_0)" in out
    assert 'id="clip_0"' in out and 'href="#clip_0"' in out
    assert "gradient_7" not in out


def test_canonical_svg_idempotent():
    svg = '<a id="gradient_0"/><b id="gradient_1"/><c fill="url(#gradient_1)"/><d xlink:href="#gradient_0"/>'
    assert canonical_svg(svg) == svg


def test_canonical_svg_no_collision_on_swap():
    svg = '<a id="g_1"/><b id="g_0"/><c fill="url(#g_1)"/>'
    out = canonical_svg(svg)
    assert out == '<a id="g_0"/><b id="g_1"/><c fill="url(#g_0)"/>'


def test_save_figure_writes_three(tmp_path):
    rec = save_figure(heat(), tmp_path / "figures", "heat", root=tmp_path)
    assert rec["name"] == "heat"
    for k in ("spec", "svg", "png"):
        assert (tmp_path / rec[k]).is_file()
    assert rec["spec"] == "figures/heat.vl.json"


def test_save_figure_deterministic(tmp_path):
    a = save_figure(heat(), tmp_path / "a", "h", root=tmp_path)
    b = save_figure(heat(), tmp_path / "b", "h", root=tmp_path)
    for k in ("spec", "svg", "png"):
        assert (tmp_path / a[k]).read_bytes() == (tmp_path / b[k]).read_bytes()


def test_spec_embeds_theme(tmp_path):
    rec = save_figure(heat(), tmp_path, "h", root=tmp_path)
    spec = json.loads((tmp_path / rec["spec"]).read_text())
    assert spec["config"]["font"] == theme.THEME["font"]


def test_save_table_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [0.123456789, 2.5]})
    rec = save_table(df, tmp_path / "tables", "t", root=tmp_path)
    back = pd.read_parquet(tmp_path / rec["parquet"])
    pd.testing.assert_frame_equal(back, df)
    assert (tmp_path / rec["csv"]).read_text().splitlines()[1] == "1,0.123457"


def test_register_idempotent():
    try:
        theme.register()
        theme.register()
        assert alt.theme.active == "synthdata"
    finally:
        alt.theme.enable("default")
