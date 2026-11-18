import numpy as np
import pandas as pd

from synthdata.generators.particle_plate.summaries import (
    condition_colours, histogram_frame, intended_pdf_frame, layout_grid, render_tile, styled_plate,
)


def test_intended_pdf_frame(small_design):
    df = intended_pdf_frame(small_design, "diameter")
    assert list(df.columns) == list(small_design.conditions) and len(df) == 200
    x = df.index.to_numpy()
    assert x[df["large"].to_numpy().argmax()] > x[df["control"].to_numpy().argmax()]


def test_layout_grid(small_design):
    g = layout_grid(small_design)
    assert g.shape == (8, 4) and g.loc["C", 1] == "large"


def test_histogram_frame():
    df = pd.DataFrame({"condition": ["a"] * 50 + ["b"] * 50, "diameter": np.r_[np.full(50, 5.0), np.full(50, 9.0)]})
    h = histogram_frame(df, "diameter", bins=10)
    assert list(h.columns) == ["a", "b"] and len(h) == 10


def test_render_tile_overlay():
    img = np.zeros((20, 20), dtype=np.uint16)
    img[5:15, 5:15] = 1000
    mask = (img > 0).astype(np.uint32)
    t = render_tile(img, mask)
    assert t.shape == (20, 20, 3) and t.dtype == np.uint8
    assert tuple(t[5, 10]) == (255, 64, 64) and tuple(t[10, 10]) != (255, 64, 64)


def test_plate_map():
    from synthdata.generators.particle_plate.summaries import plate_map

    df = pd.DataFrame({"row": ["A", "B", "A"], "col": [1, 2, 3], "condition": ["x", "y", "z"]})
    g = plate_map(df)
    assert list(g.index) == ["A", "B"] and list(g.columns) == [1, 2, 3]
    assert g.loc["A", 1] == "x" and g.loc["B", 2] == "y" and g.loc["A", 3] == "z"
    assert g.loc["A", 2] == "" and g.loc["B", 1] == ""


def test_condition_colours_distinct_stable_and_empty_unstyled():
    names = [f"c{i}" for i in range(8)]
    c = condition_colours(names)
    assert len(set(c.values())) == 8 and all(v.startswith("#") for v in c.values())
    assert condition_colours(names) == c and "" not in c
    assert len(condition_colours([f"x{i}" for i in range(30)])) == 30  # cycles, still defined
    from synthdata.report.theme import condition_palette, tint
    assert c == {k: tint(v) for k, v in condition_palette(names).items()}  # same hues as the report
    grid = pd.DataFrame([["c0", ""]], index=["A"], columns=[1, 2])
    css = styled_plate(grid, c).to_html()
    assert c["c0"] in css and css.count("background-color") == 1


def test_layout_and_inspect_maps_share_colours(small_design):
    from synthdata.app.pages import design_colours
    assert set(design_colours(small_design)) == set(small_design.conditions)


def test_tints_keep_dark_text_readable():
    from synthdata.generators.particle_plate.summaries import TEXT_COLOUR
    from synthdata.report.theme import CATEGORY, tint

    def lum(h):
        v = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    for h in CATEGORY:
        a, b = sorted((lum(tint(h)), lum(TEXT_COLOUR)))
        assert (b + 0.05) / (a + 0.05) >= 7, h
