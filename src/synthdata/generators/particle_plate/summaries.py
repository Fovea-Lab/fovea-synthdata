"""Tables and tiles for the app and for figures. No Streamlit here, so it stays unit-testable."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from synthdata.generators.particle_plate.design import ParticlePlateDesign
from synthdata.generators.particle_plate.layout import assign
from synthdata.generators.particle_plate.wells import ROW_LETTERS

OUTLINE = (255, 64, 64)

# Light qualitative palette (Tableau-10 tints + two extras); dark text stays readable on all of them.
TEXT_COLOUR = "#1a1a1a"


def condition_colours(conditions: list[str]) -> dict[str, str]:
    """Condition -> hex colour, in the given order; light tints of the report palette (Tableau-10, cycling); '' gets none."""
    from synthdata.report.theme import condition_palette, tint
    names = list(dict.fromkeys(c for c in conditions if c))
    return {c: tint(h) for c, h in condition_palette(names).items()}


def styled_plate(grid: pd.DataFrame, colours: dict[str, str]):
    """A pandas Styler: each cell's background follows its condition; empty wells stay unstyled."""
    def one(v):
        c = colours.get(v)
        return f"background-color: {c}; color: {TEXT_COLOUR}" if c else ""
    return grid.style.map(one)


def intended_pdf_frame(design: ParticlePlateDesign, param: str, n: int = 200) -> pd.DataFrame:
    dists = {c: getattr(design.particle_for(c), param) for c in design.conditions}
    descs = [d.describe() for d in dists.values()]
    x = np.linspace(min(d.lo for d in descs), max(d.hi for d in descs), n)
    return pd.DataFrame({c: d.pdf(x) for c, d in dists.items()}, index=pd.Index(x, name=param))


def plate_map(df: pd.DataFrame) -> pd.DataFrame:
    """Row letters × column numbers → condition name ('' for empty wells)."""
    rows = sorted(df["row"].unique(), key=ROW_LETTERS.index)
    cols = sorted(int(c) for c in df["col"].unique())
    grid = pd.DataFrame("", index=pd.Index(rows, name="row"), columns=pd.Index(cols, name="col"))
    for r in df.itertuples():
        grid.loc[r.row, int(r.col)] = r.condition
    return grid


def layout_grid(design: ParticlePlateDesign, plate: int = 0) -> pd.DataFrame:
    rows, cols = design.plate.grid()
    grid = pd.DataFrame("", index=pd.Index(list(ROW_LETTERS[:rows]), name="row"),
                        columns=pd.Index(list(range(1, cols + 1)), name="col"))
    for s in assign(design):
        if s.plate == plate:
            grid.loc[ROW_LETTERS[s.row], s.col + 1] = s.condition
    return grid


def histogram_frame(particles: pd.DataFrame, param: str, bins: int = 40) -> pd.DataFrame:
    values = particles[param].to_numpy(dtype=float)
    lo, hi = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    cols = {}
    for cond, g in particles.groupby("condition", sort=False):
        counts, _ = np.histogram(g[param].to_numpy(dtype=float), bins=edges, density=True)
        cols[cond] = counts
    return pd.DataFrame(cols, index=pd.Index(centres, name=param))


def render_tile(image: np.ndarray, mask: np.ndarray | None = None, lo: float | None = None,
                hi: float | None = None) -> np.ndarray:
    img = image.astype(float)
    lo = float(np.percentile(img, 1)) if lo is None else lo
    hi = float(np.percentile(img, 99.5)) if hi is None else hi
    g = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    if mask is not None:
        rgb[find_boundaries(mask, mode="inner")] = OUTLINE
    return rgb
