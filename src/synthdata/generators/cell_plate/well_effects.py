"""Per-well random variation (`design.well_effects`) and per-cell partial response (`Compound.responder_fraction`).

Each well draws from its own stream, keyed by (plate, row, col) in its own namespace, so the draws do not depend on
worker count or on any other well, and no existing stream moves. Responder draws use a per-site stream of their own,
so a responder fraction of 1 leaves every other draw of the site where it was.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthdata.core.rng import stream
from synthdata.generators.cell_plate.design import CellPlateDesign
from synthdata.generators.cell_plate.layout import WellSlot

NS_WELL = 14
NS_RESPONDER = 15


@dataclass(frozen=True)
class WellDraw:
    gain: dict[str, float]  # channel -> factor on the blurred signal
    count_factor: float
    size_factor: float
    response_factor: float | None  # None where the well has no main phenotype (DMSO)


def draw_well(design: CellPlateDesign, slot: WellSlot) -> WellDraw | None:
    """The well's draws, or None when the design has no well effects or the well is blank.

    Draw order is fixed: count, size, response, then one gain per channel in plate order. DMSO draws the response too,
    so its stream matches any other well's, but it has no main phenotype to scale."""
    fx = design.well_effects
    if fx is None or slot.role == "blank":
        return None
    names = [c.name for c in design.plate.channels]
    z = stream(design.seed, NS_WELL, slot.plate, slot.row, slot.col).standard_normal(3 + len(names))
    response = max(1.0 + fx.response_sd * float(z[2]), 0.0) if slot.phenotype is not None else None
    return WellDraw({n: float(np.exp(fx.gain_sd * z[3 + i])) for i, n in enumerate(names)},
                    float(np.exp(fx.count_sd * z[0])), float(np.exp(fx.size_sd * z[1])), response)


def responders(design: CellPlateDesign, slot: WellSlot, site: int, n: int) -> np.ndarray | None:
    """Which of the site's n requested cells take the main phenotype; None when the compound has no fraction."""
    if slot.responder_fraction is None:
        return None
    u = stream(design.seed, NS_RESPONDER, slot.plate, slot.row, slot.col, site).uniform(size=n)
    return u < slot.responder_fraction


def well_effects_rows(design: CellPlateDesign, slots: list[WellSlot]) -> list[dict]:
    """One row per non-blank well: the drawn gains and factors."""
    rows = []
    for s in slots:
        w = draw_well(design, s)
        if w is None:
            continue
        rows.append({"plate": s.plate_name, "well": s.well, "condition": s.condition, "role": s.role,
                     **{f"gain_{n}": g for n, g in w.gain.items()}, "count_factor": w.count_factor,
                     "size_factor": w.size_factor, "response_factor": w.response_factor})
    return rows
