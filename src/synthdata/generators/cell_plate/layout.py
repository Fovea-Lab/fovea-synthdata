"""Controlled-random plate layout: fixed controls, blanks, then replicate groups under constraints."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from synthdata.core.rng import stream
from synthdata.generators.cell_plate.design import BLANK, CellPlateDesign
from synthdata.generators.particle_plate.wells import parse_well, plate_name, well_name

NS_LAYOUT = 2
MAX_RESTARTS = 5000


@dataclass(frozen=True)
class WellSlot:
    plate: int
    row: int
    col: int
    compound: str
    dose_label: str  # empty for DMSO and BLANK
    condition: str  # "CPD-001|1 uM" for test compounds, the bare name otherwise
    role: str  # negative_control | positive_control | test | blank
    replicate: int  # 1-based, row-major within a condition
    phenotype: str | None
    strength: float
    side_effect: str | None = None  # a second phenotype, applied at side_strength
    side_strength: float = 0.0
    mechanism: str = ""
    potency: float = 1.0
    responder_fraction: float | None = None  # share of cells that take the main phenotype; None: every cell

    @property
    def well(self) -> str:
        return well_name(self.row, self.col)

    @property
    def plate_name(self) -> str:
        return plate_name(self.plate)

    def is_outer(self, rows: int, cols: int) -> bool:
        return self.row in (0, rows - 1) or self.col in (0, cols - 1)


def _place_groups(rng, groups, free, rows, cols, no_shared, doses_not_adjacent):
    """One randomised greedy attempt. groups: (compound, dose index, replicates). Returns {group index: cells} or None."""
    free = set(free)
    order = list(rng.permutation(len(groups)))
    placed: dict[int, list[tuple[int, int]]] = {}
    by_compound: dict[str, dict[int, set[tuple[int, int]]]] = {}
    for gi in order:
        compound, dose, n = groups[gi]
        others = set()
        if doses_not_adjacent:
            for d, cells in by_compound.get(compound, {}).items():
                if d != dose:
                    for r, c in cells:
                        others.update(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        chosen: list[tuple[int, int]] = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        for _ in range(n):
            cand = [rc for rc in free if rc not in others
                    and not (no_shared and (rc[0] in used_r or rc[1] in used_c))]
            if not cand:
                return None
            cand.sort()
            r, c = cand[int(rng.integers(len(cand)))]
            chosen.append((r, c))
            used_r.add(r)
            used_c.add(c)
            free.discard((r, c))
        placed[gi] = chosen
        by_compound.setdefault(compound, {}).setdefault(dose, set()).update(chosen)
    return placed


def _on_one_edge(cells, rows, cols):
    return (all(r == 0 for r, _ in cells) or all(r == rows - 1 for r, _ in cells)
            or all(c == 0 for _, c in cells) or all(c == cols - 1 for _, c in cells))


def _draw_blanks(rng, free, n, rows, cols):
    """Draw n blank wells, redrawing (bounded) while they all lie on one edge of the plate."""
    if n == 0:
        return set()
    for _ in range(MAX_RESTARTS):
        cells = {free[int(i)] for i in rng.choice(len(free), size=n, replace=False)}
        if not _on_one_edge(cells, rows, cols):
            return cells
    warnings.warn("cell_plate layout: relaxing constraint blanks_not_all_on_one_edge "
                  f"after {MAX_RESTARTS} failed draws", UserWarning, stacklevel=3)
    return cells


def assign(design: CellPlateDesign) -> list[WellSlot]:
    """Place every well of every plate. Sorted by plate, row, column; deterministic in the design seed.

    Plate 0 draws from stream(seed, NS_LAYOUT) and plate p > 0 from stream(seed, NS_LAYOUT, p), so plate 0 is the
    same layout a one-plate design gets. Fixed controls sit in the same wells on every plate."""
    return [s for p in range(design.plate.plates) for s in _assign_plate(design, p)]


def _assign_plate(design: CellPlateDesign, plate: int) -> list[WellSlot]:
    rows, cols = design.plate.grid()
    lay = design.layout
    rng = stream(design.seed, NS_LAYOUT) if plate == 0 else stream(design.seed, NS_LAYOUT, plate)

    slots: dict[tuple[int, int], WellSlot] = {}
    high = max(design.doses.values(), key=lambda d: d.strength, default=None)

    def control_slot(r, c, name):
        comp = design.compounds[name]
        if comp.role == "positive_control":
            label, strength = (high.label if high else ""), float(comp.dose)
        else:
            label, strength = "", 0.0
        return WellSlot(plate, r, c, name, label, name, comp.role, 0, comp.phenotype, strength)

    for well, name in lay.fixed.items():
        r, c = parse_well(well)
        slots[(r, c)] = control_slot(r, c, name)

    free = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in slots]
    blank_cells = _draw_blanks(rng, free, lay.blanks, rows, cols)
    for r, c in blank_cells:
        slots[(r, c)] = WellSlot(plate, r, c, BLANK, "", BLANK, "blank", 0, None, 0.0)
    free = [rc for rc in free if rc not in blank_cells]

    doses = list(design.doses.values())
    groups = [(name, di, lay.replicates) for name, comp in design.compounds.items() if comp.role == "test"
              for di in range(len(doses))]

    placed = None
    phases = [(lay.constraints.no_shared_row_or_col, lay.constraints.doses_not_adjacent)]
    if lay.constraints.doses_not_adjacent:
        phases.append((lay.constraints.no_shared_row_or_col, False))
    for k, (no_shared, not_adj) in enumerate(phases):
        if k:
            warnings.warn("cell_plate layout: relaxing constraint doses_not_adjacent "
                          f"after {MAX_RESTARTS} failed attempts", UserWarning, stacklevel=2)
        for _ in range(MAX_RESTARTS):
            placed = _place_groups(rng, groups, free, rows, cols, no_shared, not_adj)
            if placed is not None:
                break
        if placed is not None:
            break
    if placed is None:
        if lay.constraints.no_shared_row_or_col:
            raise ValueError(f"layout: no placement satisfies constraint no_shared_row_or_col on the "
                             f"{rows}x{cols} plate after {MAX_RESTARTS} attempts")
        raise ValueError(f"layout: could not place every compound well on the {rows}x{cols} plate; "
                         "no constraint is left to relax, so the free wells are too few")

    for gi, cells in placed.items():
        name, di, _ = groups[gi]
        dose = doses[di]
        comp = design.compounds[name]
        for rep, (r, c) in enumerate(sorted(cells), start=1):
            side = comp.side_effect
            slots[(r, c)] = WellSlot(plate, r, c, name, dose.label, f"{name}|{dose.label}", "test", rep,
                                     comp.phenotype, float(dose.strength) * (comp.potency or 1.0),
                                     side.phenotype if side else None,
                                     float(side.strength) * float(dose.strength) if side else 0.0,
                                     comp.mechanism or "", comp.potency or 1.0, comp.responder_fraction)

    # Replicate numbers of fixed controls: 1-based, row-major within a condition.
    counter: dict[str, int] = {}
    out = []
    for key in sorted(slots):
        s = slots[key]
        if s.role in ("negative_control", "positive_control", "blank"):
            counter[s.condition] = counter.get(s.condition, 0) + 1
            s = WellSlot(s.plate, s.row, s.col, s.compound, s.dose_label, s.condition, s.role,
                         counter[s.condition], s.phenotype, s.strength)
        out.append(s)
    return out
