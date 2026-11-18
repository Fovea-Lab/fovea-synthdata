"""Which condition goes in which well."""

from __future__ import annotations

from dataclasses import dataclass

from synthdata.core.rng import stream
from synthdata.generators.particle_plate.design import ParticlePlateDesign
from synthdata.generators.particle_plate.wells import parse_well, plate_name, well_name

NS_LAYOUT = 2


@dataclass(frozen=True)
class WellSlot:
    plate: int
    row: int
    col: int
    condition: str
    role: str
    replicate: int  # 1-based, row-major within the plate

    @property
    def well(self) -> str:
        return well_name(self.row, self.col)

    @property
    def plate_name(self) -> str:
        return plate_name(self.plate)

    def is_outer(self, rows: int, cols: int) -> bool:
        return self.row in (0, rows - 1) or self.col in (0, cols - 1)


def assign(design: ParticlePlateDesign) -> list[WellSlot]:
    rows, cols = design.plate.grid()
    order = design.layout.order or list(design.conditions)
    kind = design.layout.kind
    cells: dict[tuple[int, int], str] = {}
    if kind == "by_row":
        for i, name in enumerate(order):
            for c in range(cols):
                cells[(i, c)] = name
    elif kind == "by_column":
        for i, name in enumerate(order):
            for r in range(rows):
                cells[(r, i)] = name
    elif kind == "explicit":
        for w, name in design.layout.wells.items():
            cells[parse_well(w)] = name
    else:  # randomized: equal replicate counts, leftover wells stay empty
        per = (rows * cols) // len(order)
        labels = [n for n in order for _ in range(per)]
        positions = [(r, c) for r in range(rows) for c in range(cols)]
        seed = design.layout.seed if design.layout.seed is not None else design.seed
        pick = stream(seed, NS_LAYOUT).permutation(len(positions))[: len(labels)]
        for label, idx in zip(labels, pick):
            cells[positions[int(idx)]] = label
    slots: list[WellSlot] = []
    for p in range(design.plate.plates):
        counters: dict[str, int] = {}
        for (r, c) in sorted(cells):
            name = cells[(r, c)]
            counters[name] = counters.get(name, 0) + 1
            slots.append(WellSlot(p, r, c, name, design.conditions[name].role, counters[name]))
    return slots
