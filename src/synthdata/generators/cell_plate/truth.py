"""Ground-truth tables written under `truth/`. The measurement path never reads them."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from synthdata.generators.cell_plate.design import CellPlateDesign
from synthdata.generators.cell_plate.files import well_metadata
from synthdata.generators.cell_plate.layout import WellSlot
from synthdata.generators.cell_plate.well_effects import well_effects_rows
from synthdata.generators.particle_plate.wells import ROW_LETTERS, plate_name

_FALLBACK = pa.schema([("plate", pa.string()), ("well", pa.string()), ("site", pa.int32())])


def _table(rows: list[dict]) -> pa.Table:
    """Rows to a table with the union of their keys, in first-seen order (cell rows differ in extra columns)."""
    if not rows:
        return _FALLBACK.empty_table()
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows])


def wells_rows(slots: list[WellSlot]) -> list[dict]:
    rows = []
    mech = any(s.mechanism for s in slots)  # mechanism columns exist only when some compound has one
    partial = any(s.responder_fraction is not None for s in slots)  # likewise the responder fraction
    for s in slots:
        m = well_metadata(s)
        rows.append({"plate": s.plate_name, "well": s.well, "row": ROW_LETTERS[s.row], "col": s.col + 1,
                     "condition": s.condition, "role": s.role, "replicate": s.replicate,
                     "Metadata_Plate": s.plate_name, "Metadata_Well": s.well,
                     "Metadata_broad_sample": m["broad_sample"], "Metadata_control_type": m["control_type"],
                     "Metadata_concentration": m["concentration"],
                     "phenotype": s.phenotype or "", "strength": float(s.strength)})
        if mech:
            rows[-1].update({"mechanism": s.mechanism, "potency": float(s.potency),
                             "side_effect": s.side_effect or "", "side_strength": float(s.side_strength)})
        if partial:
            rows[-1]["responder_fraction"] = s.responder_fraction  # None: every cell takes the main phenotype
    return rows


def plate_effects_rows(design: CellPlateDesign) -> list[dict]:
    """One row per (plate, channel): the gain and background shift applied there, and the plate's count factor."""
    return [{"plate": plate_name(p), "channel": ch.name, "gain": float(eff.gain.get(ch.name, 1.0)),
             "background_shift": float(eff.background_shift.get(ch.name, 0.0)),
             "count_factor": float(eff.count_factor)}
            for p, eff in enumerate(design.plate_effects or []) for ch in design.plate.channels]


def write_truth(out: Path, slots: list[WellSlot], cells: list[dict], nuclei: list[dict], images: list[dict],
                qc_rows: list[dict], design: CellPlateDesign | None = None) -> None:
    t = Path(out) / "truth"
    t.mkdir(parents=True, exist_ok=True)
    for name, rows in [("cells", cells), ("nuclei", nuclei), ("images", images), ("qc", qc_rows),
                       ("wells", wells_rows(slots))]:
        pq.write_table(_table(rows), t / f"{name}.parquet")
    if design is not None and design.plate_effects is not None:
        pq.write_table(_table(plate_effects_rows(design)), t / "plate_effects.parquet")
    if design is not None and design.well_effects is not None:
        pq.write_table(_table(well_effects_rows(design, slots)), t / "well_effects.parquet")
