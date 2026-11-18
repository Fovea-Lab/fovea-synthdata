"""The non-truth files of a cell_plate dataset: platemap.csv, images.csv and acquisition.json.

Nothing here names a phenotype or a dose strength. The measurement path reads these and nothing under `truth/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from synthdata.generators.cell_plate.design import CellPlateDesign
from synthdata.generators.cell_plate.layout import WellSlot

SOLVENT = "DMSO"
CONTROL_TYPE = {"negative_control": "negcon", "positive_control": "poscon", "blank": "blank"}
PLATEMAP_COLUMNS = ["plate", "well_position", "broad_sample", "solvent", "concentration", "control_type"]


def well_metadata(slot: WellSlot) -> dict[str, str]:
    """CPJUMP1-style metadata for one well, shared by platemap.csv and truth/wells.parquet."""
    return {
        "broad_sample": slot.compound,
        "solvent": "" if slot.role == "blank" else SOLVENT,
        "concentration": "" if slot.role in ("negative_control", "blank") else slot.dose_label,
        "control_type": CONTROL_TYPE.get(slot.role, ""),
    }


def write_platemap(out: Path, slots: list[WellSlot]) -> None:
    with open(Path(out) / "platemap.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        moa = any(s.mechanism for s in slots)  # the column exists only when some compound has a mechanism
        w.writerow([*PLATEMAP_COLUMNS, "moa"] if moa else PLATEMAP_COLUMNS)
        for s in slots:
            m = well_metadata(s)
            row = [s.plate_name, s.well, m["broad_sample"], m["solvent"], m["concentration"], m["control_type"]]
            w.writerow([*row, s.mechanism] if moa else row)


def write_images_csv(out: Path, design: CellPlateDesign, slots: list[WellSlot], image_path) -> None:
    """One row per image file to measure. `image_path(slot, site, channel_name)` is the writer's relative path."""
    with open(Path(out) / "images.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["plate", "well", "site", "channel", "path"])
        for s in slots:
            for site in range(1, design.plate.sites_per_well + 1):
                for ch in design.plate.channels:
                    w.writerow([s.plate_name, s.well, site, ch.name, image_path(s, site, ch.name)])


def write_acquisition(out: Path, design: CellPlateDesign) -> None:
    rows, cols = design.plate.grid()
    img = design.plate.image
    doc = {
        "grid": {"rows": rows, "cols": cols},
        "plates": design.plate.plates,
        "sites_per_well": design.plate.sites_per_well,
        "image": {"height": img.height, "width": img.width, "dtype": img.dtype},
        "pixel_size_um": img.pixel_size_um,
        "channels": [{"name": c.name, "stain": c.stain} for c in design.plate.channels],
    }
    (Path(out) / "acquisition.json").write_text(json.dumps(doc, indent=2) + "\n")
