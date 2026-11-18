"""Plain folder writer: one TIFF per image and mask, plus platemap.csv (what today's platform ingest reads)."""

from __future__ import annotations

import csv
from pathlib import Path

import tifffile

from synthdata.generators.particle_plate.wells import ROW_LETTERS


class TiffPlatemapWriter:
    name = "tiff"

    def open(self, out: Path, design, slots) -> None:
        self.out = Path(out)
        (self.out / "images").mkdir()
        (self.out / "masks").mkdir()
        with open(self.out / "platemap.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["plate", "well", "row", "col", "condition", "role", "replicate"])
            for s in slots:
                w.writerow([s.plate_name, s.well, ROW_LETTERS[s.row], s.col + 1, s.condition, s.role, s.replicate])

    def write(self, res) -> None:
        stem = f"{res.slot.plate_name}_{res.slot.well}_s{res.site}"
        tifffile.imwrite(self.out / "images" / f"{stem}.tif", res.image, compression="zlib")
        tifffile.imwrite(self.out / "masks" / f"{stem}_mask.tif", res.mask, compression="zlib")

    def close(self) -> None:
        pass
