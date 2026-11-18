"""One TIFF per site holding every channel (`cyx`, design order); instance masks apart in `truth/labels/`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from synthdata.generators.cell_plate.writers.tiff_channels import _stem, write_labels


class TiffStackWriter:
    name = "tiff_stack"

    def open(self, out: Path, design, slots) -> None:
        self.out = Path(out)
        self.channels = [c.name for c in design.plate.channels]
        (self.out / "images").mkdir()
        (self.out / "truth" / "labels").mkdir(parents=True)

    def image_path(self, slot, site: int, channel: str) -> str:
        return f"images/{_stem(slot, site)}.tif"

    def write(self, res) -> None:
        stack = np.stack([res.channels[n] for n in self.channels])
        tifffile.imwrite(self.out / self.image_path(res.slot, res.site, self.channels[0]), stack,
                         photometric="minisblack", compression="zlib", metadata={"axes": "CYX"})
        write_labels(self.out, res)

    def close(self) -> None:
        pass
