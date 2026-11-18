"""Plain folder writer: one TIFF per channel in `images/`, instance masks apart in `truth/labels/`."""

from __future__ import annotations

from pathlib import Path

import tifffile


def _stem(slot, site: int) -> str:
    return f"{slot.plate_name}_{slot.well}_s{site}"


def write_labels(out: Path, res) -> None:
    """Instance masks of one site into `truth/labels/`; shared by the TIFF writers."""
    stem = _stem(res.slot, res.site)
    labels = Path(out) / "truth" / "labels"
    tifffile.imwrite(labels / f"{stem}_nuclei.tif", res.nuclei_mask, compression="zlib")
    tifffile.imwrite(labels / f"{stem}_cells.tif", res.cell_mask, compression="zlib")


class TiffChannelsWriter:
    name = "tiff"

    def open(self, out: Path, design, slots) -> None:
        self.out = Path(out)
        (self.out / "images").mkdir()
        (self.out / "truth" / "labels").mkdir(parents=True)

    def image_path(self, slot, site: int, channel: str) -> str:
        return f"images/{_stem(slot, site)}_{channel}.tif"

    def write(self, res) -> None:
        for name, img in res.channels.items():
            tifffile.imwrite(self.out / self.image_path(res.slot, res.site, name), img, compression="zlib")
        write_labels(self.out, res)

    def close(self) -> None:
        pass
