"""OME-NGFF (0.5) HCS plate writer: plate → row → column → field, pyramids, and truth masks as NGFF labels."""

from __future__ import annotations

from pathlib import Path

import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels, write_plate_metadata, write_well_metadata

from synthdata.generators.particle_plate.wells import ROW_LETTERS, plate_name


def scale_factors(h: int, w: int, min_size: int = 256) -> tuple[int, ...]:
    """Downsampling factors so the smallest pyramid level is ≤ min_size px on its long side."""
    factors, f = [], 2
    while max(h, w) / (f // 2) > min_size:
        factors.append(f)
        f *= 2
    return tuple(factors)


def _fix_labels_group(image_group: zarr.Group) -> None:
    # ome-zarr 0.19.2 writes the labels group as {"ome": {"labels": [...]}} without "version";
    # NGFF 0.5 validators (ome-zarr-models) reject the whole plate without it.
    labels = image_group["labels"]
    ome = dict(labels.attrs.get("ome", {}))
    ome.setdefault("version", "0.5")
    labels.attrs["ome"] = ome


class OmeZarrHcsWriter:
    name = "ome_zarr"

    def open(self, out: Path, design, slots) -> None:
        rows, cols = design.plate.grid()
        img = design.plate.image
        self.sites = design.plate.sites_per_well
        self.factors = scale_factors(img.height, img.width)
        px = img.pixel_size_um
        self.kw = {"axes": "yx", "scale_factors": self.factors, "scale": {"y": px, "x": px},
                   "axes_units": {"y": "micrometer", "x": "micrometer"}}
        self.roots: dict[int, zarr.Group] = {}
        for p in range(design.plate.plates):
            pslots = [s for s in slots if s.plate == p]
            name = plate_name(p)
            root = zarr.group(store=parse_url(str(Path(out) / f"{name}.ome.zarr"), mode="w").store)
            write_plate_metadata(root, list(ROW_LETTERS[:rows]), [str(c + 1) for c in range(cols)],
                                 [f"{ROW_LETTERS[s.row]}/{s.col + 1}" for s in pslots],
                                 field_count=self.sites, name=name)
            for s in pslots:
                wg = root.require_group(ROW_LETTERS[s.row]).require_group(str(s.col + 1))
                write_well_metadata(wg, [{"path": str(i)} for i in range(self.sites)])
                wg.attrs["synthdata"] = {"condition": s.condition, "role": s.role, "replicate": s.replicate}
            self.roots[p] = root

    def write(self, res) -> None:
        s = res.slot
        wg = self.roots[s.plate][ROW_LETTERS[s.row]][str(s.col + 1)]
        ig = wg.require_group(str(res.site - 1))
        write_image(res.image, ig, **self.kw)
        write_labels(res.mask, ig, name="particles", **self.kw)
        _fix_labels_group(ig)

    def close(self) -> None:
        pass
