"""OME-NGFF (0.5) HCS plate writer for multi-channel sites: axes `cyx`, truth masks as NGFF labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels, write_plate_metadata, write_well_metadata

from synthdata.generators.cell_plate.files import well_metadata
from synthdata.generators.particle_plate.wells import ROW_LETTERS, plate_name
from synthdata.generators.particle_plate.writers.ome_zarr_hcs import _fix_labels_group, scale_factors

CHANNEL_COLORS = ("0000FF", "00FF00", "FF0000", "FFFF00", "FF00FF", "00FFFF", "FFFFFF")


class OmeZarrHcsWriter:
    name = "ome_zarr"

    def open(self, out: Path, design, slots) -> None:
        rows, cols = design.plate.grid()
        img = design.plate.image
        self.sites = design.plate.sites_per_well
        factors = scale_factors(img.height, img.width)
        px = img.pixel_size_um
        units = {"y": "micrometer", "x": "micrometer"}
        scale = {"c": 1.0, "y": px, "x": px}
        self.image_kw = {"axes": "cyx", "scale_factors": factors, "scale": scale, "axes_units": units}
        self.label_kw = {"axes": "yx", "scale_factors": factors, "scale": {"y": px, "x": px}, "axes_units": units}
        top = int(np.iinfo(img.dtype).max)
        self.omero = {"channels": [
            {"label": c.name, "color": CHANNEL_COLORS[i % len(CHANNEL_COLORS)], "active": True,
             "window": {"min": 0, "max": top, "start": 0, "end": top}}
            for i, c in enumerate(design.plate.channels)]}
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
                wg.attrs["synthdata"] = well_metadata(s)
            self.roots[p] = root

    def image_path(self, slot, site: int, channel: str) -> str:
        """The site's group, relative to the dataset root; all channels of a site share it."""
        return f"{slot.plate_name}.ome.zarr/{ROW_LETTERS[slot.row]}/{slot.col + 1}/{site - 1}"

    def write(self, res) -> None:
        s = res.slot
        wg = self.roots[s.plate][ROW_LETTERS[s.row]][str(s.col + 1)]
        ig = wg.require_group(str(res.site - 1))
        write_image(np.stack(list(res.channels.values())), ig, **self.image_kw)
        ome = dict(ig.attrs["ome"])
        ome["omero"] = self.omero  # ome-zarr 0.19 drops a nested `metadata=` argument, so it is set here
        ig.attrs["ome"] = ome
        write_labels(res.nuclei_mask, ig, name="nuclei", **self.label_kw)
        write_labels(res.cell_mask, ig, name="cells", **self.label_kw)
        _fix_labels_group(ig)

    def close(self) -> None:
        pass
