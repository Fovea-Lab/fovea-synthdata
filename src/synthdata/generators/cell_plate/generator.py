"""The cell_plate generator: design → rendered multi-channel plate + truth + manifest."""

from __future__ import annotations

import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from synthdata.core.generator import GenerationError, ProgressFn, prepare_output, writer_factory
from synthdata.core.provenance import Manifest, write_manifest
from synthdata.generators.cell_plate.design import CellPlateDesign, bundled_designs, dump_design, load_design
from synthdata.generators.cell_plate.files import write_acquisition, write_images_csv, write_platemap
from synthdata.generators.cell_plate.layout import WellSlot, assign
from synthdata.generators.cell_plate.qc import SiteDefect, assign_defects

if TYPE_CHECKING:
    from synthdata.generators.cell_plate.render import SiteResult

LABEL_BAND = 18  # px of caption above each contact-sheet tile


@dataclass
class Preview:
    tiles: dict[str, list[SiteResult]]  # condition -> rendered sites


def _render_task(args: tuple[CellPlateDesign, WellSlot, int, SiteDefect | None]) -> SiteResult:
    from .render import render_site

    design, slot, site, defect = args
    try:
        return render_site(design, slot, site, defect)
    except Exception as exc:  # re-raised picklable, naming the site that failed
        raise GenerationError(
            f"render failed at {slot.plate_name} {slot.well} site {site}: {type(exc).__name__}: {exc}"
        ) from None


class CellPlateGenerator:
    name = "cell_plate"
    version = "0.1.0"
    Design = CellPlateDesign
    presets = staticmethod(bundled_designs)
    writers = {
        "ome_zarr": writer_factory(f"{__package__}.writers.ome_zarr_hcs", "OmeZarrHcsWriter"),
        "tiff": writer_factory(f"{__package__}.writers.tiff_channels", "TiffChannelsWriter"),
        "tiff_stack": writer_factory(f"{__package__}.writers.tiff_stack", "TiffStackWriter"),
    }
    default_writer = "ome_zarr"

    @staticmethod
    def check(dataset: Path, *, write: bool = True):
        from .check import check_dataset

        return check_dataset(dataset, write=write)

    def load_design(self, source: Any) -> CellPlateDesign:
        return load_design(source)

    def dump_design(self, design: CellPlateDesign) -> str:
        return dump_design(design)

    def preview(self, design: Any, n_per_condition: int = 1) -> Preview:
        """Render the first well of every condition, without writing anything."""
        from .render import render_site

        design = load_design(design)
        first: dict[str, WellSlot] = {}
        slots = assign(design)
        defects = assign_defects(design, slots)
        for s in slots:
            first.setdefault(s.condition, s)
        return Preview({
            condition: [render_site(design, slot, site, defects.get((slot.plate, slot.row, slot.col, site)))
                        for site in range(1, n_per_condition + 1)]
            for condition, slot in first.items()
        })

    def contact_sheet(self, preview: Preview, tile: int = 256):
        """One row per condition, with the channels of each previewed site side by side."""
        from PIL import Image, ImageDraw

        sites = [site for row in preview.tiles.values() for site in row]
        names = list(sites[0].channels)
        limits = {}
        for n in names:
            sample = np.concatenate([r.channels[n].ravel()[::7] for r in sites]).astype(float)
            limits[n] = np.percentile(sample, [1, 99.5])
        cols = max(len(row) for row in preview.tiles.values()) * len(names)
        sheet = Image.new("L", (cols * tile, len(preview.tiles) * (tile + LABEL_BAND)), 0)
        draw = ImageDraw.Draw(sheet)
        for i, (cond, rs) in enumerate(preview.tiles.items()):
            top = i * (tile + LABEL_BAND)
            for site_index, site in enumerate(rs):
                for channel_index, name in enumerate(names):
                    left = (site_index * len(names) + channel_index) * tile
                    lo, hi = limits[name]
                    g = np.clip((site.channels[name].astype(float) - lo) / max(hi - lo, 1e-9), 0, 1)
                    im = Image.fromarray((g * 255).astype(np.uint8)).resize((tile, tile), Image.BILINEAR)
                    sheet.paste(im, (left, top + LABEL_BAND))
                    draw.text((left + 4, top + 3), f"{cond} · {site.slot.well} s{site.site} · {name}", fill=255)
        return sheet

    def generate(self, design: Any, out: Path, writer: str | None = None, workers: int = 1,
                 progress: ProgressFn | None = None, overwrite: bool = False) -> Manifest:
        from .truth import write_truth

        design = load_design(design)
        out = Path(out)
        writer_name = writer or self.default_writer
        if writer_name not in self.writers:
            raise GenerationError(f"unknown writer {writer_name!r}; choose from {sorted(self.writers)}")
        w = self.writers[writer_name]()
        prepare_output(out, overwrite)
        slots = assign(design)
        defects = assign_defects(design, slots)
        tasks = [(design, s, site, defects.get((s.plate, s.row, s.col, site)))
                 for s in slots for site in range(1, design.plate.sites_per_well + 1)]
        w.open(out, design, slots)
        resolved_yaml = dump_design(design)
        (out / "design.yaml").write_text(resolved_yaml)
        write_platemap(out, slots)
        write_images_csv(out, design, slots, w.image_path)
        write_acquisition(out, design)
        cells: list[dict] = []
        nuclei: list[dict] = []
        images: list[dict] = []
        qc_rows: list[dict] = []
        not_placed = 0

        def consume(results) -> None:
            nonlocal not_placed
            for i, res in enumerate(results, 1):
                w.write(res)
                cells.extend(res.cell_rows)
                nuclei.extend(res.nucleus_rows)
                images.extend({**r, "path": w.image_path(res.slot, res.site, r["channel"])} for r in res.image_rows)
                qc_rows.extend(res.qc_rows)
                not_placed += res.image_rows[0]["n_cells_requested"] - res.image_rows[0]["n_cells"]
                if progress:
                    progress(i, len(tasks))

        if workers <= 1:
            consume(map(_render_task, tasks))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                consume(ex.map(_render_task, tasks, chunksize=2))
        w.close()
        write_truth(out, slots, cells, nuclei, images, qc_rows, design)
        if not_placed:
            warnings.warn(f"cell_plate: {not_placed} of {len(cells) + not_placed} requested cells could not be "
                          "placed; the design is too dense for the image", UserWarning, stacklevel=2)
        return write_manifest(out, generator=self.name, generator_version=self.version, seed=design.seed,
                              design_yaml=resolved_yaml, writer=writer_name,
                              extra={"n_images": len(tasks), "n_wells": len(slots), "n_cells": len(cells),
                                     "n_nuclei": len(nuclei), "n_cells_not_placed": not_placed})
