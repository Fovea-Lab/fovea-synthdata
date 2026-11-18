"""The particle_plate generator: design → rendered plate + truth + manifest."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synthdata.core.generator import GenerationError, ProgressFn, prepare_output, writer_factory
from synthdata.core.provenance import Manifest, write_manifest
from synthdata.generators.particle_plate.design import ParticlePlateDesign, bundled_designs, dump_design, load_design
from synthdata.generators.particle_plate.layout import WellSlot, assign
from synthdata.generators.particle_plate.qc import assign_aggregation_detail
from synthdata.generators.particle_plate.render import RenderResult, render_image

LABEL_BAND = 18  # px of caption above each contact-sheet tile


@dataclass
class Preview:
    tiles: dict[str, list[RenderResult]]


def _render_task(args: tuple[ParticlePlateDesign, WellSlot, int, float]) -> RenderResult:
    design, slot, site, degree = args
    try:
        return render_image(design, slot, site, degree) if degree else render_image(design, slot, site)
    except Exception as exc:  # re-raised picklable, naming the image that failed
        raise GenerationError(
            f"render failed at {slot.plate_name} {slot.well} site {site}: {type(exc).__name__}: {exc}"
        ) from None


class ParticlePlateGenerator:
    name = "particle_plate"
    version = "0.1.0"
    Design = ParticlePlateDesign
    presets = staticmethod(bundled_designs)
    writers = {
        "ome_zarr": writer_factory(f"{__package__}.writers.ome_zarr_hcs", "OmeZarrHcsWriter"),
        "tiff": writer_factory(f"{__package__}.writers.tiff_platemap", "TiffPlatemapWriter"),
    }
    default_writer = "ome_zarr"

    @staticmethod
    def check(dataset: Path, *, write: bool = True):
        from .check import check_dataset

        return check_dataset(dataset, write=write)

    @staticmethod
    def report_inputs(dataset: Path):
        from .truth_report import truth_inputs

        return truth_inputs(dataset)

    def load_design(self, source: Any) -> ParticlePlateDesign:
        return load_design(source)

    def dump_design(self, design: ParticlePlateDesign) -> str:
        return dump_design(design)

    def preview(self, design: Any, n_per_condition: int = 1) -> Preview:
        design = load_design(design)
        first: dict[str, WellSlot] = {}
        slots = assign(design)
        degrees, _ = assign_aggregation_detail(design, slots)
        for s in slots:
            first.setdefault(s.condition, s)
        return Preview({c: [render_image(design, s, site, degrees.get((s.plate, s.row, s.col, site), 0.0))
                            for site in range(1, n_per_condition + 1)]
                        for c, s in first.items()})

    def contact_sheet(self, preview: Preview, tile: int = 256):
        from PIL import Image, ImageDraw

        results = [r for rs in preview.tiles.values() for r in rs]
        sample = np.concatenate([r.image.ravel()[::7] for r in results]).astype(float)
        lo, hi = np.percentile(sample, [1, 99.5])
        cols = max(len(rs) for rs in preview.tiles.values())
        sheet = Image.new("L", (cols * tile, len(preview.tiles) * (tile + LABEL_BAND)), 0)
        draw = ImageDraw.Draw(sheet)
        for i, (cond, rs) in enumerate(preview.tiles.items()):
            for j, r in enumerate(rs):
                g = np.clip((r.image.astype(float) - lo) / max(hi - lo, 1e-9), 0, 1)
                im = Image.fromarray((g * 255).astype(np.uint8)).resize((tile, tile), Image.BILINEAR)
                top = i * (tile + LABEL_BAND)
                sheet.paste(im, (j * tile, top + LABEL_BAND))
                draw.text((j * tile + 4, top + 3), f"{cond} · {r.slot.well} s{r.site}", fill=255)
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
        degrees, clipped = assign_aggregation_detail(design, slots)
        tasks = [(design, s, site, degrees.get((s.plate, s.row, s.col, site), 0.0)) for s in slots for site in range(1, design.plate.sites_per_well + 1)]
        w.open(out, design, slots)
        resolved_yaml = dump_design(design.expanded())
        (out / "design.yaml").write_text(resolved_yaml)
        particles: list[dict] = []
        images: list[dict] = []
        qc_rows: list[dict] = []

        def consume(results) -> None:
            for i, res in enumerate(results, 1):
                w.write(res)
                particles.extend(res.particles)
                images.append(res.image_row)
                qc_rows.append({**res.qc_row, "degree_clipped": (res.slot.plate, res.slot.row, res.slot.col, res.site) in clipped})
                if progress:
                    progress(i, len(tasks))

        if workers <= 1:
            consume(map(_render_task, tasks))
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                consume(ex.map(_render_task, tasks, chunksize=2))
        w.close()
        write_truth(out, design, slots, particles, images, qc_rows)
        return write_manifest(out, generator=self.name, generator_version=self.version, seed=design.seed,
                              design_yaml=resolved_yaml, writer=writer_name,
                              extra={"n_images": len(tasks), "n_wells": len(slots), "n_particles": len(particles),
                                    **({"qc_degree_clipped": sum(r["degree_clipped"] for r in qc_rows)}
                                       if design.qc_defects else {})})
