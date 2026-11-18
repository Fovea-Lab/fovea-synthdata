"""Read a generated dataset back, whichever writer produced it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import tifffile
import zarr

from synthdata.core.provenance import read_manifest
from synthdata.generators.particle_plate.wells import ROW_LETTERS, parse_well


@dataclass
class DatasetImage:
    plate: str
    well: str
    site: int
    image: np.ndarray
    mask: np.ndarray


def dataset_writer(out: Path) -> str:
    return read_manifest(out).writer


def list_images(out: Path) -> list[tuple[str, str, int]]:
    t = pq.read_table(Path(out) / "truth" / "images.parquet", columns=["plate", "well", "site"]).to_pylist()
    return [(r["plate"], r["well"], int(r["site"])) for r in t]


def _level0(group: zarr.Group) -> np.ndarray:
    path = dict(group.attrs)["ome"]["multiscales"][0]["datasets"][0]["path"]
    return group[path][:]


def read_image(out: Path, plate: str, well: str, site: int) -> tuple[np.ndarray, np.ndarray]:
    out = Path(out)
    if dataset_writer(out) == "tiff":
        stem = f"{plate}_{well}_s{site}"
        return tifffile.imread(out / "images" / f"{stem}.tif"), tifffile.imread(out / "masks" / f"{stem}_mask.tif")
    row, col = parse_well(well)
    ig = zarr.open_group(str(out / f"{plate}.ome.zarr" / ROW_LETTERS[row] / str(col + 1) / str(site - 1)), mode="r")
    return _level0(ig), _level0(ig["labels"]["particles"])


def iter_images(out: Path) -> Iterator[DatasetImage]:
    for plate, well, site in list_images(out):
        image, mask = read_image(out, plate, well, site)
        yield DatasetImage(plate, well, site, image, mask)


def validate_ngff(zarr_path: Path) -> None:
    """Raise (pydantic ValidationError) if the plate is not valid OME-NGFF 0.5 HCS."""
    from ome_zarr_models.v05.hcs import HCS

    HCS.from_zarr(zarr.open_group(str(zarr_path), mode="r"))
