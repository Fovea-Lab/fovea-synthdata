"""Read a cell_plate dataset back, whichever writer produced it.

`read_site` and the listing functions touch only non-truth files. `read_truth_masks` is truth only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr

from synthdata.core.provenance import read_manifest
from synthdata.generators.particle_plate.reader import _level0
from synthdata.generators.particle_plate.wells import ROW_LETTERS, parse_well


def read_platemap(out: Path) -> pd.DataFrame:
    return pd.read_csv(Path(out) / "platemap.csv", keep_default_na=False, dtype=str)


def read_acquisition(out: Path) -> dict:
    return json.loads((Path(out) / "acquisition.json").read_text())


def list_sites(out: Path) -> list[tuple[str, str, int]]:
    """Every (plate, well, site) in `images.csv`, in file order."""
    df = pd.read_csv(Path(out) / "images.csv", keep_default_na=False, dtype=str)
    return list(dict.fromkeys((r.plate, r.well, int(r.site)) for r in df.itertuples()))


def _site_group(out: Path, plate: str, well: str, site: int) -> zarr.Group:
    row, col = parse_well(well)
    return zarr.open_group(str(Path(out) / f"{plate}.ome.zarr" / ROW_LETTERS[row] / str(col + 1) / str(site - 1)),
                           mode="r")


def read_site(out: Path, plate: str, well: str, site: int) -> dict[str, np.ndarray]:
    """The channel images of one site, keyed by channel name. Never opens truth."""
    out = Path(out)
    names = [c["name"] for c in read_acquisition(out)["channels"]]
    writer = read_manifest(out).writer
    if writer == "tiff_stack":
        stack = tifffile.imread(out / "images" / f"{plate}_{well}_s{site}.tif")
        return {n: stack[i] for i, n in enumerate(names)}
    if writer == "tiff":
        return {n: tifffile.imread(out / "images" / f"{plate}_{well}_s{site}_{n}.tif") for n in names}
    data = _level0(_site_group(out, plate, well, site))
    return {n: data[i] for i, n in enumerate(names)}


def read_truth_masks(out: Path, plate: str, well: str, site: int) -> tuple[np.ndarray, np.ndarray]:
    """(nuclei, cells) instance masks of one site. Truth only."""
    out = Path(out)
    if read_manifest(out).writer in ("tiff", "tiff_stack"):
        stem = out / "truth" / "labels" / f"{plate}_{well}_s{site}"
        return tifffile.imread(f"{stem}_nuclei.tif"), tifffile.imread(f"{stem}_cells.tif")
    labels = _site_group(out, plate, well, site)["labels"]
    return _level0(labels["nuclei"]), _level0(labels["cells"])
