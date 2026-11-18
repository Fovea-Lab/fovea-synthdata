"""Deterministic figure and table export."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import vl_convert as vlc

from synthdata.report.theme import apply

_ID = re.compile(r'id="([A-Za-z_]+)_(\d+)"')
_REF = re.compile(r'(id="|url\(#|href="#)([A-Za-z_]+)_(\d+)(?=["\)])')


def canonical_svg(svg: str) -> str:
    """Renumber `prefix_N` ids per prefix, in order of first appearance, so exports are byte-stable."""
    counters: dict[str, int] = {}
    mapping: dict[tuple[str, str], int] = {}
    for prefix, num in _ID.findall(svg):
        if (prefix, num) not in mapping:
            mapping[(prefix, num)] = counters.get(prefix, 0)
            counters[prefix] = mapping[(prefix, num)] + 1

    def sub(m: re.Match) -> str:
        new = mapping.get((m.group(2), m.group(3)))
        return m.group(0) if new is None else f"{m.group(1)}{m.group(2)}_{new}"

    return _REF.sub(sub, svg)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def save_figure(chart, dirpath: Path, name: str, *, root: Path | None = None) -> dict:
    """Write `name.vl.json` (record), `.svg` and `.png`; paths are relative to `root` (default: dirpath)."""
    dirpath = Path(dirpath)
    root = Path(root) if root is not None else dirpath
    dirpath.mkdir(parents=True, exist_ok=True)
    spec = apply(chart).to_dict()
    spec_p, svg_p, png_p = (dirpath / f"{name}{ext}" for ext in (".vl.json", ".svg", ".png"))
    spec_p.write_text(json.dumps(spec, sort_keys=True, indent=2))
    svg_p.write_text(canonical_svg(vlc.vegalite_to_svg(spec)))
    png_p.write_bytes(vlc.vegalite_to_png(spec, scale=2))
    return {"name": name, "spec": _rel(spec_p, root), "svg": _rel(svg_p, root), "png": _rel(png_p, root)}


def save_table(df: pd.DataFrame, dirpath: Path, name: str, *, root: Path | None = None) -> dict:
    dirpath = Path(dirpath)
    root = Path(root) if root is not None else dirpath
    dirpath.mkdir(parents=True, exist_ok=True)
    pq_p, csv_p = dirpath / f"{name}.parquet", dirpath / f"{name}.csv"
    df.to_parquet(pq_p, engine="pyarrow", compression="zstd", index=False)
    df.to_csv(csv_p, float_format="%.6g", index=False)
    return {"name": name, "parquet": _rel(pq_p, root), "csv": _rel(csv_p, root)}
