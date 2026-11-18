"""The shared state every report section reads, and the records a section returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from synthdata.report.contract import FeatureSpec, ReportInputs, validate
from synthdata.report.profiles import PROFILE_COLUMNS, n_sites, robust_z, well_profiles
from synthdata.report.stats import effects
from synthdata.report.theme import condition_palette


@dataclass
class FigureRecord:
    name: str
    title: str
    spec: str | None  # relpath of the .vl.json; None for a raster tile
    svg: str | None
    png: str


@dataclass
class TableRecord:
    name: str
    parquet: str
    csv: str


@dataclass
class SectionResult:
    key: str
    title: str
    caption: str
    figures: list[FigureRecord] = field(default_factory=list)
    tables: list[TableRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ReportContext:
    inputs: ReportInputs
    root: Path
    profiles: pd.DataFrame
    z: pd.DataFrame
    z_notes: list[str]
    effects: pd.DataFrame
    columns: list[str]
    palette: dict[str, str]
    n_sites: int
    # filled by build before sections 5-7
    embedding: Any = None
    object_sample: Any = None
    object_embedding: Any = None
    clustering: Any = None

    @classmethod
    def create(cls, inputs: ReportInputs, root: Path) -> "ReportContext":
        inputs = validate(inputs)
        prof = well_profiles(inputs)
        cols = PROFILE_COLUMNS(inputs.features)
        z, notes = robust_z(prof, inputs.reference, cols)
        return cls(inputs=inputs, root=Path(root), profiles=prof, z=z, z_notes=notes,
                   effects=effects(prof, inputs.reference, cols), columns=cols,
                   palette=condition_palette(list(inputs.wells["condition"].cat.categories)),
                   n_sites=n_sites(inputs))

    @property
    def conditions(self) -> list[str]:
        return list(self.inputs.wells["condition"].cat.categories)

    def feature(self, name: str) -> FeatureSpec:
        return next(f for f in self.inputs.features if f.name == name)

    def label(self, column: str, with_unit: bool = True) -> str:
        """Reader-facing name of a profile column or feature."""
        return column_label(column, self.inputs.features, with_unit, self.inputs.labels)


_FIXED = {
    "count_per_site": ("Objects per site", ""),
    "orientation_coherence": ("Orientation coherence", ""),
    "nn_distance": ("Nearest-neighbour distance", "px"),
    "clark_evans": ("Clark-Evans index", ""),
    "diameter_bimodality": ("Diameter bimodality", ""),
}


def column_label(column: str, features: tuple[FeatureSpec, ...], with_unit: bool = True,
                 labels: dict[str, str] | None = None) -> str:
    by = {f.name: f for f in features}
    if labels and column in labels:  # the adapter's name wins; the unit is still the one the column carries
        unit = by[column].unit if column in by else _FIXED.get(column, ("", ""))[1]
        return f"{labels[column]} ({unit})" if unit and with_unit else labels[column]
    if column in by:
        f = by[column]
        return f"{f.label} ({f.unit})" if f.unit and with_unit else f.label
    if column in _FIXED:
        text, unit = _FIXED[column]
    else:
        base, _, stat = column.rpartition("_")
        if base in by and stat in ("median", "iqr"):
            f = by[base]
            text, unit = f"{f.label} {stat.upper() if stat == 'iqr' else stat}", f.unit
        else:
            text, unit = column.replace("_", " "), ""
    return f"{text} ({unit})" if unit and with_unit else text
