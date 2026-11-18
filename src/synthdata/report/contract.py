"""The report input contract: what any object-level plate dataset must provide."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd


class ReportError(ValueError):
    """The inputs do not satisfy the report contract."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: Literal["linear", "axial"]
    label: str
    unit: str = ""


DEFAULT_FEATURES = (
    FeatureSpec("diameter", "linear", "Diameter", "px"),
    FeatureSpec("eccentricity", "linear", "Eccentricity"),
    FeatureSpec("orientation", "axial", "Orientation", "°"),
    FeatureSpec("intensity", "linear", "Intensity", "a.u."),
)

OBJECT_COLUMNS = ("plate", "well", "site", "object_id", "x", "y")
WELL_COLUMNS = ("plate", "well", "row", "col", "condition", "role", "replicate")


@dataclass
class ReportInputs:
    objects: pd.DataFrame  # plate, well, site, object_id, x, y + feature columns
    wells: pd.DataFrame  # plate, well, row, col, condition, role, replicate
    features: tuple[FeatureSpec, ...] = DEFAULT_FEATURES
    reference: str = "control"
    title: str = "Plate profile report"
    source: str = ""
    site_area_px: float | None = None
    n_sites: int | None = None  # sites per well as designed; inferred from the objects when None
    intended: dict[str, dict[str, Any]] | None = None  # {feature: {condition: Distribution}}
    images: dict[str, np.ndarray] | None = None  # {condition: 2-D example image}
    source_manifest_sha256: str | None = None
    grid: tuple[int, int] | None = None  # (rows, cols)
    notes_for_reader: dict[str, str] | None = None  # {section key: extra "what to look for" sentence}
    qc: pd.DataFrame | None = None  # plate, well, label in {pass, warn, fail} + optional numeric metric columns
    qc_title: str | None = None  # section title override from the adapter
    labels: dict[str, str] | None = None  # {column: reader-facing name}, consulted first; the unit is kept


QC_LABELS = ("pass", "warn", "fail")
QC_REQUIRED = ("plate", "well", "label")


def _need(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ReportError(f"{what} table is missing required column(s) {missing}")


def validate(inputs: ReportInputs) -> ReportInputs:
    """Check the contract; return inputs whose wells carry an ordered `condition` categorical."""
    _need(inputs.objects, OBJECT_COLUMNS, "objects")
    _need(inputs.wells, WELL_COLUMNS, "wells")
    for f in inputs.features:
        if f.name not in inputs.objects.columns:
            raise ReportError(f"declared feature {f.name!r} is not a column of objects")
    if inputs.objects.empty:
        raise ReportError("objects table is empty")
    known = set(zip(inputs.wells["plate"], inputs.wells["well"]))
    used = set(zip(inputs.objects["plate"], inputs.objects["well"]))
    unknown = sorted(used - known)
    if unknown:
        raise ReportError(f"objects reference well(s) absent from the wells table: {unknown[:5]}")
    if inputs.qc is not None:
        qc = inputs.qc
        _need(qc, QC_REQUIRED, "qc")
        bad = sorted(set(qc["label"].astype(str)) - set(QC_LABELS))
        if bad:
            raise ReportError(f"qc labels must be one of {list(QC_LABELS)}; got {bad}")
        if qc.duplicated(["plate", "well"]).any():
            raise ReportError("qc has more than one row for a well")
        stray = sorted(set(zip(qc["plate"], qc["well"])) - known)
        if stray:
            raise ReportError(f"qc references well(s) absent from the wells table: {stray[:5]}")
        for c in qc.columns:
            if c not in QC_REQUIRED and not pd.api.types.is_numeric_dtype(qc[c]):
                raise ReportError(f"qc metric column {c!r} must be numeric")
    wells = inputs.wells.copy()
    order = list(dict.fromkeys(wells["condition"].astype(str)))
    if inputs.reference not in order:
        raise ReportError(f"reference {inputs.reference!r} is not a condition; conditions: {order}")
    n_ref = int((wells["condition"].astype(str) == inputs.reference).sum())
    if n_ref < 2:
        raise ReportError(f"reference {inputs.reference!r} needs at least 2 wells, has {n_ref}")
    wells["condition"] = pd.Categorical(wells["condition"].astype(str), categories=order, ordered=True)
    return replace(inputs, wells=wells)


def inputs_sha256(inputs: ReportInputs) -> str:
    """Hash of a canonical serialisation of the objects and wells tables (sorted, fixed float format)."""
    import hashlib

    h = hashlib.sha256()
    tables = [("objects", inputs.objects, ["plate", "well", "site", "object_id"]),
              ("wells", inputs.wells, ["plate", "well"])]
    if inputs.qc is not None:  # only when present, so a report without qc keeps its hash
        tables.append(("qc", inputs.qc, ["plate", "well"]))
    for name, df, keys in tables:
        d = df.copy()
        for c in d.columns:
            if isinstance(d[c].dtype, pd.CategoricalDtype):
                d[c] = d[c].astype(str)
        d = d[sorted(d.columns)]
        d = d.sort_values([k for k in keys if k in d.columns], kind="stable")
        h.update(name.encode() + b"\n" + ",".join(d.columns).encode() + b"\n")
        h.update(d.to_csv(index=False, float_format="%.9g", header=False, lineterminator="\n").encode())
    return h.hexdigest()
