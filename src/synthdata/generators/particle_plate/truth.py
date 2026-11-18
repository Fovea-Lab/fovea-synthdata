"""Ground-truth tables written next to every dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from synthdata.generators.particle_plate.design import DIST_FIELDS, MEASURED_PARAMS, ParticlePlateDesign
from synthdata.generators.particle_plate.layout import WellSlot
from synthdata.generators.particle_plate.qc import worst
from synthdata.generators.particle_plate.wells import ROW_LETTERS

STAT_KEYS = ("mean", "sd", "q05", "q25", "q50", "q75", "q95")

PARTICLE_SCHEMA = pa.schema([
    ("plate", pa.string()), ("well", pa.string()), ("site", pa.int32()), ("condition", pa.string()),
    ("particle_id", pa.int32()), ("x", pa.float64()), ("y", pa.float64()), ("diameter", pa.float64()),
    ("eccentricity", pa.float64()), ("orientation", pa.float64()), ("intensity", pa.float64()),
    ("profile", pa.string()), ("edge_softness", pa.float64()), ("clipped", pa.bool_()),
    ("touching", pa.bool_()), ("overlap_frac", pa.float64()), ("visible_px", pa.int32()),
])
QC_SCHEMA = pa.schema([
    ("plate", pa.string()), ("well", pa.string()), ("site", pa.int32()), ("condition", pa.string()),
    ("degree", pa.float64()), ("label", pa.string()), ("n_clustered", pa.int32()),
    ("clusters", pa.float64()), ("spread", pa.float64()), ("degree_clipped", pa.bool_()),
    ("well_label", pa.string()),
])
IMAGE_SCHEMA = pa.schema([
    ("plate", pa.string()), ("well", pa.string()), ("site", pa.int32()), ("condition", pa.string()),
    ("n_particles", pa.int32()), ("background", pa.float64()), ("illumination", pa.string()),
    ("illumination_strength", pa.float64()), ("illumination_angle", pa.float64()), ("edge_effect", pa.bool_()), ("clipped_px", pa.int64()),
    ("clamped_params", pa.int32()),
])


def _wells_schema() -> pa.Schema:
    fields = [("plate", pa.string()), ("well", pa.string()), ("row", pa.string()), ("col", pa.int32()),
              ("condition", pa.string()), ("role", pa.string()), ("replicate", pa.int32()),
              ("n_particles", pa.int32())]
    fields += [(f"dist_{f}", pa.string()) for f in (*DIST_FIELDS, "arrangement", "profile")]
    fields += [(f"{p}_{k}", pa.float64()) for p in MEASURED_PARAMS for k in STAT_KEYS]
    return pa.schema(fields)


def _stats(values: np.ndarray, prefix: str) -> dict:
    if values.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in STAT_KEYS}
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {f"{prefix}_mean": float(values.mean()), f"{prefix}_sd": float(values.std()),
            **{f"{prefix}_{k}": float(v) for k, v in zip(STAT_KEYS[2:], q)}}


def write_truth(out: Path, design: ParticlePlateDesign, slots: list[WellSlot],
                particles: list[dict], images: list[dict], qc_rows: list[dict] | None = None) -> None:
    t = Path(out) / "truth"
    t.mkdir(parents=True, exist_ok=True)
    pschema = PARTICLE_SCHEMA.append(pa.field("qc_clustered", pa.bool_())) if design.qc_defects else PARTICLE_SCHEMA
    pq.write_table(pa.Table.from_pylist(particles, schema=pschema), t / "particles.parquet")
    qc_rows = qc_rows or []
    well_worst: dict[tuple[str, str], str] = {}
    for r in qc_rows:
        k = (r["plate"], r["well"])
        well_worst[k] = worst([well_worst[k], r["label"]]) if k in well_worst else r["label"]
    qc_out = [{**r, "well_label": well_worst[(r["plate"], r["well"])]} for r in qc_rows]
    pq.write_table(pa.Table.from_pylist(qc_out, schema=QC_SCHEMA), t / "qc.parquet")
    pq.write_table(pa.Table.from_pylist(images, schema=IMAGE_SCHEMA), t / "images.parquet")
    by_well: dict[tuple[str, str], list[dict]] = {}
    for p in particles:
        by_well.setdefault((p["plate"], p["well"]), []).append(p)
    rows = []
    for s in slots:
        part = design.particle_for(s.condition)
        ps = by_well.get((s.plate_name, s.well), [])
        row = {"plate": s.plate_name, "well": s.well, "row": ROW_LETTERS[s.row], "col": s.col + 1,
               "condition": s.condition, "role": s.role, "replicate": s.replicate, "n_particles": len(ps)}
        for f in (*DIST_FIELDS, "arrangement", "profile"):
            row[f"dist_{f}"] = json.dumps(getattr(part, f).model_dump(mode="json"), sort_keys=True)
        for prm in MEASURED_PARAMS:
            row.update(_stats(np.array([p[prm] for p in ps], dtype=float), prm))
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows, schema=_wells_schema()), t / "wells.parquet")
