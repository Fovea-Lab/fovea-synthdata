"""QC defect assignment (which images carry an aggregation defect, and how strongly) and QC labels."""

from __future__ import annotations

from synthdata.core.rng import stream
from synthdata.generators.particle_plate.design import ParticlePlateDesign, QCThresholds
from synthdata.generators.particle_plate.layout import WellSlot

NS_QC = 3  # distinct from NS_IMAGE (1) and NS_LAYOUT (2)
LABELS = ("pass", "warn", "fail")  # ordered worst-last


def label_for(degree: float, th: QCThresholds) -> str:
    return "pass" if degree < th.warn else ("warn" if degree < th.fail else "fail")


def worst(labels) -> str:
    return max(labels, key=LABELS.index)


def thresholds_for(design: ParticlePlateDesign) -> QCThresholds:
    return design.qc_defects.thresholds if design.qc_defects else QCThresholds()


def assign_aggregation(design: ParticlePlateDesign, slots: list[WellSlot]) -> dict[tuple[int, int, int, int], float]:
    """Degree per (plate, row, col, site) for affected units only. See `assign_aggregation_detail`."""
    return assign_aggregation_detail(design, slots)[0]


def assign_aggregation_detail(design: ParticlePlateDesign, slots: list[WellSlot]):
    """(degrees, clipped): degree per (plate, row, col, site) for affected units only, and the set of keys whose
    raw degree draw fell outside [0, 1] and was clipped. Deterministic in the design seed.

    Unit selection uses stream (NS_QC, 0); each unit's degree is drawn from its own stream
    (NS_QC, 2, plate, row, col[, site]), so one unit's membership never shifts another's degree.
    Explicit units are keyed by well ('B03') or image ('B03:s2') and apply to every plate."""
    agg = design.qc_defects.aggregation if design.qc_defects else None
    if agg is None or agg.assign.kind == "none":
        return {}, set()
    sites = design.plate.sites_per_well
    image_level = agg.level == "image"
    # a unit is (slot, site) at image level, or (slot, None) at well level
    units = [(s, site) for s in slots for site in (range(1, sites + 1) if image_level else (None,))]
    rng = stream(design.seed, NS_QC, 0)
    a = agg.assign
    degrees: dict[int, float] = {}  # unit index -> degree
    if a.kind == "random":
        k = round(a.fraction * len(units))
        picked = sorted(int(i) for i in rng.permutation(len(units))[:k])
    elif a.kind == "by_condition":
        picked = [i for i, (s, _) in enumerate(units) if rng.random() < a.conditions.get(s.condition, 0.0)]
    else:  # explicit
        for i, (s, site) in enumerate(units):
            key = s.well if site is None else f"{s.well}:s{site}"
            if key in a.units:
                degrees[i] = float(a.units[key])
        picked = sorted(degrees)
    clipped_units: set[int] = set()
    if a.kind != "explicit":
        for i in picked:
            s, site = units[i]
            key = (NS_QC, 2, s.plate, s.row, s.col) + (() if site is None else (site,))
            raw = float(agg.degree.sample(stream(design.seed, *key), 1)[0])
            degrees[i] = min(1.0, max(0.0, raw))
            if degrees[i] != raw:
                clipped_units.add(i)
    out: dict[tuple[int, int, int, int], float] = {}
    clipped: set[tuple[int, int, int, int]] = set()
    for i, deg in degrees.items():
        s, site = units[i]
        for st in ([site] if site is not None else range(1, sites + 1)):
            out[(s.plate, s.row, s.col, st)] = deg
            if i in clipped_units:
                clipped.add((s.plate, s.row, s.col, st))
    return out, clipped
