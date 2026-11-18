"""Planted QC defects: which sites get a blur or a saturated debris blob, drawn from their own stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from synthdata.core.rng import stream
from synthdata.generators.cell_plate.design import CellPlateDesign
from synthdata.generators.cell_plate.layout import WellSlot

NS_DEFECT = 13
INELIGIBLE_ROLES = ("blank", "negative_control")  # keeps the DMSO reference clean and every well measurable


@dataclass(frozen=True)
class SiteDefect:
    kind: Literal["blur", "saturated_debris"]
    psf_sigma_px: float | None = None  # blur only


def assign_defects(design: CellPlateDesign, slots: list[WellSlot]) -> dict[tuple[int, int, int, int], SiteDefect]:
    """Map (plate, row, col, site) to its planted defect. Counts are per plate; at most one defect per well; never blank or DMSO wells."""
    qc = design.qc_defects
    if qc is None:
        return {}
    n_blur = qc.blurred_sites.n if qc.blurred_sites else 0
    n_debris = qc.saturated_debris.n if qc.saturated_debris else 0
    sites = design.plate.sites_per_well
    out: dict[tuple[int, int, int, int], SiteDefect] = {}
    for plate in sorted({s.plate for s in slots}):
        eligible = sorted((s.plate, s.row, s.col) for s in slots if s.plate == plate and s.role not in INELIGIBLE_ROLES)
        if n_blur + n_debris > len(eligible):
            raise ValueError(f"qc_defects asks for {n_blur + n_debris} defective wells per plate but plate {plate} has "
                             f"only {len(eligible)} eligible wells (not blank, not DMSO; at most one defect per well)")
        rng = stream(design.seed, NS_DEFECT) if plate == 0 else stream(design.seed, NS_DEFECT, plate)
        picked = [eligible[int(i)] for i in rng.choice(len(eligible), size=n_blur + n_debris, replace=False)]
        for i, well in enumerate(picked):
            site = int(rng.integers(sites)) + 1
            if i < n_blur:
                out[(*well, site)] = SiteDefect("blur", float(qc.blurred_sites.psf_sigma_px))
            else:
                out[(*well, site)] = SiteDefect("saturated_debris", None)
    return out


def debris_mask(design: CellPlateDesign, key: tuple[int, int, int, int]) -> tuple[int, np.ndarray]:
    """Channel index and native-resolution boolean disc for the debris blob of one site."""
    spec = design.plate.image
    rng = stream(design.seed, NS_DEFECT, *key)
    channel = int(rng.integers(len(design.plate.channels)))
    radius = float(rng.uniform(0.02, 0.04)) * min(spec.height, spec.width)
    cy = rng.uniform(radius, spec.height - radius)
    cx = rng.uniform(radius, spec.width - radius)
    yy, xx = np.mgrid[0:spec.height, 0:spec.width]
    return channel, (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def qc_row(base: dict, defect: SiteDefect | None) -> dict:
    if defect is None:
        return {**base, "defect": "none", "degree": 0.0, "label": "pass"}
    return {**base, "defect": defect.kind, "degree": 1.0, "label": "fail"}
