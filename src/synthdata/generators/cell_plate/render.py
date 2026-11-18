"""Render one (plate, well, site) of a cell plate: cells -> ideal channels -> optics and sensor -> pixels.

Physics order, fixed by the spec: ideal channels at 2x supersampling, box downsample, per-channel Gaussian PSF,
background, illumination falloff, edge dim, Poisson shot noise, Gaussian read noise, clip to dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from synthdata.core.rng import stream
from synthdata.generators.cell_plate import channels as marker
from synthdata.generators.cell_plate.cells import _THETA, SitePlan, _paint, rasterise, sample_cells
from synthdata.generators.cell_plate.design import CellPlateDesign, Phenotype
from synthdata.generators.cell_plate.layout import WellSlot
from synthdata.generators.cell_plate.qc import SiteDefect, debris_mask, qc_row
from synthdata.generators.cell_plate.well_effects import draw_well
from synthdata.generators.particle_plate.design import Illumination

NS_CELLS = 10
NS_CHANNEL = 11
NS_NOISE = 12
SUPERSAMPLE = marker.SUPERSAMPLE


@dataclass
class SiteResult:
    slot: WellSlot
    site: int  # 1-based
    channels: dict[str, np.ndarray]
    nuclei_mask: np.ndarray
    cell_mask: np.ndarray
    cell_rows: list[dict]
    nucleus_rows: list[dict]
    image_rows: list[dict]
    qc_rows: list[dict] = field(default_factory=list)


def downsample(fine: np.ndarray, ss: int = SUPERSAMPLE) -> np.ndarray:
    """Box average of ss x ss blocks."""
    h, w = fine.shape[0] // ss, fine.shape[1] // ss
    return fine.reshape(h, ss, w, ss).mean(axis=(1, 3))


def illumination_field(h: int, w: int, ill: Illumination) -> np.ndarray:
    """Multiplicative falloff: 1 at the centre, 1 - strength in the corners (radial) or at the far side (linear)."""
    if ill.kind == "none" or ill.strength == 0:
        return np.ones((h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    if ill.kind == "radial":
        cy, cx = (h - 1) / 2, (w - 1) / 2
        r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (cy**2 + cx**2)
        return 1 - ill.strength * r2
    t = math.radians(ill.angle)
    proj = xx * math.cos(t) + yy * math.sin(t)
    span = np.ptp(proj) or 1.0
    return 1 - ill.strength * (proj - proj.min()) / span


def _crop(cell, shape, ss, rng) -> marker.Crop | None:
    painted = _paint(cell.body_radius_fn, cell.x, cell.y, shape, ss)
    if painted is None:
        return None
    rs, cs, body, _ = painted
    yy = ((np.arange(rs.start, rs.stop) + 0.5) / ss - 0.5)[:, None]
    xx = ((np.arange(cs.start, cs.stop) + 0.5) / ss - 0.5)[None, :]
    nuclei = []
    for nu in cell.nuclei:
        dy, dx = yy - nu.y, xx - nu.x
        nuclei.append(np.hypot(dx, dy) <= nu.radius_fn(np.arctan2(dy, dx)))
    return marker.Crop(cell, ss, rs, cs, body, nuclei, rng)


def render_ideal(plan: SitePlan, design: CellPlateDesign, rng, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Noise-free channels on the fine grid: `shape` is the native (height, width); arrays are SUPERSAMPLE times larger.

    Draws only the channels the design declares. Writes each cell's drawn intensities and pattern counts into
    `cell.params`, for the truth rows. Cells are painted in their own crop; where bodies overlap, signals add.
    """
    ss = SUPERSAMPLE
    names = [c.name for c in design.plate.channels]
    unknown = [n for n in names if n not in marker.PATTERNS]
    if unknown:
        raise ValueError(f"cell_plate: no marker pattern for channel(s) {unknown}; known: {sorted(marker.PATTERNS)}")
    out = {n: np.zeros((shape[0] * ss, shape[1] * ss)) for n in names}
    for cell in plan.cells:
        crop = _crop(cell, shape, ss, rng)
        if crop is None:
            continue
        for n in names:
            intensity = max(float(design.base_cell.channels[n].intensity.sample(rng, 1)[0]), 0.0)
            cell.params[f"intensity_{n}"] = intensity
            out[n][crop.rs, crop.cs] += marker.PATTERNS[n](crop, intensity)
    return out


def _delta_columns(params: dict) -> dict:
    """Planted deltas from the neutral phenotype: factors report f - 1, adds report the add."""
    return {f"delta_{name}": float(params[name]) - (0.0 if name.endswith("_add") else 1.0)
            for name in Phenotype.model_fields if name in params}


def _lumpiness(fn) -> float:
    return float(sum(a for _, a, _ in fn.harmonics) / len(fn.harmonics))


def render_site(design: CellPlateDesign, slot: WellSlot, site: int, defect: SiteDefect | None = None) -> SiteResult:
    """Render one site. `defect` plants a blur or a clipped debris blob and is recorded in `qc_rows`."""
    spec, fx = design.plate.image, design.image_effects
    h, w = spec.height, spec.width
    rows, cols = design.plate.grid()
    outer = slot.is_outer(rows, cols)
    key = (slot.plate, slot.row, slot.col, site)

    well = draw_well(design, slot)
    plan = sample_cells(design, slot, site, stream(design.seed, NS_CELLS, *key), outer, well)
    ideal = render_ideal(plan, design, stream(design.seed, NS_CHANNEL, *key), (h, w))
    nuclei_mask, cell_mask = rasterise(plan, (h, w), supersample=1)

    rng = stream(design.seed, NS_NOISE, *key)
    field_ = illumination_field(h, w, fx.illumination)
    edge = fx.edge_effect.enabled and outer
    if edge:
        field_ = field_ * (1 - fx.edge_effect.dim)
    maxv = np.iinfo(spec.dtype).max
    debris = debris_mask(design, key) if defect and defect.kind == "saturated_debris" else None
    images, image_rows = {}, []
    base = {"plate": slot.plate_name, "well": slot.well, "site": site}
    for ci, ch in enumerate(design.plate.channels):
        sigma = ch.psf_sigma_px
        if defect and defect.kind == "blur":
            sigma = max(sigma, defect.psf_sigma_px)
        img = ndi.gaussian_filter(downsample(ideal[ch.name]), sigma, mode="nearest")
        if well is not None:
            img = img * well.gain[ch.name]
        background = float(fx.background.sample(rng, 1)[0])
        if design.plate_effects is None:
            img = (img + background) * field_
        else:
            eff = design.plate_effects[slot.plate]
            gain, shift = eff.gain.get(ch.name, 1.0), eff.background_shift.get(ch.name, 0.0)
            img = (img * gain + background + shift) * field_
        if fx.noise.poisson:
            img = rng.poisson(np.clip(img, 0, None)).astype(float)
        if fx.noise.gaussian_sd > 0:
            img = img + rng.normal(0.0, fx.noise.gaussian_sd, img.shape)
        if debris is not None and debris[0] == ci:
            img = np.where(debris[1], maxv + 1.0, img)
        clipped = int(((img > maxv) | (img < 0)).sum())
        images[ch.name] = np.clip(np.rint(img), 0, maxv).astype(spec.dtype)
        image_rows.append({**base, "channel": ch.name, "psf_sigma_px": sigma, "background": background,
                           "illumination": fx.illumination.kind, "illumination_strength": fx.illumination.strength,
                           "illumination_angle": float(fx.illumination.angle), "edge_effect": bool(edge),
                           "edge_dim": fx.edge_effect.dim if edge else 0.0,
                           "illumination_factor": float(field_.mean()), "clipped_px": clipped,
                           "n_cells": len(plan.cells), "n_cells_requested": plan.n_requested,
                           **({} if design.plate_effects is None
                              else {"plate_gain": float(gain), "plate_background_shift": float(shift)})})

    ident = {"compound": slot.compound, "dose_label": slot.dose_label, "condition": slot.condition,
             "role": slot.role, "phenotype": slot.phenotype or "", "dose_strength": float(slot.strength)}
    cell_rows, nucleus_rows = [], []
    for cell in plan.cells:
        p = cell.params
        extra = {k: v for k, v in p.items() if k not in Phenotype.model_fields and k != "nucleus_lumpiness"}
        cell_rows.append({**base, **ident, "cell_id": cell.cell_id, "nucleus_ids": [n.nucleus_id for n in cell.nuclei],
                          "x": cell.x, "y": cell.y, "body_diameter": cell.body_diameter,
                          "body_eccentricity": cell.body_eccentricity, "body_orientation": cell.body_orientation,
                          **extra, **{f"param_{k}": float(v) for k, v in p.items() if k in Phenotype.model_fields},
                          **_delta_columns(p)})
        for nu in cell.nuclei:
            nucleus_rows.append({**base, **ident, "nucleus_id": nu.nucleus_id, "cell_id": cell.cell_id, "x": nu.x,
                                 "y": nu.y, "diameter": nu.diameter, "eccentricity": nu.eccentricity,
                                 "orientation": nu.orientation, "lumpiness": _lumpiness(nu.radius_fn)})
    return SiteResult(slot, site, images, nuclei_mask, cell_mask, cell_rows, nucleus_rows, image_rows,
                      [qc_row(base, defect)])
