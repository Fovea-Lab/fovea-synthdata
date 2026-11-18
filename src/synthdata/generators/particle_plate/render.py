"""Render one (plate, well, site) image: particles → pixels, a label mask and per-particle truth."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from synthdata.core.rng import stream
from synthdata.generators.particle_plate.design import Arrangement, Illumination, ParticlePlateDesign, Profile
from synthdata.generators.particle_plate.layout import WellSlot
from synthdata.generators.particle_plate.qc import NS_QC, label_for, thresholds_for

NS_IMAGE = 1
SUPERSAMPLE = 4
MIN_DIAMETER = 2.0
MAX_ECCENTRICITY = 0.99
HARD_EDGE = 0.05  # edge_softness below this (px) renders a hard, anti-aliased edge


@dataclass
class ParticleSet:
    y: np.ndarray
    x: np.ndarray
    diameter: np.ndarray
    eccentricity: np.ndarray
    orientation: np.ndarray  # degrees, major axis from +x toward +y
    intensity: np.ndarray
    edge_softness: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Raster:
    signal: np.ndarray
    mask: np.ndarray
    cover: np.ndarray
    boxes: list = field(default_factory=list)


@dataclass
class RenderResult:
    slot: WellSlot
    site: int  # 1-based
    image: np.ndarray
    mask: np.ndarray
    particles: list[dict]
    image_row: dict
    qc_row: dict = field(default_factory=dict)


def semi_axes(d: float, e: float) -> tuple[float, float]:
    """Semi-axes of an ellipse with equivalent diameter d and eccentricity e (a·b = (d/2)²)."""
    a = (d / 2) / (1 - e * e) ** 0.25
    return a, a * math.sqrt(1 - e * e)


def place(arr: Arrangement, n: int, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Particle centres as an (n, 2) array of (y, x) pixel coordinates."""
    if arr.kind == "uniform":
        return np.column_stack([rng.uniform(-0.5, h - 0.5, n), rng.uniform(-0.5, w - 0.5, n)])
    parents = np.column_stack([rng.uniform(-0.5, h - 0.5, arr.clusters), rng.uniform(-0.5, w - 0.5, arr.clusters)])
    which = rng.integers(0, arr.clusters, n)
    pts = parents[which] + rng.normal(0.0, arr.spread, (n, 2))
    pts[:, 0] = np.mod(pts[:, 0] + 0.5, h) - 0.5  # wrap around (torus) so clusters never pile on the border
    pts[:, 1] = np.mod(pts[:, 1] + 0.5, w) - 0.5
    return pts


def _rotate(ys, xs, cy, cx, theta_deg):
    t = math.radians(theta_deg)
    dy, dx = ys - cy, xs - cx
    return dx * math.cos(t) + dy * math.sin(t), -dx * math.sin(t) + dy * math.cos(t)


def _profile(rho2: np.ndarray, profile: Profile) -> np.ndarray:
    if profile.kind == "flat":
        return np.ones_like(rho2)
    if profile.kind == "gaussian":
        return np.exp(-rho2 / (2 * profile.sigma_frac**2))
    half = profile.width_frac / 2  # ring: brightest just inside the rim
    return np.exp(-((np.sqrt(rho2) - (1 - half)) ** 2) / (2 * half**2))


def rasterise(h: int, w: int, ps: ParticleSet, profile: Profile) -> Raster:
    signal = np.zeros((h, w))
    mask = np.zeros((h, w), dtype=np.uint32)
    best = np.full((h, w), np.inf)
    cover = np.zeros((h, w), dtype=np.int32)
    boxes = []
    off = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5
    for i in range(len(ps)):
        cy, cx, th = float(ps.y[i]), float(ps.x[i]), float(ps.orientation[i])
        soft = float(ps.edge_softness[i])
        a, b = semi_axes(float(ps.diameter[i]), float(ps.eccentricity[i]))
        pad = a + 4 * soft + 2
        y0, y1 = max(0, math.floor(cy - pad)), min(h, math.ceil(cy + pad) + 1)
        x0, x1 = max(0, math.floor(cx - pad)), min(w, math.ceil(cx + pad) + 1)
        if y1 <= y0 or x1 <= x0:
            boxes.append((0, 0, 0, 0, np.zeros((0, 0), dtype=bool)))
            continue
        bh, bw = y1 - y0, x1 - x0
        # supersampled signal
        ys = (np.arange(y0, y1)[:, None] + off[None, :]).ravel()
        xs = (np.arange(x0, x1)[:, None] + off[None, :]).ravel()
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        u, v = _rotate(Y, X, cy, cx, th)
        rho2 = (u / a) ** 2 + (v / b) ** 2
        if soft < HARD_EDGE:
            cov = (rho2 <= 1).astype(float)
        else:
            grad = 2 * np.sqrt((u / a**2) ** 2 + (v / b**2) ** 2)
            rho = np.sqrt(rho2)
            s = (rho - 1) * 2 * rho / np.maximum(grad, 1e-9)  # ≈ signed distance to the edge, px (exact for a circle)
            s = np.where(rho2 == 0, -1e9, s)  # at the exact centre the gradient vanishes: treat as deep inside
            cov = 0.5 * (1 + np.tanh(-s / (2 * soft)))
        val = (cov * _profile(rho2, profile)).reshape(bh, SUPERSAMPLE, bw, SUPERSAMPLE).mean(axis=(1, 3))
        signal[y0:y1, x0:x1] += val * float(ps.intensity[i])
        # hard mask at pixel centres; overlaps go to the nearer centre
        py, px = np.mgrid[y0:y1, x0:x1].astype(float)
        u, v = _rotate(py, px, cy, cx, th)
        hard = (u / a) ** 2 + (v / b) ** 2 <= 1
        dist2 = (py - cy) ** 2 + (px - cx) ** 2
        best_view, mask_view = best[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        win = hard & (dist2 < best_view)
        mask_view[win] = i + 1
        best_view[win] = dist2[win]
        cover[y0:y1, x0:x1] += hard
        boxes.append((y0, y1, x0, x1, hard))
    return Raster(signal, mask, cover, boxes)


def _touching_ids(mask: np.ndarray) -> set[int]:
    ids: set[int] = set()
    pairs = [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :]),
             (np.s_[:-1, :-1], np.s_[1:, 1:]), (np.s_[:-1, 1:], np.s_[1:, :-1])]
    for sa, sb in pairs:
        a, b = mask[sa], mask[sb]
        m = (a > 0) & (b > 0) & (a != b)
        ids.update(np.unique(a[m]).tolist())
        ids.update(np.unique(b[m]).tolist())
    return ids


def particle_flags(ps: ParticleSet, raster: Raster, h: int, w: int) -> list[dict]:
    touch = _touching_ids(raster.mask)
    visible = np.bincount(raster.mask.ravel(), minlength=len(ps) + 1)
    out = []
    for i in range(len(ps)):
        a, b = semi_axes(float(ps.diameter[i]), float(ps.eccentricity[i]))
        t = math.radians(float(ps.orientation[i]))
        ex = math.sqrt((a * math.cos(t)) ** 2 + (b * math.sin(t)) ** 2)
        ey = math.sqrt((a * math.sin(t)) ** 2 + (b * math.cos(t)) ** 2)
        cy, cx = float(ps.y[i]), float(ps.x[i])
        clipped = cx - ex < -0.5 or cx + ex > w - 0.5 or cy - ey < -0.5 or cy + ey > h - 0.5
        y0, y1, x0, x1, hard = raster.boxes[i]
        ov = float((raster.cover[y0:y1, x0:x1][hard] > 1).mean()) if hard.any() else 0.0
        out.append({
            "clipped": bool(clipped),
            "touching": bool((i + 1) in touch or ov > 0),
            "overlap_frac": ov,
            "visible_px": int(visible[i + 1]),
        })
    return out


def illumination_field(h: int, w: int, ill: Illumination) -> np.ndarray:
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


def render_image(design: ParticlePlateDesign, slot: WellSlot, site: int, degree: float = 0.0) -> RenderResult:
    part, fx = design.particle_for(slot.condition), design.effects_for(slot.condition)
    spec = design.plate.image
    h, w = spec.height, spec.width
    rng = stream(design.seed, NS_IMAGE, slot.plate, slot.row, slot.col, site)
    # Draw order is part of the determinism contract: count, centres, parameters, background, noise.
    n = int(max(0, round(float(part.count.sample(rng, 1)[0]))))
    centres = place(part.arrangement, n, h, w, rng)
    # QC aggregation defect: its own stream; untouched (no draws anywhere) when degree is 0.
    clustered = np.zeros(n, dtype=bool)
    qc_clusters, qc_spread = 0, 0.0
    agg = design.qc_defects.aggregation if design.qc_defects else None
    if agg is not None and degree > 0 and n > 0:
        qc_clusters, qc_spread = float("nan"), float("nan")
        k = round(degree * n)
        if k > 0:  # k == 0: nothing to move; the QC stream is not consumed and clusters/spread stay NaN
            rq = stream(design.seed, NS_QC, 1, slot.plate, slot.row, slot.col, site)
            idx = np.sort(rq.choice(n, k, replace=False))
            qc_clusters = max(1, round(float(agg.clusters.sample(rq, 1)[0])))
            qc_spread = max(1e-3, float(agg.spread.sample(rq, 1)[0]))
            centres = centres.copy()
            centres[idx] = place(Arrangement(kind="clumped", clusters=qc_clusters, spread=qc_spread), k, h, w, rq)
            clustered[idx] = True
    d = part.diameter.sample(rng, n)
    e = part.eccentricity.sample(rng, n)
    th = np.mod(part.orientation.sample(rng, n), 180.0)
    inten = part.intensity.sample(rng, n)
    soft = part.edge_softness.sample(rng, n)
    clamped = int((d < MIN_DIAMETER).sum() + (e < 0).sum() + (e > MAX_ECCENTRICITY).sum()
                  + (inten < 0).sum() + (soft < 0).sum())
    ps = ParticleSet(centres[:, 0], centres[:, 1], np.maximum(d, MIN_DIAMETER),
                     np.clip(e, 0.0, MAX_ECCENTRICITY), th, np.maximum(inten, 0.0), np.maximum(soft, 0.0))
    background = float(fx.background.sample(rng, 1)[0])

    raster = rasterise(h, w, ps, part.profile)
    flags = particle_flags(ps, raster, h, w)

    field_ = illumination_field(h, w, fx.illumination)
    edge = fx.edge_effect.enabled and slot.is_outer(*design.plate.grid())
    if edge:
        field_ = field_ * (1 - fx.edge_effect.dim)
    img = (background + raster.signal) * field_
    if fx.noise.poisson:
        img = rng.poisson(np.clip(img, 0, None)).astype(float)
    if fx.noise.gaussian_sd > 0:
        img = img + rng.normal(0.0, fx.noise.gaussian_sd, img.shape)
    maxv = np.iinfo(spec.dtype).max
    clipped_px = int(((img > maxv) | (img < 0)).sum())  # both low-side and high-side clipped pixels
    image = np.clip(np.rint(img), 0, maxv).astype(spec.dtype)

    base = {"plate": slot.plate_name, "well": slot.well, "site": site, "condition": slot.condition}
    particles = [
        {**base, "particle_id": i + 1, "x": float(ps.x[i]), "y": float(ps.y[i]),
         "diameter": float(ps.diameter[i]), "eccentricity": float(ps.eccentricity[i]),
         "orientation": float(ps.orientation[i]), "intensity": float(ps.intensity[i]),
         "profile": part.profile.kind, "edge_softness": float(ps.edge_softness[i]), **flags[i],
         **({"qc_clustered": bool(clustered[i])} if design.qc_defects else {})}
        for i in range(n)
    ]
    image_row = {**base, "n_particles": n, "background": background, "illumination": fx.illumination.kind,
                 "illumination_strength": fx.illumination.strength,
                 "illumination_angle": float(fx.illumination.angle), "edge_effect": bool(edge),
                 "clipped_px": clipped_px, "clamped_params": clamped}
    n_cl = int(clustered.sum())
    qc_row = {**base, "degree": float(degree), "label": label_for(degree, thresholds_for(design)),
              "n_clustered": n_cl, "clusters": float(qc_clusters), "spread": float(qc_spread),
              "degree_clipped": False}
    return RenderResult(slot, site, image, raster.mask, particles, image_row, qc_row)
