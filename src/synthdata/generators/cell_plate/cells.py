"""Cell geometry and placement: lumpy bodies with lumpy nuclei, rejection-sampled onto a site.

Positions are in pixel units with pixel (row, col) centred at (y, x) = (row, col).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from synthdata.generators.cell_plate.design import CellPlateDesign, Phenotype
from synthdata.generators.cell_plate.layout import WellSlot
from synthdata.generators.cell_plate.well_effects import WellDraw, responders

HARMONICS = (2, 3, 4, 5)
BODY_HARMONICS = (2, 3, 4, 5, 6, 7, 8)  # bodies carry higher orders than nuclei, for an irregular outline
MAX_PROTRUSIONS = 3
BODY_ATTEMPTS = 30  # position draws per cell
MAX_RESPONSE_STRENGTH = 1.0  # the well response multiplier cannot push a main phenotype past strength 1
MAX_CONSECUTIVE_FAILURES = 15  # stop placing once this many cells in a row cannot be placed
_THETA = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)


@dataclass(frozen=True)
class RadiusFn:
    """Polar radius of a lumpy ellipse: an area-equivalent ellipse times `1 + sum a_k cos(k t + phi_k)`.

    A frozen dataclass rather than a closure, so it pickles across processes.
    """

    diameter: float
    eccentricity: float
    orientation: float
    harmonics: tuple[tuple[int, float, float], ...]  # (k, a_k, phi_k)
    bumps: tuple[tuple[float, float, float], ...] = ()  # short protrusions: (angle, relative height, angular width)

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        e2 = self.eccentricity**2
        a = (self.diameter / 2) / (1 - e2) ** 0.25
        b = a * np.sqrt(1 - e2)
        t = theta - self.orientation
        base = a * b / np.sqrt((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2)
        mod = 1.0 + sum(ak * np.cos(k * theta + phi) for k, ak, phi in self.harmonics)
        for ang, height, width in self.bumps:
            mod = mod + height * np.exp(-0.5 * (np.angle(np.exp(1j * (theta - ang))) / width) ** 2)
        return base * np.maximum(mod, 0.2)


@dataclass(frozen=True)
class NucleusSpec:
    nucleus_id: int  # unique across the site; the label written by `rasterise`
    x: float
    y: float
    diameter: float
    eccentricity: float
    orientation: float
    radius_fn: RadiusFn

    @property
    def bounding_radius(self) -> float:
        return float(self.radius_fn(_THETA).max())


@dataclass
class CellSpec:
    cell_id: int  # 1-based; the label written to `cell_labels`
    x: float
    y: float
    body_radius_fn: RadiusFn
    nuclei: list[NucleusSpec]
    body_diameter: float
    body_eccentricity: float
    body_orientation: float
    params: dict = field(default_factory=dict)


@dataclass
class SitePlan:
    cells: list[CellSpec]
    n_requested: int
    n_placed: int


def _harmonics(rng, lumpiness: float, orders: tuple[int, ...] = HARMONICS) -> tuple[tuple[int, float, float], ...]:
    return tuple((k, float(lumpiness * rng.uniform(0.5, 1.0) / (k - 1)), float(rng.uniform(0, 2 * np.pi)))
                 for k in orders)


def _bumps(rng, lumpiness: float) -> tuple[tuple[float, float, float], ...]:
    """0 to 3 short protrusions, scaled by body lumpiness (none for a smooth body)."""
    if lumpiness <= 0:
        return ()
    k = int(rng.integers(0, MAX_PROTRUSIONS + 1))
    return tuple((float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0.15, 0.35) * min(lumpiness / 0.15, 1.0)), float(rng.uniform(0.15, 0.3)))
                 for _ in range(k))


def _fit_factor(body: RadiusFn, bx: float, by: float, nuc: RadiusFn, nx: float, ny: float) -> float:
    """>= 1 when the nucleus outline lies inside 95% of the body outline."""
    px = nx + nuc(_THETA) * np.cos(_THETA) - bx
    py = ny + nuc(_THETA) * np.sin(_THETA) - by
    return float(np.min(0.95 * body(np.arctan2(py, px)) / np.hypot(px, py)))


def _shrink_until_fits(body, bx, by, nuc: RadiusFn, nx, ny, tries=30):
    for _ in range(tries):
        if _fit_factor(body, bx, by, nuc, nx, ny) >= 1.0:
            return nuc
        nuc = replace(nuc, diameter=nuc.diameter * 0.92)
    return None


def _make_nuclei(rng, design, body: RadiusFn, bx, by, offset, phen: Phenotype, n_nuclei, next_id,
                 size: float = 1.0) -> list[NucleusSpec]:
    base = design.base_cell.nucleus
    out: list[NucleusSpec] = []

    def draw_fn(diam):
        ecc = float(np.clip(base.eccentricity.sample(rng, 1)[0], 0.0, 0.9))
        lump = max(float(base.lumpiness.sample(rng, 1)[0]), 0.0)
        return RadiusFn(diam, ecc, float(rng.uniform(0, np.pi)), _harmonics(rng, lump))

    for i in range(n_nuclei):
        diam = max(float(base.diameter.sample(rng, 1)[0]) * phen.nucleus_diameter_factor * size, 3.0)
        fn = draw_fn(diam)
        placed = None
        if i == 0:
            phi = rng.uniform(0, 2 * np.pi)
            for off in (offset, 0.0):
                nx, ny = bx + off * body(phi) * np.cos(phi), by + off * body(phi) * np.sin(phi)
                fitted = _shrink_until_fits(body, bx, by, fn, nx, ny)
                if fitted is not None:
                    placed = (fitted, nx, ny)
                    break
        else:
            for attempt in range(200):
                if attempt and attempt % 25 == 0:
                    fn = replace(fn, diameter=fn.diameter * 0.85)
                phi = rng.uniform(0, 2 * np.pi)
                rho = rng.uniform(0.2, 0.6) * body(phi)
                nx, ny = bx + rho * np.cos(phi), by + rho * np.sin(phi)
                if _fit_factor(body, bx, by, fn, nx, ny) < 1.0:
                    continue
                if all(np.hypot(nx - o.x, ny - o.y) >= fn(_THETA).max() + o.bounding_radius + 1 for o in out):
                    placed = (fn, nx, ny)
                    break
        if placed is None:
            return []  # a requested nucleus did not fit: the caller redraws this cell's position
        fn, nx, ny = placed
        out.append(NucleusSpec(next_id + len(out), float(nx), float(ny), fn.diameter, fn.eccentricity,
                               fn.orientation, fn))
    return out


def sample_cells(design: CellPlateDesign, slot: WellSlot, site: int, rng, outer: bool,
                 well: WellDraw | None = None) -> SitePlan:
    """Draw and place the cells of one site. A blank well has none. Warns when not all can be placed.

    `well` carries the well's random draws (`design.well_effects`); None leaves every value as the design gives it.
    With a responder fraction, each cell takes the main phenotype or not; the side effect reaches every cell."""
    base = design.base_cell
    strength = slot.strength
    if well is not None and well.response_factor is not None:
        # capped at full strength: a well's response never goes past the planted phenotype
        strength = min(slot.strength * well.response_factor, MAX_RESPONSE_STRENGTH)
    phen = Phenotype() if slot.phenotype is None else design.phenotypes[slot.phenotype].at(strength)
    still = Phenotype()  # a cell that does not respond: the side effect alone
    if slot.side_effect:
        side = design.phenotypes[slot.side_effect].at(slot.side_strength)
        phen, still = phen.combined(side), side
    partial = slot.responder_fraction is not None
    record = slot.phenotype is not None and any(c.responder_fraction is not None for c in design.compounds.values())
    if slot.role == "blank":
        return SitePlan([], 0, 0)

    if partial:
        p = slot.responder_fraction
        factor = p * phen.count_factor + (1 - p) * still.count_factor
    else:
        factor = phen.count_factor
    edge = design.image_effects.edge_effect
    if outer and edge.enabled:
        factor *= edge.count_factor
    if design.plate_effects is not None:
        factor *= design.plate_effects[slot.plate].count_factor
    if well is not None:
        factor *= well.count_factor
    n = max(int(round(float(base.count.sample(rng, 1)[0]) * factor)), 0)
    responds = responders(design, slot, site, n) if partial else np.ones(n, bool)

    def per_cell(name):
        return np.where(responds, getattr(phen, name), getattr(still, name)) if partial else getattr(phen, name)

    size = 1.0 if well is None else well.size_factor
    height, width = design.plate.image.height, design.plate.image.width
    body_d = np.maximum(base.body.diameter.sample(rng, n) * per_cell("body_diameter_factor") * size, 4.0)
    body_e = np.clip(base.body.eccentricity.sample(rng, n) + per_cell("body_eccentricity_add"), 0.0, 0.92)
    body_l = np.maximum(base.body.lumpiness.sample(rng, n) * per_cell("body_lumpiness_factor"), 0.0)
    offsets = np.clip(base.body.nucleus_offset.sample(rng, n), 0.0, 0.8)
    kinds = {True: (phen, phen.model_dump(), float(np.clip(phen.multinucleate_add, 0.0, 1.0))),
             False: (still, still.model_dump(), float(np.clip(still.multinucleate_add, 0.0, 1.0)))}

    px, py, pr = np.zeros(n), np.zeros(n), np.zeros(n)  # placed bodies: centre and equivalent radius
    nuc_x, nuc_y, nuc_r = [], [], []  # placed nuclei, for the global non-overlap check
    cells: list[CellSpec] = []
    next_nucleus = 1
    failures = 0
    for i in range(n):
        cell_phen, resolved, p_multi = kinds[bool(responds[i])]
        r = body_d[i] / 2
        body_fn = RadiusFn(float(body_d[i]), float(body_e[i]), float(rng.uniform(0, np.pi)),
                           _harmonics(rng, float(body_l[i]), BODY_HARMONICS), _bumps(rng, float(body_l[i])))
        m = len(cells)
        n_nuclei = (2 + int(rng.integers(2))) if rng.uniform() < p_multi else 1  # fixed across attempts
        accepted = None
        for _ in range(BODY_ATTEMPTS):
            x, y = rng.uniform(0, width), rng.uniform(0, height)
            if m and np.any(np.hypot(px[:m] - x, py[:m] - y)
                            < pr[:m] + r - 2 * base.overlap_fraction * np.minimum(pr[:m], r)):
                continue
            nuclei = _make_nuclei(rng, design, body_fn, x, y, float(offsets[i]), cell_phen, n_nuclei,
                                  next_nucleus, size)
            if not nuclei or any(not (0 <= nu.x <= width - 1 and 0 <= nu.y <= height - 1) for nu in nuclei):
                continue  # every nucleus centre must land on a pixel, so no nucleus vanishes from the labels
            if nuc_x and any(np.any(np.hypot(np.array(nuc_x) - nu.x, np.array(nuc_y) - nu.y)
                                    < np.array(nuc_r) + nu.bounding_radius + 1) for nu in nuclei):
                continue
            accepted = (x, y, nuclei)
            break
        if accepted is None:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                break
            continue
        failures = 0
        x, y, nuclei = accepted
        px[m], py[m], pr[m] = x, y, r
        for nu in nuclei:
            nuc_x.append(nu.x)
            nuc_y.append(nu.y)
            nuc_r.append(nu.bounding_radius)
        next_nucleus += len(nuclei)
        params = {**resolved, "body_lumpiness": float(body_l[i]), "nucleus_offset": float(offsets[i]),
                  "n_nuclei": len(nuclei), "multinucleate": len(nuclei) > 1,
                  "nucleus_lumpiness": [sum(a for _, a, _ in nu.radius_fn.harmonics) / len(HARMONICS)
                                        for nu in nuclei]}
        if record:  # only when the design uses partial response, so other designs keep their truth columns
            params["responder"] = bool(responds[i])
        cells.append(CellSpec(m + 1, float(x), float(y), body_fn, nuclei, float(body_d[i]), float(body_e[i]),
                              body_fn.orientation, params))

    if len(cells) < n:
        warnings.warn(f"cell_plate: placed {len(cells)} of {n} requested cells in {slot.well} site {site}; "
                      "the design is too dense for the image", UserWarning, stacklevel=2)
    return SitePlan(cells, n, len(cells))


def _paint(fn: RadiusFn, cx, cy, shape, ss):
    """Boolean mask window of a shape on the fine grid: (rows slice, cols slice, inside mask, dist2)."""
    reach = float(fn(_THETA).max()) * 1.05 + 2
    r0, r1 = max(int(np.floor((cy - reach) * ss)), 0), min(int(np.ceil((cy + reach) * ss)) + 1, shape[0] * ss)
    c0, c1 = max(int(np.floor((cx - reach) * ss)), 0), min(int(np.ceil((cx + reach) * ss)) + 1, shape[1] * ss)
    if r0 >= r1 or c0 >= c1:
        return None
    yy = (np.arange(r0, r1) + 0.5) / ss - 0.5 - cy
    xx = (np.arange(c0, c1) + 0.5) / ss - 0.5 - cx
    dy, dx = np.meshgrid(yy, xx, indexing="ij")
    d2 = dx * dx + dy * dy
    inside = np.sqrt(d2) <= fn(np.arctan2(dy, dx))
    return slice(r0, r1), slice(c0, c1), inside, d2


def rasterise(plan: SitePlan, shape: tuple[int, int], supersample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Label images (uint32): nuclei carry `nucleus_id`, cell pixels carry `cell_id`.

    Where bodies overlap, the cell whose centre is nearer owns the pixel. A nucleus pixel always belongs to
    its own cell. With supersample > 1 the shapes are drawn on a finer grid and each pixel takes the majority
    label of its block, so the labels stay aligned with the native grid.
    """
    ss = max(int(supersample), 1)
    fine = (shape[0] * ss, shape[1] * ss)
    nuclei = np.zeros(fine, np.uint32)
    cells = np.zeros(fine, np.uint32)
    best = np.full(fine, np.inf)
    for cell in plan.cells:
        painted = _paint(cell.body_radius_fn, cell.x, cell.y, shape, ss)
        if painted is None:
            continue
        rs, cs, inside, d2 = painted
        win = best[rs, cs]
        take = inside & (d2 < win)
        win[take] = d2[take]
        cells[rs, cs][take] = cell.cell_id
    for cell in plan.cells:  # second pass, so no later body can overwrite a nucleus pixel
        for nu in cell.nuclei:
            painted = _paint(nu.radius_fn, nu.x, nu.y, shape, ss)
            if painted is None:
                continue
            rs, cs, inside, _ = painted
            nuclei[rs, cs][inside] = nu.nucleus_id
            cells[rs, cs][inside] = cell.cell_id
    if ss > 1:
        nuclei, cells = _block_mode(nuclei, ss), _block_mode(cells, ss)
    return np.ascontiguousarray(nuclei), np.ascontiguousarray(cells)


def _block_mode(labels: np.ndarray, ss: int) -> np.ndarray:
    """Most common label in each ss x ss block (ties go to the first in raster order)."""
    h, w = labels.shape[0] // ss, labels.shape[1] // ss
    blocks = labels.reshape(h, ss, w, ss).transpose(0, 2, 1, 3).reshape(h, w, ss * ss)
    votes = np.stack([(blocks == blocks[..., k:k + 1]).sum(-1) for k in range(ss * ss)], axis=-1)
    return np.take_along_axis(blocks, votes.argmax(-1)[..., None], axis=-1)[..., 0]
