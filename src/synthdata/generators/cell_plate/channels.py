"""Marker patterns for the five Cell Painting channels, drawn per cell in a local crop of the fine grid.

Every pattern shares the cell's masks (nucleus, cytoplasm, body) and differs only in where the marker sits:
DNA fills the nucleus, ER hugs the nucleus, RNA is nucleoli plus a faint cytoplasm, AGP is cortex plus oriented
fibres, Mito is many short curved tubes. Coordinates are native pixels; arrays are on the fine grid
(`ss` fine pixels per native pixel). Values are in photons-equivalent counts before the PSF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from synthdata.generators.cell_plate.cells import CellSpec

SUPERSAMPLE = 2
MITO_LENGTH_PX = 6.0  # mean tube length of an unperturbed cell, native px
MITO_PX2_PER_TUBE = 30.0  # cytoplasm area per tube
FIBRE_PX2_PER_FIBRE = 60.0
NUCLEOLUS_SIGMA_PX = 1.4
FOCUS_SIGMA_PX = 1.0


@dataclass
class Crop:
    """One cell's window on the fine grid with its masks. `rng` is shared across the cell's channels."""

    cell: CellSpec
    ss: int
    rs: slice
    cs: slice
    body: np.ndarray
    nuclei: list[np.ndarray]
    rng: np.random.Generator
    nuc: np.ndarray = field(init=False)
    cyto: np.ndarray = field(init=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.nuc = np.any(self.nuclei, axis=0) if self.nuclei else np.zeros_like(self.body)
        self.cyto = self.body & ~self.nuc

    @property
    def shape(self):
        return self.body.shape

    @property
    def phen(self) -> dict:
        return self.cell.params

    def cached(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def dist_to_nucleus(self) -> np.ndarray:
        """Native px from each pixel to the nearest nucleus pixel."""
        return self.cached("dn", lambda: ndi.distance_transform_edt(~self.nuc) / self.ss
                           if self.nuc.any() else np.full(self.shape, 1e3))

    def dist_to_edge(self) -> np.ndarray:
        """Native px from each body pixel to the body outline."""
        return self.cached("de", lambda: ndi.distance_transform_edt(self.body) / self.ss)

    def texture(self, sigma_px: float) -> np.ndarray:
        """Zero-mean, unit-sd smooth noise with correlation length `sigma_px` native px."""
        ss = self.ss
        h, w = -(-self.shape[0] // ss), -(-self.shape[1] // ss)
        t = ndi.gaussian_filter(self.rng.normal(size=(h, w)), sigma_px, mode="wrap")
        t = (t - t.mean()) / (t.std() + 1e-12)
        return np.repeat(np.repeat(t, ss, 0), ss, 1)[:self.shape[0], :self.shape[1]]

    def blobs(self, points: np.ndarray, sigma_px: float) -> np.ndarray:
        """Unit-peak Gaussian blobs at fine-pixel (row, col) `points`."""
        out = np.zeros(self.shape)
        if len(points) == 0:
            return out
        np.add.at(out, (points[:, 0], points[:, 1]), 1.0)
        s = sigma_px * self.ss
        return ndi.gaussian_filter(out, s) * (2 * np.pi * s * s)

    def strokes(self, centres: np.ndarray, angles: np.ndarray, lengths: np.ndarray, curvature: float,
                sigma_px: float, mask: np.ndarray) -> np.ndarray:
        """Blurred polylines (fibres, tubes) with unit peak: centres are fine (row, col), lengths native px.

        Each stroke is a random walk in direction, `curvature` rad of angular step per segment, sampled at 24
        points and clipped to `mask`. Overlaps add.
        """
        n, k = len(centres), 24
        out = np.zeros(self.shape)
        if n == 0:
            return out
        steps = lengths[:, None] * self.ss / (k - 1)  # fine px per segment, shape (n, 1)
        turn = self.rng.normal(0.0, curvature, (n, k)) if curvature > 0 else np.zeros((n, k))
        theta = angles[:, None] + np.cumsum(turn, axis=1) - turn.sum(1, keepdims=True) / 2
        dy, dx = steps * np.sin(theta), steps * np.cos(theta)
        py = centres[:, :1] + np.cumsum(dy, 1) - dy.sum(1, keepdims=True) / 2
        px = centres[:, 1:] + np.cumsum(dx, 1) - dx.sum(1, keepdims=True) / 2
        iy, ix = np.rint(py).astype(int), np.rint(px).astype(int)
        ok = (iy >= 0) & (iy < self.shape[0]) & (ix >= 0) & (ix < self.shape[1])
        ok[ok] = mask[iy[ok], ix[ok]]
        w = np.broadcast_to(steps, (n, k))[ok]  # weight by segment length so density per px is uniform
        np.add.at(out, (iy[ok], ix[ok]), w)
        s = sigma_px * self.ss
        return ndi.gaussian_filter(out, s) * (s * np.sqrt(2 * np.pi))

    def nuclear_dim(self, level: float = 0.4, edge_px: float = 1.0) -> np.ndarray:
        """Widefield signal over the nucleus: `level` of the local cytoplasm level, with a soft edge (1 elsewhere)."""
        def make():
            soft = ndi.gaussian_filter(self.nuc.astype(float), edge_px * self.ss) if self.nuc.any() else self.nuc * 0.0
            return 1.0 - (1.0 - level) * np.clip(soft * 1.6, 0.0, 1.0)
        return self.cached(("dim", level, edge_px), make)

    def weighted_points(self, mask: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) == 0 or n <= 0:
            return np.zeros((0, 2), int)
        p = weights.ravel()[idx]
        pick = self.rng.choice(idx, size=n, p=p / p.sum())
        return np.column_stack(np.unravel_index(pick, self.shape))

    def random_points(self, mask: np.ndarray, n: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) == 0 or n <= 0:
            return np.zeros((0, 2), int)
        pick = idx[self.rng.integers(0, len(idx), n)]
        return np.column_stack(np.unravel_index(pick, self.shape))


def _stochastic_round(x: float, rng) -> int:
    return int(np.floor(max(x, 0.0) + rng.uniform()))


NUCLEOLI_MAX = 4
RETICULUM_WIDTH = 0.35  # zero-crossing band of the ER noise, in sd units


def _nucleoli(c: Crop) -> list[tuple[int, np.ndarray]]:
    """Per nucleus: the base nucleolus count (1 to 3) and candidate sites, shared by DNA (voids) and RNA (spots)."""
    def make():
        out = []
        for nuc in c.nuclei:
            inner = ndi.distance_transform_edt(nuc) / c.ss >= 3.0
            out.append((int(c.rng.integers(1, 4)), c.random_points(inner if inner.any() else nuc, NUCLEOLI_MAX)))
        return out
    return c.cached("nucleoli", make)


def dna(c: Crop, intensity: float) -> np.ndarray:
    """Hoechst: mottled chromatin with dim nucleolar voids and a dimmer rim, plus `dna_foci_add` bright foci."""
    amp = intensity * c.phen.get("dna_intensity_factor", 1.0)
    out = np.zeros(c.shape)
    sites = _nucleoli(c)
    for nuc, (k, pts) in zip(c.nuclei, sites):
        inward = np.minimum(ndi.distance_transform_edt(nuc) / c.ss / 3.0, 1.0)
        mottle = 1.0 + 0.12 * c.texture(1.0) + 0.1 * c.texture(2.5)
        voids = 1.0 - 0.3 * np.minimum(c.blobs(pts[:k], 1.3), 1.0)
        out += nuc * (0.7 + 0.3 * inward) * np.maximum(mottle, 0.3) * voids
    out *= amp
    n_foci = 0
    for nuc in c.nuclei:
        k = _stochastic_round(c.phen.get("dna_foci_add", 0.0), c.rng)
        n_foci += k
        deep = ndi.binary_erosion(nuc, iterations=max(int(c.ss), 1)) if nuc.sum() > 40 else nuc
        out += 1.4 * amp * c.blobs(c.random_points(deep, k), FOCUS_SIGMA_PX) * nuc
    c.phen["n_foci"] = n_foci
    return out


def er(c: Crop, intensity: float) -> np.ndarray:
    """Concanavalin A: reticular cytoplasm, brightest next to the nucleus, fading toward the edge."""
    length = max(0.2 * c.cell.body_diameter, 2.0)
    profile = 0.12 + 0.88 * np.exp(-c.dist_to_nucleus() / length)
    net = np.exp(-(c.texture(1.0) ** 2) / (2 * RETICULUM_WIDTH**2))  # thin lines where smooth noise crosses zero
    tex = (0.35 + 0.65 * net) / 0.56 * np.maximum(1.0 + 0.15 * c.texture(2.5), 0.0)
    return intensity * c.body * profile * tex * c.nuclear_dim()


def rna(c: Crop, intensity: float) -> np.ndarray:
    """SYTO 14: 1 to 3 bright nucleoli per nucleus on a faint nucleoplasm, with weak cytoplasm."""
    amp = intensity * c.phen.get("rna_intensity_factor", 1.0)
    factor = c.phen.get("nucleoli_count_factor", 1.0)
    cyto = 0.15 * c.cyto * np.maximum(1.0 + 0.2 * c.texture(1.5), 0.0)
    nucleoplasm = 0.2 * c.nuc
    spots = np.zeros(c.shape)
    total = 0
    for nuc, (base_k, pts) in zip(c.nuclei, _nucleoli(c)):
        k = min(_stochastic_round(base_k * factor, c.rng), len(pts))
        total += k
        spots += c.blobs(pts[:k], NUCLEOLUS_SIGMA_PX) * nuc
    c.phen["n_nucleoli"] = total
    return amp * (cyto + nucleoplasm + 2.8 * spots)


def agp(c: Crop, intensity: float) -> np.ndarray:
    """Phalloidin + WGA: diffuse cytoplasm with fibres of varied orientation and length, and a thin uneven cortex.

    Each cell has its own fibre spread, a share of fibres in a second (crossing) direction, and a cortex whose
    brightness varies around the edge. The fibre factor scales contrast around a fixed mean, so 0.1 dissolves
    the fibres into diffuse signal.
    """
    ff = c.phen.get("agp_fibre_factor", 1.0)
    rng = c.rng
    n = int(round(c.body.sum() / c.ss**2 / FIBRE_PX2_PER_FIBRE))
    length = c.cell.body_diameter * rng.uniform(0.2, 1.1, n)
    spread = rng.uniform(0.2, 0.8)
    angles = c.cell.body_orientation + rng.normal(0.0, spread, n)
    cross = rng.uniform(size=n) < rng.uniform(0.1, 0.35)
    angles[cross] += rng.choice([-1.0, 1.0]) * rng.uniform(0.7, 1.5)
    fibres = c.strokes(c.random_points(c.body, n), angles, length, 0.12, 0.7, c.body)
    inside = fibres[c.body]
    fibres = fibres / inside.mean() if inside.size and inside.mean() > 0 else fibres
    cortex = np.exp(-(np.maximum(c.dist_to_edge() - 0.5 / c.ss, 0.0) ** 2) / (2 * 0.8**2)) * c.body
    h, w = c.shape
    cy, cx = c.cell.y * c.ss - c.rs.start, c.cell.x * c.ss - c.cs.start
    ang = np.arctan2(np.arange(h)[:, None] - cy, np.arange(w)[None, :] - cx)
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    around = np.clip(1.0 + 0.5 * np.cos(2 * ang + p1) + 0.35 * np.cos(3 * ang + p2) + 0.2 * c.texture(2.0), 0.15, None)
    cortex_weight = 0.9 * (0.4 + 0.6 * min(ff, 1.0))
    cyto = 0.45 + 0.35 * ff * (fibres - 1.0) + 0.1 * c.texture(2.0)
    return intensity * (c.body * np.maximum(cyto, 0.0) * c.nuclear_dim() + cortex_weight * cortex * around)


def mito(c: Crop, intensity: float) -> np.ndarray:
    """MitoTracker: short curved tubes, concentrated around the nucleus. Fragmentation shortens and multiplies them."""
    length_f = c.phen.get("mito_length_factor", 1.0)
    count_f = c.phen.get("mito_count_factor", 1.0)
    n = int(round(c.cyto.sum() / c.ss**2 / MITO_PX2_PER_TUBE * count_f))
    lengths = np.maximum(MITO_LENGTH_PX * length_f * c.rng.lognormal(0.0, 0.3, n), 1.0)
    angles = c.rng.uniform(0, 2 * np.pi, n)
    reach = 0.18 * c.cell.body_diameter + 3.0
    starts = c.weighted_points(c.cyto, 0.15 + np.exp(-c.dist_to_nucleus() / reach), n)
    tubes = c.strokes(starts, angles, lengths, 0.35, 0.6, c.body)
    c.phen["n_mito_tubes"] = n
    c.phen["mito_length_px"] = float(lengths.mean()) if n else 0.0
    haze = 0.08 * c.body * np.maximum(1 + 0.3 * c.texture(2.0), 0.0)
    return intensity * (haze + 0.9 * np.minimum(tubes, 2.0) * c.body) * c.nuclear_dim()


PATTERNS = {"DNA": dna, "ER": er, "RNA": rna, "AGP": agp, "Mito": mito}
