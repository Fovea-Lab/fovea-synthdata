"""Recoverability self-check for cell plates, and the naive segmenter a platform pipeline must match.

Stage 1 compares the written masks with the truth tables (is the dataset intact?). Stage 2 segments the
images with `segment_site` and asserts the design's `expectations` (does a simple pipeline see the planted
differences?). This file imports nothing from platemeasure; the dependency runs the other way.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_multiotsu, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from synthdata.generators.cell_plate.design import load_design
from synthdata.generators.cell_plate.reader import list_sites, read_site, read_truth_masks

CELL_CHANNELS = ("AGP", "ER")

# Segmenter parameters. A pipeline that must match `segment_site` object for object copies every one of these.
SIGMA = 1.0                  # Gaussian smoothing of DNA and of the cell image, px
NOISE_K = 6.0                # noise floor: multiples of the smoothed noise SD above the background
MIN_DISTANCE = 8             # nuclei split: minimum distance between distance-transform peaks, px
MIN_AREA = 60                # nuclei smaller than this, px, are dropped
COMPACTNESS = 0.01           # cell watershed compactness
FLOOR_BG_PERCENTILE = 10.0   # noise floor: background level is this percentile of the smoothed image
FLOOR_LOW_PERCENTILE = 1.0   # noise floor: spread allowance is (background - this percentile)
FLOOR_SPREAD_MULT = 2.0      # noise floor: multiples of that spread added
NUCLEI_OTSU_CLASSES = 2      # nuclei threshold: ordinary Otsu
CELL_OTSU_CLASSES = 3        # cell threshold: lowest level of a 3-class multi-Otsu
OTSU_BINS = 256              # histogram bins for both Otsu variants
NORM_LOW_PERCENTILE = 1.0    # per-channel normalisation of the cell image: this percentile maps to 0
NORM_HIGH_PERCENTILE = 99.5  # and this one to 1, clipped
SUPPORTED = ("Image_Count_Cells", "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", "Cells_AreaShape_Eccentricity")


# ---------- the naive segmenter ----------

def noise_floor(smoothed: np.ndarray, raw: np.ndarray, sigma: float, noise_k: float) -> float:
    """A level that background and noise alone do not reach.

    Background is the FLOOR_BG_PERCENTILE of the smoothed image, not the median: cells can cover more than half
    of the frame, and then the median is cell signal. The floor adds `noise_k` times the smoothed noise SD
    (estimated from neighbour differences of the raw image, scaled by the Gaussian's noise reduction) and
    FLOOR_SPREAD_MULT times the spread of the background below it (that percentile minus FLOOR_LOW_PERCENTILE)
    as an allowance for a smooth illumination gradient.
    """
    low, bg = (float(v) for v in np.percentile(smoothed, (FLOOR_LOW_PERCENTILE, FLOOR_BG_PERCENTILE)))
    d = np.abs(np.diff(raw.astype(float), axis=1))
    sd_raw = float(np.median(d)) / 0.6745 / math.sqrt(2)
    sd_smooth = sd_raw / (2 * math.sqrt(math.pi) * sigma)
    return bg + noise_k * sd_smooth + FLOOR_SPREAD_MULT * (bg - low)


def _foreground(img: np.ndarray, raw: np.ndarray, sigma: float, noise_k: float, classes: int) -> np.ndarray:
    """Pixels above max(lowest Otsu level for `classes`, noise floor)."""
    if np.ptp(img) == 0:
        return np.zeros(img.shape, bool)
    if classes == 2:
        level = float(threshold_otsu(img, nbins=OTSU_BINS))
    else:
        level = float(threshold_multiotsu(img, classes=classes, nbins=OTSU_BINS)[0])
    return img > max(level, noise_floor(img, raw, sigma, noise_k))


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, (NORM_LOW_PERCENTILE, NORM_HIGH_PERCENTILE))
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)


def segment_site(channels: dict[str, np.ndarray], *, sigma: float = SIGMA, noise_k: float = NOISE_K,
                 min_distance: int = MIN_DISTANCE, min_area: int = MIN_AREA, compactness: float = COMPACTNESS) -> tuple[np.ndarray, np.ndarray]:
    """(nuclei, cells) label images of one site; cell `i` is seeded by nucleus `i`.

    Nuclei: Gaussian smooth DNA (SIGMA), threshold at max(Otsu with NUCLEI_OTSU_CLASSES, noise floor), remove
    objects under MIN_AREA, fill holes, split touching nuclei by a distance-transform watershed with peaks at
    least MIN_DISTANCE apart, drop objects under MIN_AREA, number sequentially. Cells: the sum of the
    per-channel normalised (NORM_LOW_PERCENTILE to NORM_HIGH_PERCENTILE) cell channels present from AGP and ER,
    smoothed, thresholded at max(lowest of a CELL_OTSU_CLASSES multi-Otsu, noise floor), then a watershed on the
    inverted image seeded by the nuclei, inside foreground-or-nucleus, with COMPACTNESS. Compactness keeps a
    bright cell from flooding its dimmer neighbours where cells overlap. Without a cell channel, cells = nuclei.

    Parameters, all module constants a matching pipeline copies: sigma=SIGMA, noise_k=NOISE_K,
    min_distance=MIN_DISTANCE, min_area=MIN_AREA, compactness=COMPACTNESS, and the noise floor and Otsu
    constants FLOOR_BG_PERCENTILE, FLOOR_LOW_PERCENTILE, FLOOR_SPREAD_MULT, NUCLEI_OTSU_CLASSES,
    CELL_OTSU_CLASSES, OTSU_BINS, NORM_LOW_PERCENTILE, NORM_HIGH_PERCENTILE.
    """
    raw = channels["DNA"].astype(float)
    smooth = gaussian(raw, sigma=sigma, preserve_range=True)
    fg = _foreground(smooth, raw, sigma, noise_k, NUCLEI_OTSU_CLASSES)
    fg = ndi.binary_fill_holes(remove_small_objects(fg, max_size=min_area - 1))
    nuclei = np.zeros(raw.shape, np.int32)
    if fg.any():
        dist = ndi.distance_transform_edt(fg)
        peaks = peak_local_max(dist, min_distance=min_distance, labels=label(fg), exclude_border=False)
        markers = np.zeros(raw.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        ws = watershed(-dist, markers, mask=fg)
        sizes = np.bincount(ws.ravel())
        keep = np.flatnonzero(sizes >= min_area)
        keep = keep[keep > 0]
        lut = np.zeros(len(sizes), np.int32)
        lut[keep] = np.arange(1, len(keep) + 1)
        nuclei = lut[ws]
    cells = np.zeros_like(nuclei)
    present = [channels[c].astype(float) for c in CELL_CHANNELS if c in channels]
    if nuclei.any() and present:
        total_raw = sum(_normalise(p) for p in present)
        total = gaussian(total_raw, sigma=sigma, preserve_range=True)
        mask = _foreground(total, total_raw, sigma, noise_k, CELL_OTSU_CLASSES) | (nuclei > 0)
        cells = watershed(-total, markers=nuclei, mask=mask, compactness=compactness).astype(np.int32)
    elif nuclei.any():
        cells = nuclei.copy()
    return nuclei, cells


# ---------- report types ----------

@dataclass
class Check:
    stage: int
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)
    status: str = ""  # "pass", "fail" or "not_checked" (the feature is not measurable by the naive pipeline)

    def __post_init__(self):
        if not self.status:
            self.status = "pass" if self.passed else "fail"


@dataclass
class Report:
    stage1: list[Check]
    stage2: list[Check]
    asserted: bool
    condition_medians: dict
    condition_stats: dict = field(default_factory=dict)
    exclusions: dict = field(default_factory=dict)  # sites left out of stage 2 and the wells that remain per condition

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.stage1 + self.stage2)


# ---------- measuring ----------

def _site_objects(nuclei: np.ndarray, cells: np.ndarray) -> dict:
    return {"count": int(len(np.unique(cells[cells > 0]))),
            "nuclei_area": np.array([p.area for p in regionprops(nuclei)], float),
            "cell_area": np.array([p.area for p in regionprops(cells)], float),
            "cell_ecc": np.array([p.eccentricity for p in regionprops(cells)], float)}


def _nanmedian(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


# ---------- stage 1: written masks against the truth tables ----------

DIAMETER_MEDIAN_TOL, DIAMETER_P95_TOL = 0.03, 0.08  # relative error of mask equivalent diameter vs truth diameter


def _stage1(out: Path, nuclei_t: pd.DataFrame, cells_t: pd.DataFrame, images_t: pd.DataFrame,
            wells: pd.DataFrame) -> list[Check]:
    sites = list_sites(out)
    n_by = {k: g for k, g in nuclei_t.groupby(["plate", "well", "site"])}
    c_by = {k: set(g.cell_id) for k, g in cells_t.groupby(["plate", "well", "site"])}
    blank_wells = set(zip(wells.loc[wells.role == "blank", "plate"], wells.loc[wells.role == "blank", "well"]))
    bad_ids, bad_shape, bad_blank, rel = [], [], [], []
    for plate, well, site in sites:
        key = (plate, well, site)
        nuc, cel = read_truth_masks(out, plate, well, site)
        shape = next(iter(read_site(out, plate, well, site).values())).shape
        if nuc.shape != shape or cel.shape != shape:
            bad_shape.append(key)
        g = n_by.get(key)
        want_n = set(g.nucleus_id) if g is not None else set()
        if set(np.unique(nuc[nuc > 0])) != want_n or set(np.unique(cel[cel > 0])) != c_by.get(key, set()):
            bad_ids.append(key)
        if (plate, well) in blank_wells and (nuc.any() or cel.any()):
            bad_blank.append(key)
        if g is not None:
            t = g.set_index("nucleus_id")
            h, w_ = nuc.shape
            for p in regionprops(nuc):
                b = p.bbox
                if b[0] > 0 and b[1] > 0 and b[2] < h and b[3] < w_:  # whole nucleus inside the frame
                    if p.label not in t.index:  # already reported by the label-set check
                        continue
                    d = float(t.loc[p.label, "diameter"])
                    rel.append(abs(p.equivalent_diameter_area - d) / d)
    n_img = images_t.groupby(["plate", "well", "site"]).n_cells.first()
    n_truth = cells_t.groupby(["plate", "well", "site"]).size()
    count_mismatch = [k for k, v in n_img.items() if int(v) != int(n_truth.get(k, 0))]
    checks = [
        Check(1, "stage1: mask shapes match the images", not bad_shape, {"bad_sites": len(bad_shape), "sites": len(sites)}),
        Check(1, "stage1: mask labels match the truth tables", not bad_ids, {"bad_sites": len(bad_ids), "sites": len(sites)}),
        Check(1, "stage1: blank wells have empty masks", not bad_blank, {"bad_sites": len(bad_blank)}),
        Check(1, "stage1: images table cell counts match the cells table", not count_mismatch,
              {"bad_sites": len(count_mismatch)}),
    ]
    if rel:
        for name, v, tol in (("median", float(np.median(rel)), DIAMETER_MEDIAN_TOL),
                             ("p95", float(np.percentile(rel, 95)), DIAMETER_P95_TOL)):
            checks.append(Check(1, f"stage1: nucleus_diameter_{name}_rel_err", v <= tol,
                                {"value": v, "tolerance": tol, "n": len(rel)}))
    else:
        checks.append(Check(1, "stage1: no whole nucleus to measure", False, {"n": 0}))
    return checks


# ---------- stage 2: the naive pipeline on the images ----------

COUNT_TOL = 0.25            # per-well nuclei count within this fraction of truth (non-defect, non-blank wells)
# ... or within this many nuclei, whichever is larger. A well with 6 nuclei allows 25% = 1.5, but a nucleus smaller
# than MIN_AREA after thresholding cannot be found by any pipeline with this constant, and wells of a strongly
# shrinking condition can hold a few of them. A miss of 2 is such a granularity artefact; 3 of 6 is not.
COUNT_TOL_NUCLEI = 2


def count_within_tolerance(found: float, truth: float) -> bool:
    return abs(found - truth) <= max(COUNT_TOL * truth, COUNT_TOL_NUCLEI)
MIN_EFFECT_D = 1.0          # a planted effect must move the well median by this many pooled SDs, in its direction
P_DIFFERS = 0.05
# A null twin must stay inside the band a planted effect has to clear: a twin that moves as far as the weakest
# acceptable planted effect would be indistinguishable from one. Chance movement below that is allowed.
NULL_D_MAX = MIN_EFFECT_D
# With 4 wells against 8, a true null still lands above NULL_D_MAX by chance on some seeds (small seed 4321: the
# CPD-001 cell count had d = +1.50, p = 0.105, with nothing planted). So the null is a bounded property of the
# set: at most this many of its features may be marginal (called an effect: |d| >= NULL_D_MAX and p < P_DIFFERS);
# more than that fails them all.
NULL_MAX_MARGINAL = 1
# Chance cannot explain a large shift in a single feature, so one feature past this ceiling fails on its own. Small
# seed 4321's chance d = 1.50 is well under it; a segmenter bias moving only one feature (say d = +8) is not.
NULL_D_CEILING = 3.0


def null_verdicts(effects: list[tuple[float, float]]) -> list[bool]:
    """Pass/fail per null feature from its (|d|, p): none at or past NULL_D_CEILING whatever p, and at most
    NULL_MAX_MARGINAL marginal features. A feature is marginal when it would itself be called an effect
    (|d| >= NULL_D_MAX and p < P_DIFFERS); more marginal features than allowed fails every marginal one."""
    marginal = [d >= NULL_D_MAX and p < P_DIFFERS for d, p in effects]
    too_many = sum(marginal) > NULL_MAX_MARGINAL
    return [d < NULL_D_CEILING and not (m and too_many) for (d, _), m in zip(effects, marginal)]


def _well_features(out: Path, plate: str, well: str, sites: list[int], keep_sites: set) -> dict | None:
    objs = []
    n_nuclei = []
    for s in sites:
        if (plate, well, s) not in keep_sites:
            continue
        nuc, cel = segment_site(read_site(out, plate, well, s))
        objs.append(_site_objects(nuc, cel))
        n_nuclei.append(int(nuc.max()))
    if not objs:
        return None
    cat = lambda k: np.concatenate([o[k] for o in objs])  # noqa: E731
    return {"Image_Count_Cells": float(np.mean([o["count"] for o in objs])),
            "Nuclei_AreaShape_Area": _nanmedian(cat("nuclei_area")),
            "Cells_AreaShape_Area": _nanmedian(cat("cell_area")),
            "Cells_AreaShape_Eccentricity": _nanmedian(cat("cell_ecc")),
            "nuclei_per_site": float(np.mean(n_nuclei))}


def _effect(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(signed effect d = median difference over pooled SD, two-sided Mann-Whitney p)."""
    pooled = math.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    diff = float(np.median(a) - np.median(b))
    d = 0.0 if diff == 0 else diff / max(pooled, 1e-12)
    return d, float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _values(wells: dict, cond: str, feat: str) -> np.ndarray:
    v = np.array([w[feat] for w in wells.get(cond, [])], float)
    return v[~np.isnan(v)]


def _stage2(out: Path, design, wells_df: pd.DataFrame, qc_df: pd.DataFrame, nuclei_t: pd.DataFrame):
    # Only saturated-debris sites are excluded: a global threshold dominated by a saturated blob is a documented
    # limit of a naive segmenter. Blurred sites still segment, so they stay in.
    defect_sites = {(r.plate, r.well, int(r.site)) for r in qc_df.itertuples() if r.defect == "saturated_debris"}
    truth_n = nuclei_t.groupby(["plate", "well", "site"]).size()
    sites_of: dict[tuple[str, str], list[int]] = {}
    for p, w, s in list_sites(out):
        sites_of.setdefault((p, w), []).append(s)
    keep = {(p, w, s) for (p, w), ss in sites_of.items() for s in ss} - defect_sites
    cond_of = {(r.plate, r.well): r.condition for r in wells_df.itertuples()}
    role_of = {(r.plate, r.well): r.role for r in wells_df.itertuples()}

    wells: dict[str, list[dict]] = {}
    count_bad, count_n, blank_bad, blank_n = [], 0, [], 0
    for (plate, well), sites in sites_of.items():
        feats = _well_features(out, plate, well, sites, keep)
        if feats is None:
            continue
        if role_of[(plate, well)] == "blank":
            blank_n += 1
            if feats["nuclei_per_site"] != 0:
                blank_bad.append(well)
            continue
        t = float(np.mean([truth_n.get((plate, well, s), 0) for s in sites if (plate, well, s) in keep]))
        count_n += 1
        if not count_within_tolerance(feats["nuclei_per_site"], t):
            count_bad.append({"well": well, "found": feats["nuclei_per_site"], "truth": t})
        wells.setdefault(cond_of[(plate, well)], []).append(feats)

    checks = [
        Check(2, "blank wells give zero nuclei", not blank_bad, {"blank_wells": blank_n, "bad": blank_bad}),
        Check(2, f"per-well nuclei count within {COUNT_TOL:.0%} or {COUNT_TOL_NUCLEI} nuclei of truth", not count_bad,
              {"wells": count_n, "excluded_sites": len(defect_sites), "bad": count_bad}),
    ]
    features = SUPPORTED
    medians = {c: {f: _nanmedian(_values(wells, c, f)) for f in features} for c in wells}
    stats = {}
    for c, ws in wells.items():
        stats[c] = {"n_wells": len(ws)}
        for f in features:
            v = _values(wells, c, f)
            stats[c][f] = ({"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                            "q3": float(np.percentile(v, 75))} if v.size else None)
    exclusions = {"saturated_debris": sorted(f"{p}/{w}/s{si}" for p, w, si in defect_sites),
                  "n_per_condition": {c: len(ws) for c, ws in wells.items()}}
    def excluded_for(*conds):  # excluded sites that belong to wells of the compared conditions
        return sorted(f"{p}/{w}/s{si}" for p, w, si in defect_sites if cond_of[(p, w)] in conds)

    exp = design.expectations
    if exp is None:
        return checks, medians, stats, exclusions
    high = max(design.doses.values(), key=lambda d: d.strength).label

    for comp, entries in exp.differs.items():
        cond = f"{comp}|{high}"
        for e in entries:
            name = f"differs: {cond} vs {exp.reference} · {e.feature} ({e.direction})"
            if e.feature not in SUPPORTED:
                checks.append(Check(2, name, True, {"reason": "feature not measurable by the naive pipeline"}, "not_checked"))
                continue
            a, b = _values(wells, cond, e.feature), _values(wells, exp.reference, e.feature)
            if a.size < 2 or b.size < 2:
                checks.append(Check(2, name, False, {"insufficient": True, "n_a": int(a.size), "n_b": int(b.size)}))
                continue
            d, p = _effect(a, b)
            sign = 1 if e.direction == "up" else -1
            checks.append(Check(2, name, bool(sign * d >= MIN_EFFECT_D and p < P_DIFFERS),
                                {"p": p, "d": d, "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                                 "n_a": int(a.size), "n_b": int(b.size),
                                 "min_effect_d": MIN_EFFECT_D,
                                 "excluded_sites": excluded_for(cond, exp.reference)}))
    for comp in exp.same:
        cond = f"{comp}|{high}"
        rows = []
        for feat in SUPPORTED:
            name = f"same: {cond} vs {exp.reference} \u00b7 {feat}"
            a, b = _values(wells, cond, feat), _values(wells, exp.reference, feat)
            if a.size < 2 or b.size < 2:
                checks.append(Check(2, name, False, {"insufficient": True, "n_a": int(a.size), "n_b": int(b.size)}))
                continue
            d, p = _effect(a, b)
            rows.append((name, {"p": p, "d": d, "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                                "n_a": int(a.size), "n_b": int(b.size), "bound_d": NULL_D_MAX,
                                "marginal": bool(abs(d) >= NULL_D_MAX and p < P_DIFFERS),
                                "excluded_sites": excluded_for(cond, exp.reference)}))
        n_marginal = sum(r["marginal"] for _, r in rows)
        for (name, r), ok in zip(rows, null_verdicts([(abs(r["d"]), r["p"]) for _, r in rows])):
            checks.append(Check(2, name, ok, {**r, "n_marginal": n_marginal, "max_marginal": NULL_MAX_MARGINAL,
                                              "ceiling_d": NULL_D_CEILING}))
    return checks, medians, stats, exclusions


# ---------- report ----------

def _tick(c: Check) -> str:
    return {"pass": "PASS", "fail": "FAIL", "not_checked": "not checked"}[c.status]


def _fmt_stat(s: dict | None) -> str:
    return "n/a" if s is None else f"{s['median']:.4g} [{s['q1']:.4g}–{s['q3']:.4g}]"


def _markdown(rep: Report) -> str:
    lines = ["# Recoverability report (cell_plate)", "", f"**Overall:** {'PASS' if rep.passed else 'FAIL'}", "",
             "## Stage 1 — written masks vs truth tables", "", "| check | detail | pass |", "|---|---|---|"]
    lines += [f"| {c.name} | {json.dumps(c.detail)} | {_tick(c)} |" for c in rep.stage1]
    lines += ["", "## Stage 2 — naive pipeline on the images", ""]
    if not rep.asserted:
        lines.append("_No `expectations` in the design: only counts asserted._")
    lines += ["| check | detail | pass |", "|---|---|---|"]
    for c in rep.stage2:
        d = c.detail
        if "p" in d:
            lim = f"min effect {d['min_effect_d']}" if "min_effect_d" in d else f"null bound {d['bound_d']:.2f}, {d['n_marginal']} marginal of at most {d['max_marginal']}, ceiling {d['ceiling_d']:.1f}"
            text = f"n {d['n_a']} vs {d['n_b']}, p {d['p']:.2g}, d {d['d']:.2f} ({lim}), medians {d['median_a']:.4g} vs {d['median_b']:.4g}"
            if d.get("excluded_sites"):
                text += f", excluded {', '.join(d['excluded_sites'])}"
        else:
            text = json.dumps(d)
        lines.append(f"| {c.name} | {text} | {_tick(c)} |")
    lines += ["", "## Reference pipeline", "",
              "DNA: Gaussian sigma=1, threshold at max(Otsu, noise floor), fill holes, distance watershed, drop objects under 60 px.",
              "Cells: normalised sum of AGP and ER, threshold at the lowest of three multi-Otsu levels, guarded by the same noise floor, compact watershed seeded by the nuclei.",
              "Well features: count is the mean cells per site; areas and eccentricity are medians over objects.",
              "Saturated-debris sites are excluded (a global threshold dominated by a saturated blob is a limit of a naive",
              "segmenter); blurred sites are kept. Effects are tested at the",
              "high dose against the reference wells, as median difference over pooled SD with a Mann-Whitney p.", "",
              "## Expected answer — per-condition median [IQR across wells] (naive pipeline)", "",
              "| condition | n_wells | " + " | ".join(SUPPORTED) + " |", "|---" * (len(SUPPORTED) + 2) + "|"]
    for cond, st in rep.condition_stats.items():
        lines.append(f"| {cond} | {st['n_wells']} | " + " | ".join(_fmt_stat(st[f]) for f in SUPPORTED) + " |")
    lines += ["", "## Exclusions", "", f"Saturated-debris sites left out: {', '.join(rep.exclusions['saturated_debris']) or 'none'}", "",
              "Wells per condition after exclusions: " + ", ".join(f"{c}: {n}" for c, n in rep.exclusions["n_per_condition"].items())]
    return "\n".join(lines) + "\n"


def _nan_to_none(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nan_to_none(v) for v in x]
    return x


def check_dataset(out: Path, write: bool = True) -> Report:
    out = Path(out)
    design = load_design(out / "design.yaml")
    t = out / "truth"
    nuclei_t, cells_t = pd.read_parquet(t / "nuclei.parquet"), pd.read_parquet(t / "cells.parquet")
    images_t, wells_df, qc_df = (pd.read_parquet(t / f"{n}.parquet") for n in ("images", "wells", "qc"))
    stage2, medians, stats, exclusions = _stage2(out, design, wells_df, qc_df, nuclei_t)
    rep = Report(_stage1(out, nuclei_t, cells_t, images_t, wells_df), stage2, design.expectations is not None,
                 medians, stats, exclusions)
    if write:
        (out / "check").mkdir(exist_ok=True)
        body = _nan_to_none({"passed": rep.passed, "asserted": rep.asserted,
                             "stage1": [asdict(c) for c in rep.stage1], "stage2": [asdict(c) for c in rep.stage2],
                             "condition_medians": rep.condition_medians, "condition_stats": rep.condition_stats,
                             "exclusions": rep.exclusions})
        (out / "check" / "report.json").write_text(json.dumps(body, indent=2, default=float, allow_nan=False) + "\n")
        (out / "check" / "report.md").write_text(_markdown(rep))
    return rep
