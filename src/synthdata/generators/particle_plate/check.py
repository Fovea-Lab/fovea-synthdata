"""Recoverability self-check: is the truth recoverable from what was written?

Stage 1 measures the truth masks and compares against the truth tables (is the geometry exact?).
Stage 2 runs a deliberately naive pipeline on the images and asserts the design's `expectations`
(does a simple threshold-and-measure pipeline see the intended differences?). The per-condition medians
it reports are the expected answer a platform pipeline should reproduce.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops

from synthdata.generators.particle_plate.design import FEATURES, load_design
from synthdata.generators.particle_plate.reader import iter_images

ELONGATED = 0.5
MIN_OBJECT_AREA = 10
P_DIFFERS, D_DIFFERS = 0.05, 1.0
P_SAME, D_SAME = 0.001, 3.0
MIN_WEIGHT, MIN_ASHMAN_D = 0.2, 2.0
STAGE1_TOL = {
    "diameter_median_abs_err": 0.15, "diameter_p95_abs_err": 0.35,
    "ecc_elongated_median_abs_err": 0.02, "ecc_elongated_p95_abs_err": 0.06,
    "orient_elongated_median_abs_err_deg": 2.0, "orient_elongated_p95_abs_err_deg": 6.0,
    "ecc_round_median_abs_err": 0.25,
}
WELL_FEATURES = tuple(f for f in FEATURES if f != "bimodal_diameter")


@dataclass
class Check:
    stage: int
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)


@dataclass
class Report:
    stage1: list[Check]
    stage2: list[Check]
    asserted: bool
    condition_medians: dict
    condition_stats: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.stage1 + self.stage2)


def regionprops_orientation_deg(phi: float) -> float:
    """skimage orientation (radians from the row axis) → our convention (degrees from +x toward +y)."""
    return float((90.0 - np.degrees(phi)) % 180.0)


def angle_diff_deg(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 90.0) % 180.0 - 90.0)


# ---------- stage 1 ----------

def _stage1(out: Path, particles: pd.DataFrame) -> list[Check]:
    good = particles[~particles.clipped & ~particles.touching & (particles.visible_px > 0)]
    truth = {k: g.set_index("particle_id") for k, g in good.groupby(["plate", "well", "site"])}
    rows = []
    for img in iter_images(out):
        t = truth.get((img.plate, img.well, img.site))
        if t is None:
            continue
        for p in regionprops(img.mask):
            if p.label in t.index:
                r = t.loc[p.label]
                rows.append((r.diameter, r.eccentricity, r.orientation, p.equivalent_diameter_area,
                             p.eccentricity, regionprops_orientation_deg(p.orientation)))
    if len(good) == 0:
        return [Check(1, "stage1: no isolated particles to measure", True, {"n": 0, "skipped": True})]
    if not rows:
        return [Check(1, "stage1: no eligible truth particle found in the masks", False,
                      {"eligible": int(len(good)), "matched": 0})]
    a = np.array(rows)
    dd = np.abs(a[:, 3] - a[:, 0])
    el = a[:, 1] >= ELONGATED
    metrics = {
        "diameter_median_abs_err": (np.median(dd), dd.size),
        "diameter_p95_abs_err": (np.percentile(dd, 95), dd.size),
    }
    if el.any():
        de, dth = np.abs(a[el, 4] - a[el, 1]), angle_diff_deg(a[el, 5], a[el, 2])
        metrics |= {
            "ecc_elongated_median_abs_err": (np.median(de), de.size),
            "ecc_elongated_p95_abs_err": (np.percentile(de, 95), de.size),
            "orient_elongated_median_abs_err_deg": (np.median(dth), dth.size),
            "orient_elongated_p95_abs_err_deg": (np.percentile(dth, 95), dth.size),
        }
    if (~el).any():
        dr = np.abs(a[~el, 4] - a[~el, 1])
        metrics["ecc_round_median_abs_err"] = (np.median(dr), dr.size)
    checks = [Check(1, name, bool(v <= STAGE1_TOL[name]), {"value": float(v), "tolerance": STAGE1_TOL[name], "n": int(n)})
              for name, (v, n) in metrics.items()]
    checks.append(Check(1, "stage1: every eligible particle found in its mask", len(rows) == len(good),
                        {"eligible": int(len(good)), "matched": len(rows)}))
    return checks


# ---------- stage 2 ----------

def measure_objects(image: np.ndarray) -> dict:
    """The naive reference pipeline: smooth → Otsu → label → regionprops."""
    img = gaussian(image.astype(float), sigma=1, preserve_range=True)
    empty = {"diameter": np.array([]), "eccentricity": np.array([]), "orientation": np.array([]),
             "intensity": np.array([]), "count": 0, "nn_distance": np.array([])}
    if np.ptp(img) == 0:
        return empty
    props = [p for p in regionprops(label(img > threshold_otsu(img)), intensity_image=image)
             if p.area >= MIN_OBJECT_AREA]
    if not props:
        return empty
    cents = np.array([p.centroid for p in props])
    nn = cKDTree(cents).query(cents, k=2)[0][:, 1] if len(props) >= 2 else np.array([])
    return {
        "diameter": np.array([p.equivalent_diameter_area for p in props]),
        "eccentricity": np.array([p.eccentricity for p in props]),
        "orientation": np.array([regionprops_orientation_deg(p.orientation) for p in props]),
        "intensity": np.array([p.intensity_mean for p in props]),
        "count": len(props),
        "nn_distance": nn,
    }


def _nanmedian(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


def _well_features(objs: list[dict]) -> tuple[dict, np.ndarray]:
    cat = {k: np.concatenate([o[k] for o in objs]) for k in ("diameter", "eccentricity", "orientation", "intensity", "nn_distance")}
    el = cat["eccentricity"] >= ELONGATED
    th = np.radians(cat["orientation"][el] if el.sum() >= 3 else cat["orientation"])
    spread = float(1 - abs(np.mean(np.exp(2j * th)))) if th.size else float("nan")
    feats = {
        "diameter": _nanmedian(cat["diameter"]), "eccentricity": _nanmedian(cat["eccentricity"]),
        "orientation_spread": spread, "intensity": _nanmedian(cat["intensity"]),
        "count": float(np.mean([o["count"] for o in objs])), "nn_distance": _nanmedian(cat["nn_distance"]),
    }
    return feats, cat["diameter"]


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Median difference over pooled SD; callers guarantee at least 2 values per side."""
    pooled = math.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    diff = float(np.median(a) - np.median(b))
    return 0.0 if diff == 0 else diff / max(pooled, 1e-12)


def _compare(kind: str, name_a: str, name_b: str, feat: str, wells: dict, pooled_d: dict) -> Check:
    name = f"{kind}: {name_a} vs {name_b} · {feat}"
    for missing in (name_a, name_b):
        if missing not in wells:
            return Check(2, name, False, {"missing": missing, "reason": "not on the plate"})
    if feat == "bimodal_diameter":
        ba, bb = _bimodality(pooled_d[name_a]), _bimodality(pooled_d[name_b])
        ok = (ba["bimodal"] and not bb["bimodal"]) if kind == "differs" else (ba["bimodal"] == bb["bimodal"])
        return Check(2, name, bool(ok), {name_a: ba, name_b: bb})
    a = np.array([w[feat] for w in wells[name_a]], dtype=float)
    b = np.array([w[feat] for w in wells[name_b]], dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return Check(2, name, False, {"insufficient": True, "n_a": int(a.size), "n_b": int(b.size)})
    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    d = _cohen_d(a, b)
    ok = (p < P_DIFFERS and abs(d) >= D_DIFFERS) if kind == "differs" else (p > P_SAME and abs(d) <= D_SAME)
    return Check(2, name, bool(ok), {"p": p, "d": d, "median_a": _nanmedian(a), "median_b": _nanmedian(b),
                                     "n_a": int(a.size), "n_b": int(b.size)})


def _bimodality(x: np.ndarray) -> dict:
    from sklearn.mixture import GaussianMixture

    if x.size < 10:
        return {"bimodal": False, "n": int(x.size)}
    if x.size > 3000:
        x = np.random.default_rng(0).choice(x, 3000, replace=False)
    X = x.reshape(-1, 1)
    g1 = GaussianMixture(1, n_init=3, random_state=0).fit(X)
    g2 = GaussianMixture(2, n_init=3, random_state=0).fit(X)
    mu, sd = g2.means_.ravel(), np.sqrt(g2.covariances_.ravel())
    ashman = float(math.sqrt(2) * abs(mu[0] - mu[1]) / math.sqrt(sd[0] ** 2 + sd[1] ** 2))
    wmin = float(g2.weights_.min())
    delta_bic = float(g1.bic(X) - g2.bic(X))
    return {"bimodal": bool(delta_bic > 0 and wmin >= MIN_WEIGHT and ashman >= MIN_ASHMAN_D),
            "delta_bic": delta_bic, "min_weight": wmin, "ashman_d": ashman, "n": int(x.size)}


def _stage2(out: Path, design, wells_df: pd.DataFrame) -> tuple[list[Check], dict, dict]:
    cond_of = {(r.plate, r.well): r.condition for r in wells_df.itertuples()}
    per_well: dict[tuple[str, str], list[dict]] = {}
    for img in iter_images(out):
        per_well.setdefault((img.plate, img.well), []).append(measure_objects(img.image))
    wells: dict[str, list[dict]] = {}
    pooled_d: dict[str, list[np.ndarray]] = {}
    for key, objs in per_well.items():
        feats, diam = _well_features(objs)
        wells.setdefault(cond_of[key], []).append(feats)
        pooled_d.setdefault(cond_of[key], []).append(diam)
    pooled = {c: np.concatenate(v) for c, v in pooled_d.items()}
    medians = {c: {f: _nanmedian(np.array([w[f] for w in ws], dtype=float)[~np.isnan(np.array([w[f] for w in ws], dtype=float))])
                   for f in WELL_FEATURES} for c, ws in wells.items()}
    stats = {}
    for c, ws in wells.items():
        stats[c] = {"n_wells": len(ws)}
        for f in WELL_FEATURES:
            v = np.array([w[f] for w in ws], dtype=float)
            v = v[~np.isnan(v)]
            stats[c][f] = ({"median": float(np.median(v)), "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))}
                           if v.size else None)
    exp = design.expectations
    if exp is None:
        return [], medians, stats
    checks = []
    for c in exp.same:
        checks += [_compare("same", c, exp.reference, f, wells, pooled) for f in WELL_FEATURES]
    for c, feats in exp.differs.items():
        checks += [_compare("differs", c, exp.reference, f, wells, pooled) for f in feats]
    for pair in exp.pairs:
        checks += [_compare("differs", pair.a, pair.b, f, wells, pooled) for f in pair.differs]
        checks += [_compare("same", pair.a, pair.b, f, wells, pooled) for f in pair.same]
    return checks, medians, stats


# ---------- report ----------

def _tick(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _fmt_bimodal(d: dict) -> str:
    if "delta_bic" not in d:
        return f"bimodal no (n={d.get('n', 0)}, too few objects)"
    return (f"bimodal {'yes' if d['bimodal'] else 'no'}, dBIC {d['delta_bic']:.1f}, "
            f"min weight {d['min_weight']:.2f}, Ashman D {d['ashman_d']:.2f}")


def _fmt_stat(s: dict | None) -> str:
    return "n/a" if s is None else f"{s['median']:.4g} [{s['q1']:.4g}\u2013{s['q3']:.4g}]"


def _markdown(rep: Report) -> str:
    lines = ["# Recoverability report", "", f"**Overall:** {_tick(rep.passed)}", "",
             "## Stage 1 \u2014 truth masks vs truth tables", "", "| check | value | tolerance | n | pass |",
             "|---|---|---|---|---|"]
    for c in rep.stage1:
        d = c.detail
        if "value" in d:
            lines.append(f"| {c.name} | {d['value']:.4g} | {d['tolerance']} | {d['n']} | {_tick(c.passed)} |")
        else:
            lines.append(f"| {c.name} | {json.dumps(d)} | | | {_tick(c.passed)} |")
    lines += ["", "## Stage 2 \u2014 naive pipeline on the images", ""]
    if not rep.asserted:
        lines.append("_No `expectations` in the design: medians reported, nothing asserted._")
    else:
        lines += ["| check | p | d (median diff / pooled SD) | median a | median b | pass |", "|---|---|---|---|---|---|"]
        for c in rep.stage2:
            d = c.detail
            if "p" in d:
                lines.append(f"| {c.name} | {d['p']:.2g} | {d['d']:.2f} | {d['median_a']:.4g} | {d['median_b']:.4g} | {_tick(c.passed)} |")
            elif "missing" in d:
                lines.append(f"| {c.name} | \u2014 | \u2014 | {d['missing']}: {d['reason']} | | {_tick(c.passed)} |")
            elif d.get("insufficient"):
                lines.append(f"| {c.name} | \u2014 | \u2014 | insufficient wells (n_a={d['n_a']}, n_b={d['n_b']}) | | {_tick(c.passed)} |")
            else:
                sides = "; ".join(f"{k}: {_fmt_bimodal(v)}" for k, v in d.items())
                lines.append(f"| {c.name} | \u2014 | \u2014 | {sides} | | {_tick(c.passed)} |")
    lines += ["", "## Reference pipeline", "",
              "Gaussian sigma=1, Otsu threshold, label, drop objects under 10 px, regionprops with the intensity image.",
              "Well features: diameter, eccentricity and intensity are medians over objects; count is the mean objects per site;",
              "nn_distance is the median nearest-neighbour distance between object centroids; orientation_spread is",
              "1 - |mean(exp(2i*theta))| over objects with e >= 0.5 (all objects if fewer than 3).", "",
              "## Expected answer \u2014 per-condition median [IQR across wells] of well features (naive pipeline)", "",
              "| condition | n_wells | " + " | ".join(WELL_FEATURES) + " |", "|---" * (len(WELL_FEATURES) + 2) + "|"]
    for cond, st in rep.condition_stats.items():
        lines.append(f"| {cond} | {st['n_wells']} | " + " | ".join(_fmt_stat(st[f]) for f in WELL_FEATURES) + " |")
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
    particles = pd.read_parquet(out / "truth" / "particles.parquet")
    wells_df = pd.read_parquet(out / "truth" / "wells.parquet")
    stage2, medians, stats = _stage2(out, design, wells_df)
    rep = Report(_stage1(out, particles), stage2, design.expectations is not None, medians, stats)
    if write:
        (out / "check").mkdir(exist_ok=True)
        body = _nan_to_none({"passed": rep.passed, "asserted": rep.asserted,
                             "stage1": [asdict(c) for c in rep.stage1], "stage2": [asdict(c) for c in rep.stage2],
                             "condition_medians": rep.condition_medians, "condition_stats": rep.condition_stats})
        (out / "check" / "report.json").write_text(json.dumps(body, indent=2, default=float, allow_nan=False) + "\n")
        (out / "check" / "report.md").write_text(_markdown(rep))
    return rep
