"""Well-level profiles and robust z-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst
from scipy.spatial import cKDTree

from synthdata.report.contract import FeatureSpec, ReportInputs

META = ("row", "col", "condition", "role", "replicate")


def n_sites(inputs: ReportInputs) -> int:
    """The design-declared site count if given, else the largest number of distinct sites seen in any well."""
    if inputs.n_sites is not None:
        return int(inputs.n_sites)
    if inputs.objects.empty:
        return 1
    return int(max(1, inputs.objects.groupby(["plate", "well"])["site"].nunique().max()))


def axial_stats(theta_deg: np.ndarray) -> tuple[float, float]:
    """(coherence R, mean axis in degrees [0,180)) of axial data."""
    th = np.asarray(theta_deg, dtype=float)
    th = th[~np.isnan(th)]
    if th.size == 0:
        return np.nan, np.nan
    m = np.exp(2j * np.deg2rad(th)).mean()
    axis = float((np.rad2deg(np.angle(m)) / 2) % 180)
    if axis >= 180 - 1e-9:  # -1e-17 % 180 == 180.0; keep the axis in [0, 180)
        axis = 0.0
    return float(abs(m)), axis


def sarle_bimodality(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return np.nan
    g = sst.skew(x, bias=False)
    k = sst.kurtosis(x, bias=False)
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g**2 + 1) / denom) if denom else np.nan


def _nn_by_site(g: pd.DataFrame) -> list[np.ndarray]:
    out = []
    for _, s in g.groupby("site"):
        if len(s) < 2:
            continue
        d, _ = cKDTree(s[["x", "y"]].to_numpy(float)).query(s[["x", "y"]].to_numpy(float), k=2)
        out.append(d[:, 1])
    return out


def clark_evans(nn_per_site: list[np.ndarray], site_area_px: float | None) -> float:
    if site_area_px is None or not nn_per_site:
        return np.nan
    vals = []
    for nn in nn_per_site:
        n = len(nn)
        vals.append(nn.mean() / (0.5 / np.sqrt(n / site_area_px)))
    return float(np.mean(vals))


def _well_row(g: pd.DataFrame, features: tuple[FeatureSpec, ...], n_site: int, area) -> dict:
    r: dict = {"n_objects": len(g), "count_per_site": len(g) / n_site}
    for f in features:
        v = g[f.name].to_numpy(float) if len(g) else np.array([])
        if f.kind == "axial":
            r["orientation_coherence"], r["orientation_mean_axis"] = axial_stats(v)
            continue
        v = v[~np.isnan(v)]
        r[f"{f.name}_median"] = float(np.median(v)) if v.size else np.nan
        r[f"{f.name}_iqr"] = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else np.nan
        if f.name == "diameter":
            r["diameter_bimodality"] = sarle_bimodality(v)
    nns = _nn_by_site(g) if len(g) else []
    r["nn_distance"] = float(np.median(np.concatenate(nns))) if nns else np.nan
    r["clark_evans"] = clark_evans(nns, area)
    return r


def well_profiles(inputs: ReportInputs) -> pd.DataFrame:
    """One row per (plate, well), joined with the well metadata."""
    ns = n_sites(inputs)
    groups = {k: g for k, g in inputs.objects.groupby(["plate", "well"], sort=False)}
    empty = inputs.objects.iloc[0:0]
    rows = {}
    for plate, well in zip(inputs.wells["plate"], inputs.wells["well"]):
        rows[(plate, well)] = _well_row(groups.get((plate, well), empty), inputs.features, ns,
                                        inputs.site_area_px)
    prof = pd.DataFrame.from_dict(rows, orient="index")
    prof.index = pd.MultiIndex.from_tuples(prof.index, names=["plate", "well"])
    meta = inputs.wells.set_index(["plate", "well"])[list(META)]
    return meta.join(prof)


def PROFILE_COLUMNS(features: tuple[FeatureSpec, ...]) -> list[str]:  # noqa: N802
    cols = ["count_per_site"]
    for f in features:
        if f.kind == "linear":
            cols += [f"{f.name}_median", f"{f.name}_iqr"]
    if any(f.kind == "axial" for f in features):
        cols.append("orientation_coherence")
    cols += ["nn_distance", "clark_evans"]
    if any(f.name == "diameter" for f in features):
        cols.append("diameter_bimodality")
    return cols


def robust_z(profiles: pd.DataFrame, reference: str, columns: list[str] | None = None
             ) -> tuple[pd.DataFrame, list[str]]:
    """z = (x - median_ref) / (1.4826*MAD_ref); fall back to ref SD, then 1.0 (noted)."""
    if columns is None:
        columns = [c for c in profiles.columns if c not in META and c not in ("n_objects", "orientation_mean_axis")]
    ref = profiles[profiles["condition"].astype(str) == reference]
    z = pd.DataFrame(index=profiles.index)
    notes: list[str] = []
    for c in columns:
        v = ref[c].dropna()
        med = v.median() if len(v) else np.nan
        scale = 1.4826 * (v - med).abs().median() if len(v) else np.nan
        if not scale or np.isnan(scale):
            sd = v.std(ddof=1) if len(v) > 1 else np.nan
            if sd and not np.isnan(sd):
                scale, how = float(sd), "reference SD"
            else:
                scale, how = 1.0, "1.0"
            if not len(v):
                notes.append(f"{c}: the column has no values in the reference; scaled by {how}")
            else:
                notes.append(f"{c}: reference MAD is zero or undefined; scaled by {how}")
        z[c] = (profiles[c] - med) / scale
    return z, notes
