"""Effects, multiple-testing correction, replicate CV and replicate mAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sst


def _order(s: pd.Series) -> list[str]:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [str(c) for c in s.cat.categories]
    return list(dict.fromkeys(s.astype(str)))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN in, NaN out, ranked over non-NaN only."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    adj = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def effects(profiles: pd.DataFrame, reference: str, columns: list[str]) -> pd.DataFrame:
    cond = profiles["condition"].astype(str)
    ref = profiles[cond == reference]
    rows = []
    for c in _order(profiles["condition"]):
        c = str(c)
        if c == reference:
            continue
        grp = profiles[cond == c]
        for col in columns:
            a, b = ref[col].dropna().to_numpy(float), grp[col].dropna().to_numpy(float)
            mr = float(np.median(a)) if len(a) else np.nan
            mc = float(np.median(b)) if len(b) else np.nan
            diff = mc - mr if len(a) and len(b) else np.nan
            d = p = np.nan
            if len(a) >= 2 and len(b) >= 2:
                na, nb = len(a), len(b)
                sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
                d = 0.0 if diff == 0 else diff / max(sp, 1e-12)
                try:
                    p = float(sst.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                except ValueError:  # all values identical
                    p = 1.0
            rows.append({"condition": c, "column": col, "median_ref": mr, "median_cond": mc,
                         "median_diff": diff, "d": d, "p": p, "n_ref": len(a), "n_cond": len(b)})
    df = pd.DataFrame(rows, columns=["condition", "column", "median_ref", "median_cond", "median_diff",
                                     "d", "p", "n_ref", "n_cond"])
    df["q"] = bh(df["p"].to_numpy(float))
    df["flagged"] = (df["q"] < 0.05) & (df["d"].abs() >= 1)
    return df[["condition", "column", "median_ref", "median_cond", "median_diff", "d", "p", "q",
               "flagged", "n_ref", "n_cond"]]


def replicate_cv(profiles: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    cond = profiles["condition"]
    for c, grp in profiles.groupby(cond, observed=True, sort=False):
        for col in columns:
            v = grp[col].dropna()
            mean = v.mean() if len(v) else np.nan
            cv = np.nan if (len(v) < 2 or mean == 0 or np.isnan(mean)) else float(v.std(ddof=1) / abs(mean))
            rows.append({"condition": str(c), "column": col, "cv": cv})
    return pd.DataFrame(rows, columns=["condition", "column", "cv"])


def replicate_map(z: pd.DataFrame, conditions: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Mean average precision of retrieving same-condition wells by cosine similarity."""
    X = np.nan_to_num(z[columns].to_numpy(float), nan=0.0)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norm == 0, 1.0, norm)
    S = Xn @ Xn.T
    ids = [str(i) for i in z.index]
    cond = conditions.astype(str).to_numpy()
    counts = pd.Series(cond).value_counts()
    aps: dict[str, list[float]] = {}
    for i in range(len(ids)):
        if counts[cond[i]] < 2:
            continue
        others = [j for j in range(len(ids)) if j != i]
        others.sort(key=lambda j: (-S[i, j], ids[j]))
        hits = 0
        ps = []
        for rank, j in enumerate(others, 1):
            if cond[j] == cond[i]:
                hits += 1
                ps.append(hits / rank)
        aps.setdefault(cond[i], []).append(float(np.mean(ps)))
    order = _order(conditions)
    rows = [{"condition": str(c), "mAP": float(np.mean(aps[str(c)])), "n_wells": int(counts[str(c)])}
            for c in order if str(c) in aps]
    return pd.DataFrame(rows, columns=["condition", "mAP", "n_wells"])
