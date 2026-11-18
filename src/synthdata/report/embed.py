"""PCA, UMAP, Ward clustering, similarity, cluster scores and the object-level embedding."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from synthdata.report.contract import FeatureSpec


def _matrix(z: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return z[columns].to_numpy(float)


def pca(z: pd.DataFrame, columns: list[str]):
    """(scores[PC1..PC3], explained variance ratios, loadings) or None with fewer than 3 wells."""
    if len(z) < 3:
        return None
    x = np.nan_to_num(_matrix(z, columns), nan=0.0)
    k = min(3, x.shape[0], x.shape[1])
    m = PCA(n_components=k, svd_solver="full").fit(x)
    comps = m.components_.copy()
    scores = m.transform(x)
    for i in range(k):  # largest absolute loading positive: deterministic sign
        if comps[i][np.argmax(np.abs(comps[i]))] < 0:
            comps[i] *= -1
            scores[:, i] *= -1
    names = [f"PC{i + 1}" for i in range(k)]
    return (pd.DataFrame(scores, index=z.index, columns=names), [float(v) for v in m.explained_variance_ratio_],
            pd.DataFrame(comps.T, index=columns, columns=names))


def umap_embed(matrix, seed: int = 0):
    """2-D UMAP (n, 2), or None with fewer than 4 rows."""
    x = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    n = len(x)
    if n < 4:
        return None
    import umap

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)
        return umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.1, random_state=seed, n_jobs=1).fit_transform(x)


def ward(z: pd.DataFrame, columns: list[str]):
    """(linkage Z, leaf order) or None with fewer than 3 wells."""
    if len(z) < 3:
        return None
    Z = linkage(np.nan_to_num(_matrix(z, columns), nan=0.0), method="ward")
    return Z, [int(i) for i in dendrogram(Z, no_plot=True)["leaves"]]


def dendrogram_segments(Z, labels) -> pd.DataFrame:
    """One row per merge corner: 4 points per merge, x with leaves at 0..n-1 in leaf order."""
    d = dendrogram(Z, no_plot=True)
    rows = []
    for s, (ic, dc) in enumerate(zip(d["icoord"], d["dcoord"])):
        for o, (x, y) in enumerate(zip(ic, dc)):
            rows.append({"x": (x - 5.0) / 10.0, "y": float(dc[o]), "segment": s, "order": o})
    return pd.DataFrame(rows, columns=["x", "y", "segment", "order"])


def similarity(z: pd.DataFrame, columns: list[str], order: list[int]) -> pd.DataFrame:
    """Long-form well x well cosine similarity in dendrogram order."""
    x = np.nan_to_num(_matrix(z, columns), nan=0.0)[order]
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    x = x / norm
    cos = x @ x.T
    names = [z.index[i] for i in order]
    n = len(names)
    return pd.DataFrame({"a": np.repeat(np.arange(n), n), "b": np.tile(np.arange(n), n),
                         "cosine": cos.ravel()}).assign(a=lambda d: [names[i] for i in d["a"]],
                                                        b=lambda d: [names[i] for i in d["b"]])


def cluster_scores(Z, true_labels, k: int):
    """ARI/NMI of the Ward cut at k clusters against true labels; None if <3 wells or 1 condition."""
    true = np.asarray(true_labels)
    if len(true) < 3 or len(set(true.tolist())) < 2:
        return None
    lab = fcluster(Z, k, "maxclust")
    return {"ari": float(adjusted_rand_score(true, lab)), "nmi": float(normalized_mutual_info_score(true, lab)),
            "labels": [int(v) for v in lab]}


def _shares(avail: dict, n: int) -> dict:
    """Equal shares capped by availability; leftovers redistributed, remainder to the first conditions."""
    share = {c: 0 for c in avail}
    open_ = sorted(avail)
    left = n
    while open_ and left > 0:
        base, extra = divmod(left, len(open_))
        want = {c: base + (1 if i < extra else 0) for i, c in enumerate(open_)}
        nxt = []
        for c in open_:
            take = min(want[c], avail[c] - share[c])
            share[c] += take
            left -= take
            if share[c] < avail[c]:
                nxt.append(c)
        if len(nxt) == len(open_) and all(want[c] == 0 for c in open_):
            break
        open_ = nxt
    return share


def object_sample(objects: pd.DataFrame, wells: pd.DataFrame, features, n: int = 5000, seed: int = 0):
    """Condition-stratified subsample (equal shares capped by availability); None if <3 wells or 1 condition."""
    if len(wells) < 3 or wells["condition"].nunique() < 2:
        return None
    cond = wells[["plate", "well", "condition"]]
    df = objects.merge(cond, on=["plate", "well"], how="inner")
    order = {str(c): i for i, c in enumerate(pd.unique(wells["condition"]))}
    df["condition"] = df["condition"].astype(str)
    avail = df["condition"].value_counts().to_dict()
    share = _shares(avail, n)
    rng = np.random.default_rng(seed)
    parts = []
    for c in sorted(avail, key=order.get):
        g = df[df["condition"] == c].sort_values(["plate", "well", "site", "object_id"], kind="stable")
        idx = np.sort(rng.choice(len(g), size=share[c], replace=False))
        parts.append(g.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def object_embedding(sample: pd.DataFrame, features: tuple[FeatureSpec, ...]):
    """UMAP of standardised linear features plus cos 2θ / sin 2θ of the axial one; DataFrame[u1, u2]."""
    if sample is None:
        return None
    cols = []
    for f in features:
        v = sample[f.name].to_numpy(float)
        if f.kind == "axial":
            th = np.deg2rad(2 * v)
            cols += [np.cos(th), np.sin(th)]
        else:
            sd = np.nanstd(v)
            cols.append((v - np.nanmean(v)) / (sd if sd > 0 else 1.0))
    u = umap_embed(np.column_stack(cols))
    if u is None:
        return None
    return pd.DataFrame(u, columns=["u1", "u2"], index=sample.index)
