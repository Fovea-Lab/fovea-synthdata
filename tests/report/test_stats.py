import numpy as np
import pandas as pd
import pytest

from synthdata.report.profiles import PROFILE_COLUMNS, robust_z, well_profiles
from synthdata.report.contract import DEFAULT_FEATURES, validate
from synthdata.report.stats import bh, effects, replicate_cv, replicate_map
from synthdata.generators.particle_plate.truth_report import truth_inputs


def test_bh():
    q = bh(np.array([0.01, 0.04, 0.03, 0.2]))
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    q = bh(np.array([0.01, np.nan, 0.04]))
    assert np.isnan(q[1]) and q[0] == pytest.approx(0.02) and q[2] == pytest.approx(0.04)


def _prof(shift=5.0, n=6):
    rng = np.random.default_rng(0)
    base = rng.normal(10, 1, n)
    df = pd.DataFrame({"condition": pd.Categorical(
        ["control"] * n + ["shift"] * n + ["twin"] * n + ["one"],
        categories=["control", "shift", "twin", "one"], ordered=True),
        "v": np.r_[base, base + shift, base, [1.0]]})
    return df


def test_four_replicates_cannot_reach_significance():
    # 4 vs 4: the smallest Mann-Whitney p is 0.0286; with several columns tested, q = p*m/rank > 0.05.
    df = pd.DataFrame({"condition": pd.Categorical(["control"] * 4 + ["big"] * 4, ordered=True),
                       "v": [1.0, 1.1, 0.9, 1.05, 100.0, 101.0, 99.0, 100.5],
                       "n1": [1.0, 2.0, 3.0, 4.0, 2.5, 1.5, 3.5, 2.0],
                       "n2": [4.0, 3.0, 2.0, 1.0, 2.0, 3.5, 1.0, 4.0],
                       "n3": [1.0, 5.0, 2.0, 4.0, 3.0, 3.1, 2.9, 3.2]})
    e = effects(df, "control", ["v", "n1", "n2", "n3"]).set_index("column").loc["v"]
    assert e["p"] == pytest.approx(0.0286, abs=1e-4)
    assert e["d"] > 1 and e["q"] > 0.05 and not e["flagged"]


def test_effect_size_finite_when_pooled_sd_zero():
    df = pd.DataFrame({"condition": pd.Categorical(["control"] * 3 + ["t"] * 3, ordered=True),
                       "v": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]})
    e = effects(df, "control", ["v"]).iloc[0]
    assert np.isfinite(e["d"]) and e["d"] > 1


def test_effects():
    e = effects(_prof(), "control", ["v"]).set_index("condition")
    assert e.loc["shift", "flagged"] and e.loc["shift", "d"] > 1
    assert not e.loc["twin", "flagged"] and e.loc["twin", "d"] == 0
    assert np.isnan(e.loc["one", "p"]) and np.isnan(e.loc["one", "d"]) and not e.loc["one", "flagged"]
    assert "control" not in e.index


def test_replicate_cv():
    df = pd.DataFrame({"condition": ["a", "a", "b", "b"], "v": [1.0, 3.0, 1.0, -1.0]})
    cv = replicate_cv(df, ["v"]).set_index("condition")["cv"]
    assert cv["a"] == pytest.approx(np.std([1, 3], ddof=1) / 2)
    assert np.isnan(cv["b"])


def _brute_map(z, cond):
    X = z.to_numpy(float)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    ids = list(z.index)
    aps = {}
    for i in range(len(ids)):
        order = sorted((j for j in range(len(ids)) if j != i), key=lambda j: (-S[i, j], ids[j]))
        hits, ps = 0, []
        for r, j in enumerate(order, 1):
            if cond[j] == cond[i]:
                hits += 1
                ps.append(hits / r)
        aps.setdefault(cond[i], []).append(np.mean(ps))
    return {c: np.mean(v) for c, v in aps.items()}


def test_replicate_map_separated():
    rng = np.random.default_rng(0)
    centres = np.eye(3) * 10
    X = np.vstack([c + rng.normal(0, 0.1, (3, 3)) for c in centres])
    z = pd.DataFrame(X, columns=list("abc"), index=[f"w{i}" for i in range(9)])
    cond = pd.Series(np.repeat(["x", "y", "z"], 3), index=z.index)
    m = replicate_map(z, cond, list("abc")).set_index("condition")
    assert (m["mAP"] == 1.0).all() and (m["n_wells"] == 3).all()


def test_replicate_map_identical_conditions():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (3, 4))
    X = np.vstack([X, X + 0.01 * rng.normal(size=X.shape)])
    z = pd.DataFrame(X, columns=list("abcd"), index=[f"w{i}" for i in range(6)])
    cond = pd.Series(["x", "x", "x", "y", "y", "y"], index=z.index)
    got = replicate_map(z, cond, list("abcd")).set_index("condition")["mAP"]
    want = _brute_map(z, list(cond))
    for c in "xy":
        assert got[c] == pytest.approx(want[c])
        assert 0.3 < got[c] < 0.8


def test_replicate_map_skips_singletons():
    z = pd.DataFrame(np.eye(3), columns=list("abc"), index=list("pqr"))
    m = replicate_map(z, pd.Series(["x", "x", "y"], index=z.index), list("abc"))
    assert list(m["condition"]) == ["x"]


def test_replicate_truth_effects(replicate_truth_dataset):
    inp = validate(truth_inputs(replicate_truth_dataset))
    prof = well_profiles(inp)
    cols = PROFILE_COLUMNS(DEFAULT_FEATURES)
    e = effects(prof, inp.reference, cols)
    f = e.set_index(["condition", "column"])["flagged"]
    assert f[("large", "diameter_median")]
    assert not e[e["condition"] == "control_twin"]["flagged"].any()
    z, _ = robust_z(prof, inp.reference, cols)
    assert z.shape == (len(prof), len(cols))
