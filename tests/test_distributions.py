import math

import numpy as np
import pytest
from pydantic import ValidationError

from synthdata.core.distributions import parse_distribution

N = 200_000
MIX = {"dist": "mixture", "components": [
    {"weight": 0.5, "dist": "normal", "mean": 8, "sd": 1},
    {"weight": 0.5, "dist": "normal", "mean": 20, "sd": 2},
]}


@pytest.mark.parametrize("spec, mean, sd", [
    ({"dist": "normal", "mean": 12, "sd": 1}, 12, 1),
    ({"dist": "lognormal", "mean": 5, "sd": 2}, 5, 2),
    ({"dist": "uniform", "low": 0, "high": 180}, 90, 180 / math.sqrt(12)),
    ({"dist": "poisson", "mean": 80}, 80, math.sqrt(80)),
    ({"dist": "beta_mean", "mean": 0.8, "sd": 0.05}, 0.8, 0.05),
    ({"dist": "fixed", "value": 1.0}, 1.0, 0.0),
    (MIX, 14, math.sqrt(38.5)),
])
def test_sample_matches_describe(spec, mean, sd):
    d = parse_distribution(spec)
    desc = d.describe()
    assert desc.mean == pytest.approx(mean, abs=1e-9)
    assert desc.sd == pytest.approx(sd, abs=1e-6)
    x = d.sample(np.random.default_rng(0), N)
    assert x.shape == (N,)
    assert x.mean() == pytest.approx(mean, abs=6 * sd / math.sqrt(N) + 1e-9)
    assert x.std() == pytest.approx(sd, rel=0.02, abs=1e-9)


@pytest.mark.parametrize("spec", [
    {"dist": "normal", "mean": 12, "sd": 1},
    {"dist": "lognormal", "mean": 5, "sd": 2},
    {"dist": "uniform", "low": 0, "high": 180},
    {"dist": "beta_mean", "mean": 0.3, "sd": 0.1},
    MIX,
])
def test_pdf_integrates_to_one(spec):
    d = parse_distribution(spec)
    desc = d.describe()
    x = np.linspace(desc.lo, desc.hi, 20_001)
    assert np.trapezoid(d.pdf(x), x) == pytest.approx(1.0, abs=0.02)


def test_poisson_pmf_sums_to_one():
    d = parse_distribution({"dist": "poisson", "mean": 12})
    assert d.pdf(np.arange(0, 200)).sum() == pytest.approx(1.0, abs=1e-6)


def test_same_seed_same_sample():
    d = parse_distribution(MIX)
    a = d.sample(np.random.default_rng(3), 100)
    b = d.sample(np.random.default_rng(3), 100)
    assert np.array_equal(a, b)


def test_mixture_dumps_flat_and_round_trips():
    d = parse_distribution(MIX)
    dumped = d.model_dump(mode="json")
    assert dumped["components"][0] == {"weight": 0.5, "dist": "normal", "mean": 8.0, "sd": 1.0}
    assert parse_distribution(dumped) == d


@pytest.mark.parametrize("spec, fragment", [
    ({"dist": "beta_mean", "mean": 0.5, "sd": 0.6}, "impossible"),
    ({"dist": "uniform", "low": 5, "high": 1}, "high > low"),
    ({"dist": "mixture", "components": [
        {"weight": 0.5, "dist": "fixed", "value": 1}, {"weight": 0.2, "dist": "fixed", "value": 2}]},
     "sum to 1"),
    ({"dist": "normal", "mean": 1, "sd": 0}, "greater than 0"),
    ({"dist": "gamma", "shape": 1}, "dist"),
    ({"dist": "normal", "mean": 1, "sd": 1, "sdd": 2}, "sdd"),
])
def test_invalid_specs_fail_clearly(spec, fragment):
    with pytest.raises(ValidationError, match=fragment):
        parse_distribution(spec)
