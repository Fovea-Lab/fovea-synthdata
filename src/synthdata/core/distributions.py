"""One distribution spec shared by every generator.

Specs sample with a caller-supplied ``numpy.random.Generator`` (seeding stays the caller's job) and
describe themselves analytically, so an app can plot the *intended* distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_serializer, model_validator


@dataclass(frozen=True)
class Description:
    mean: float
    sd: float
    lo: float  # plotting range, not a hard support bound
    hi: float


def _lgamma(x: np.ndarray) -> np.ndarray:
    return np.array([math.lgamma(float(v)) for v in np.ravel(x)]).reshape(np.shape(x))


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> Description:
        raise NotImplementedError

    def pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Normal(_Spec):
    dist: Literal["normal"] = "normal"
    mean: float
    sd: float = Field(gt=0)

    def sample(self, rng, n):
        return rng.normal(self.mean, self.sd, n)

    def describe(self):
        return Description(self.mean, self.sd, self.mean - 4 * self.sd, self.mean + 4 * self.sd)

    def pdf(self, x):
        z = (np.asarray(x, dtype=float) - self.mean) / self.sd
        return np.exp(-0.5 * z * z) / (self.sd * math.sqrt(2 * math.pi))


class LogNormal(_Spec):
    """Set by the mean and sd of the variable itself, not of its log."""

    dist: Literal["lognormal"] = "lognormal"
    mean: float = Field(gt=0)
    sd: float = Field(gt=0)

    def _mu_sigma(self) -> tuple[float, float]:
        s2 = math.log1p((self.sd / self.mean) ** 2)
        return math.log(self.mean) - s2 / 2, math.sqrt(s2)

    def sample(self, rng, n):
        mu, s = self._mu_sigma()
        return rng.lognormal(mu, s, n)

    def describe(self):
        mu, s = self._mu_sigma()
        return Description(self.mean, self.sd, 0.0, math.exp(mu + 5 * s))

    def pdf(self, x):
        mu, s = self._mu_sigma()
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        pos = x > 0
        lx = np.log(x[pos])
        out[pos] = np.exp(-((lx - mu) ** 2) / (2 * s * s)) / (x[pos] * s * math.sqrt(2 * math.pi))
        return out


class Uniform(_Spec):
    dist: Literal["uniform"] = "uniform"
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self):
        if not self.high > self.low:
            raise ValueError(f"uniform needs high > low (got low={self.low}, high={self.high})")
        return self

    def sample(self, rng, n):
        return rng.uniform(self.low, self.high, n)

    def describe(self):
        width = self.high - self.low
        pad = 0.05 * width
        return Description((self.low + self.high) / 2, width / math.sqrt(12), self.low - pad, self.high + pad)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.low) & (x <= self.high), 1.0 / (self.high - self.low), 0.0)


class Poisson(_Spec):
    dist: Literal["poisson"] = "poisson"
    mean: float = Field(ge=0)

    def sample(self, rng, n):
        return rng.poisson(self.mean, n).astype(float)

    def describe(self):
        s = math.sqrt(self.mean)
        return Description(self.mean, s, 0.0, self.mean + 5 * s + 1)

    def pdf(self, x):
        k = np.floor(np.asarray(x, dtype=float))
        out = np.zeros_like(k)
        ok = k >= 0
        if self.mean == 0:
            return np.where(k == 0, 1.0, 0.0)
        kk = k[ok]
        out[ok] = np.exp(kk * math.log(self.mean) - self.mean - _lgamma(kk + 1))
        return out


class BetaMean(_Spec):
    """Beta distribution on [0, 1], set by its mean and sd."""

    dist: Literal["beta_mean"] = "beta_mean"
    mean: float = Field(gt=0, lt=1)
    sd: float = Field(gt=0)

    @model_validator(mode="after")
    def _feasible(self):
        limit = math.sqrt(self.mean * (1 - self.mean))
        if self.sd >= limit:
            raise ValueError(
                f"beta_mean sd={self.sd} is impossible for mean={self.mean}; it must be < {limit:.4f}"
            )
        return self

    def _ab(self) -> tuple[float, float]:
        k = self.mean * (1 - self.mean) / self.sd**2 - 1
        return self.mean * k, (1 - self.mean) * k

    def sample(self, rng, n):
        a, b = self._ab()
        return rng.beta(a, b, n)

    def describe(self):
        return Description(self.mean, self.sd, 0.0, 1.0)

    def pdf(self, x):
        a, b = self._ab()
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        inside = (x > 0) & (x < 1)
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        xi = x[inside]
        out[inside] = np.exp((a - 1) * np.log(xi) + (b - 1) * np.log1p(-xi) - lbeta)
        return out


class Fixed(_Spec):
    dist: Literal["fixed"] = "fixed"
    value: float

    def sample(self, rng, n):
        return np.full(n, float(self.value))

    def describe(self):
        return Description(self.value, 0.0, self.value - 1, self.value + 1)

    def pdf(self, x):
        # A point mass has no density; the app marks fixed values separately.
        return np.zeros_like(np.asarray(x, dtype=float))


class Component(BaseModel):
    """One weighted part of a mixture, written flat in YAML: {weight: 0.5, dist: normal, mean: 8, sd: 1}."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    weight: float = Field(gt=0)
    spec: Distribution

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Describe both the flat preset syntax and the nested Python representation."""
        nested = handler(core_schema)
        weight = nested["properties"]["weight"]
        choices = [nested]
        for ref in nested["properties"]["spec"]["oneOf"]:
            variant = handler.resolve_ref_schema(ref)
            choices.append({
                **variant,
                "properties": {**variant["properties"], "weight": weight},
                "required": list(dict.fromkeys(["weight", "dist", *variant.get("required", [])])),
            })
        return {"oneOf": choices}

    @model_validator(mode="before")
    @classmethod
    def _unflatten(cls, data: Any) -> Any:
        if isinstance(data, dict) and "spec" not in data:
            data = dict(data)
            weight = data.pop("weight", None)
            return {"weight": weight, "spec": data}
        return data

    @model_serializer(mode="wrap")
    def _flatten(self, handler):
        d = handler(self)
        if "spec" in d:
            return {"weight": d["weight"], **d["spec"]}
        else:
            # In json mode, spec is already flattened
            return d


class Mixture(_Spec):
    dist: Literal["mixture"] = "mixture"
    components: list[Component] = Field(min_length=2)

    @model_validator(mode="after")
    def _weights(self):
        total = sum(c.weight for c in self.components)
        if abs(total - 1) > 1e-6:
            raise ValueError(f"mixture weights must sum to 1 (got {total:g})")
        return self

    def sample(self, rng, n):
        w = np.array([c.weight for c in self.components])
        which = rng.choice(len(self.components), size=n, p=w / w.sum())
        out = np.empty(n, dtype=float)
        for i, c in enumerate(self.components):
            m = which == i
            out[m] = c.spec.sample(rng, int(m.sum()))
        return out

    def describe(self):
        ds = [c.spec.describe() for c in self.components]
        ws = [c.weight for c in self.components]
        mean = sum(w * d.mean for w, d in zip(ws, ds))
        second = sum(w * (d.sd**2 + d.mean**2) for w, d in zip(ws, ds))
        return Description(mean, math.sqrt(max(second - mean**2, 0.0)), min(d.lo for d in ds), max(d.hi for d in ds))

    def pdf(self, x):
        return sum(c.weight * c.spec.pdf(x) for c in self.components)


Distribution = Annotated[
    Union[Normal, LogNormal, Uniform, Poisson, BetaMean, Fixed, Mixture],
    Field(discriminator="dist"),
]
Component.model_rebuild()
Mixture.model_rebuild()

_ADAPTER: TypeAdapter = TypeAdapter(Distribution)


def parse_distribution(data: Any):
    """Validate a plain dict (from YAML) into a distribution spec."""
    return _ADAPTER.validate_python(data)
