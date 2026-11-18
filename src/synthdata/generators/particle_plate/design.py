"""Design schema for the particle_plate generator. Every value that shapes the data lives here."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthdata.core.distributions import Distribution, parse_distribution
from synthdata.generators.particle_plate.wells import parse_well, well_name

PARTICLE_FIELDS = ("count", "arrangement", "diameter", "eccentricity", "orientation", "intensity",
                   "profile", "edge_softness")
DIST_FIELDS = ("count", "diameter", "eccentricity", "orientation", "intensity", "edge_softness")
EFFECT_FIELDS = ("background", "noise", "illumination", "edge_effect")
MEASURED_PARAMS = ("diameter", "eccentricity", "orientation", "intensity", "edge_softness")
Feature = Literal["diameter", "eccentricity", "orientation_spread", "intensity", "count", "nn_distance",
                  "bimodal_diameter"]
FEATURES: tuple[str, ...] = Feature.__args__

DESIGNS_DIR = Path(__file__).parent / "designs"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlateGrid(_M):
    rows: int = Field(ge=1, le=26)
    cols: int = Field(ge=1, le=99)


class ImageSpec(_M):
    height: int = Field(ge=16)
    width: int = Field(ge=16)
    dtype: Literal["uint8", "uint16"] = "uint16"
    pixel_size_um: float = Field(0.65, gt=0)


class Channel(_M):
    name: str


class Plate(_M):
    format: Literal[96, 384] | PlateGrid = 96
    plates: int = Field(1, ge=1)
    sites_per_well: int = Field(4, ge=1)
    image: ImageSpec = Field(default_factory=lambda: ImageSpec(height=1024, width=1024))
    channels: list[Channel] = Field(default_factory=lambda: [Channel(name="particles")], min_length=1)

    @model_validator(mode="after")
    def _one_channel(self):
        if len(self.channels) != 1:
            raise ValueError("particle_plate renders one channel today; multi-channel designs are not implemented yet")
        return self

    def grid(self) -> tuple[int, int]:
        if self.format == 96:
            return 8, 12
        if self.format == 384:
            return 16, 24
        return self.format.rows, self.format.cols


class Noise(_M):
    poisson: bool = True
    gaussian_sd: float = Field(0.0, ge=0)


class Illumination(_M):
    kind: Literal["none", "radial", "linear"] = "none"
    strength: float = Field(0.0, ge=0, lt=1)
    angle: float = 0.0  # degrees, for kind=linear


class EdgeEffect(_M):
    enabled: bool = False
    dim: float = Field(0.1, ge=0, lt=1)


def _fixed(v: float):
    return parse_distribution({"dist": "fixed", "value": v})


class ImageEffects(_M):
    background: Distribution = Field(default_factory=lambda: _fixed(100.0))
    noise: Noise = Field(default_factory=Noise)
    illumination: Illumination = Field(default_factory=Illumination)
    edge_effect: EdgeEffect = Field(default_factory=EdgeEffect)


class EffectsOverride(_M):
    background: Distribution | None = None
    noise: Noise | None = None
    illumination: Illumination | None = None
    edge_effect: EdgeEffect | None = None


class Arrangement(_M):
    kind: Literal["uniform", "clumped"] = "uniform"
    clusters: int = Field(5, ge=1)
    spread: float = Field(50.0, gt=0)


class Profile(_M):
    kind: Literal["flat", "gaussian", "ring"] = "flat"
    sigma_frac: float = Field(0.35, gt=0)
    width_frac: float = Field(0.3, gt=0, le=1)


class Particle(_M):
    count: Distribution
    arrangement: Arrangement = Field(default_factory=Arrangement)
    diameter: Distribution
    eccentricity: Distribution
    orientation: Distribution
    intensity: Distribution
    profile: Profile = Field(default_factory=Profile)
    edge_softness: Distribution


class Condition(_M):
    role: Literal["control", "test"] = "test"
    count: Distribution | None = None
    arrangement: Arrangement | None = None
    diameter: Distribution | None = None
    eccentricity: Distribution | None = None
    orientation: Distribution | None = None
    intensity: Distribution | None = None
    profile: Profile | None = None
    edge_softness: Distribution | None = None
    image_effects: EffectsOverride | None = None


class PairCheck(_M):
    a: str
    b: str
    differs: list[Feature] = Field(default_factory=list)
    same: list[Feature] = Field(default_factory=list)


class Expectations(_M):
    """What the recoverability check asserts. Optional: without it the check reports but does not assert."""

    reference: str
    same: list[str] = Field(default_factory=list)
    differs: dict[str, list[Feature]] = Field(default_factory=dict)
    pairs: list[PairCheck] = Field(default_factory=list)

    def condition_names(self) -> set[str]:
        return {self.reference, *self.same, *self.differs, *(p.a for p in self.pairs), *(p.b for p in self.pairs)}


class Layout(_M):
    kind: Literal["by_row", "by_column", "explicit", "randomized"] = "by_row"
    order: list[str] | None = None
    wells: dict[str, str] = Field(default_factory=dict)
    seed: int | None = Field(None, ge=0)


class AggregationAssign(_M):
    kind: Literal["random", "by_condition", "explicit", "none"] = "random"
    fraction: float = Field(0.15, ge=0, le=1)
    conditions: dict[str, float] = Field(default_factory=dict)
    units: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ranges(self):
        for k, v in {**{f"conditions.{k}": v for k, v in self.conditions.items()},
                     **{f"units.{k}": v for k, v in self.units.items()}}.items():
            if not 0 <= v <= 1:
                raise ValueError(f"assign.{k}: must be in [0, 1], got {v:g}")
        return self


class AggregationDefect(_M):
    level: Literal["well", "image"] = "well"
    assign: AggregationAssign = Field(default_factory=AggregationAssign)
    degree: Distribution = Field(default_factory=lambda: parse_distribution({"dist": "uniform", "low": 0.1, "high": 0.9}))
    clusters: Distribution = Field(default_factory=lambda: parse_distribution({"dist": "poisson", "mean": 4}))
    spread: Distribution = Field(default_factory=lambda: parse_distribution({"dist": "normal", "mean": 25, "sd": 5}))

    @model_validator(mode="after")
    def _degree(self):
        d = self.degree.describe()
        if not 0 <= d.mean <= 1:
            raise ValueError(f"aggregation.degree: must lie in [0, 1], got mean {d.mean:g}")
        return self


class QCThresholds(_M):
    warn: float = 0.2
    fail: float = 0.5

    @model_validator(mode="after")
    def _order(self):
        if not 0 <= self.warn < self.fail <= 1:
            raise ValueError(f"thresholds: need 0 <= warn < fail <= 1, got warn={self.warn:g}, fail={self.fail:g}")
        return self


class QCDefects(_M):
    aggregation: AggregationDefect | None = None
    thresholds: QCThresholds = Field(default_factory=QCThresholds)


def _check_particle(p: Particle, where: str) -> None:
    e = p.eccentricity.describe().mean
    if not 0 <= e < 1:
        raise ValueError(f"{where}.eccentricity: mean must be in [0, 1), got {e:g}")
    if p.diameter.describe().mean <= 0:
        raise ValueError(f"{where}.diameter: mean must be > 0")
    for name in ("count", "intensity", "edge_softness"):
        if getattr(p, name).describe().mean < 0:
            raise ValueError(f"{where}.{name}: mean must be >= 0")


class ParticlePlateDesign(_M):
    generator: Literal["particle_plate"] = "particle_plate"
    seed: int = Field(0, ge=0)
    plate: Plate = Field(default_factory=Plate)
    image_effects: ImageEffects = Field(default_factory=ImageEffects)
    base_particle: Particle
    conditions: dict[str, Condition] = Field(min_length=1)
    layout: Layout = Field(default_factory=Layout)
    expectations: Expectations | None = None
    qc_defects: QCDefects | None = None

    @model_validator(mode="after")
    def _consistent(self):
        known = set(self.conditions)
        for name in self.conditions:
            _check_particle(self.particle_for(name), f"conditions.{name}")
        order = self.layout.order or list(self.conditions)
        unknown = {c for c in order if c not in known} | {c for c in self.layout.wells.values() if c not in known}
        if unknown:
            raise ValueError(f"layout names unknown condition(s) {sorted(unknown)}; known: {sorted(known)}")
        rows, cols = self.plate.grid()
        kind = self.layout.kind
        if kind == "by_row" and len(order) > rows:
            raise ValueError(f"layout by_row puts one condition per row, but {len(order)} conditions > {rows} rows")
        if kind == "by_column" and len(order) > cols:
            raise ValueError(f"layout by_column puts one condition per column, but {len(order)} conditions > {cols} columns")
        if kind == "randomized" and len(order) > rows * cols:
            raise ValueError(f"layout randomized: {len(order)} conditions > {rows * cols} wells")
        if kind == "explicit":
            if not self.layout.wells:
                raise ValueError("layout places no wells: an explicit layout needs at least one entry in layout.wells")
            seen: dict[tuple[int, int], str] = {}
            for w in self.layout.wells:
                try:
                    pos = parse_well(w)
                except ValueError:
                    continue  # reported below
                if pos in seen:
                    raise ValueError(f"layout.wells: {seen[pos]!r} and {w!r} are the same well")
                seen[pos] = w
        for w in self.layout.wells:
            try:
                r, c = parse_well(w)
            except ValueError as exc:
                raise ValueError(f"layout.wells: {exc}") from None
            if r >= rows or c >= cols:
                raise ValueError(f"layout.wells: {w} is outside the {rows}×{cols} plate")
        if self.expectations is not None:
            bad = self.expectations.condition_names() - known
            if bad:
                raise ValueError(f"expectations name unknown condition(s) {sorted(bad)}; known: {sorted(known)}")
        self._check_qc(known, rows, cols)
        return self

    def _check_qc(self, known: set[str], rows: int, cols: int) -> None:
        agg = self.qc_defects.aggregation if self.qc_defects else None
        if agg is None:
            return
        bad = set(agg.assign.conditions) - known
        if bad:
            raise ValueError(f"qc_defects.aggregation.assign.conditions: unknown condition(s) {sorted(bad)}; known: {sorted(known)}")
        if not agg.assign.units:
            return
        from synthdata.generators.particle_plate.layout import assign  # lazy: layout imports this module

        populated = {(s.row, s.col) for s in assign(self)}
        image = agg.level == "image"
        where = "qc_defects.aggregation.assign.units"
        form = f"'A01:s2' with site in 1..{self.plate.sites_per_well}" if image else "'A01'"
        seen: set[str] = set()
        for u in agg.assign.units:
            well, sep, site = u.partition(":")
            try:
                r, c = parse_well(well)
            except ValueError:
                raise ValueError(f"{where}: {u!r} is not a canonical unit key; use the form {form}") from None
            canon = well_name(r, c)
            if image:
                ok = (sep == ":" and site.startswith("s") and site.isascii() and site[1:].isdigit()
                      and 1 <= int(site[1:]) <= self.plate.sites_per_well and site == f"s{int(site[1:])}")
                key = f"{canon}:{site}"
            else:
                ok = not sep
                key = canon
            if not ok or key != u:
                raise ValueError(f"{where}: {u!r} is not a canonical unit key; use the form {form}")
            if key in seen:
                raise ValueError(f"{where}: duplicate unit {u!r}")
            seen.add(key)
            if r >= rows or c >= cols:
                raise ValueError(f"{where}: {u!r} is outside the {rows}×{cols} plate")
            if (r, c) not in populated:
                raise ValueError(f"{where}: {u!r} names a well that is empty in the layout")

    def particle_for(self, condition: str) -> Particle:
        c = self.conditions[condition]
        updates = {f: getattr(c, f) for f in PARTICLE_FIELDS if getattr(c, f) is not None}
        return self.base_particle.model_copy(update=updates)

    def effects_for(self, condition: str) -> ImageEffects:
        o = self.conditions[condition].image_effects
        if o is None:
            return self.image_effects
        return self.image_effects.model_copy(update={f: getattr(o, f) for f in EFFECT_FIELDS if getattr(o, f) is not None})

    def expanded(self) -> ParticlePlateDesign:
        """The same design with every condition written out in full — what design.yaml records."""
        conds = {}
        for name, c in self.conditions.items():
            p, fx = self.particle_for(name), self.effects_for(name)
            conds[name] = Condition(
                role=c.role,
                **{f: getattr(p, f) for f in PARTICLE_FIELDS},
                image_effects=EffectsOverride(**{f: getattr(fx, f) for f in EFFECT_FIELDS}),
            )
        return self.model_copy(update={"conditions": conds})


def load_design(source: Any) -> ParticlePlateDesign:
    """Accept a design object, a dict, a YAML string or a path to a YAML file."""
    if isinstance(source, ParticlePlateDesign):
        return source
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source
                                    and source.endswith((".yaml", ".yml"))):
        source = yaml.safe_load(Path(source).read_text())
    elif isinstance(source, str):
        source = yaml.safe_load(source)
    return ParticlePlateDesign.model_validate(source)


def dump_design(design: ParticlePlateDesign) -> str:
    return yaml.safe_dump(design.model_dump(mode="json", exclude_none=True), sort_keys=False)


def bundled_designs() -> dict[str, Path]:
    return {p.stem: p for p in sorted(DESIGNS_DIR.glob("*.yaml"))}
