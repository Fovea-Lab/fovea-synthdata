"""Design schema for the cell_plate generator. Every value that shapes the data lives here."""

from __future__ import annotations

import fnmatch  # noqa: F401  (expectations globs are matched by later tasks)
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthdata.core.distributions import Distribution, parse_distribution
from synthdata.generators.particle_plate.design import ImageSpec, Illumination, Noise, PlateGrid
from synthdata.generators.particle_plate.wells import parse_well, well_name

DESIGNS_DIR = Path(__file__).parent / "designs"
BLANK = "BLANK"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Channel(_M):
    name: str = Field(min_length=1, pattern=r"^[^/\\\x00]+$")
    stain: str
    psf_sigma_px: float = Field(0.7, gt=0)


class Plate(_M):
    format: Literal[96, 384] | PlateGrid = 96
    plates: int = Field(1, ge=1)
    sites_per_well: int = Field(2, ge=1)
    image: ImageSpec = Field(default_factory=lambda: ImageSpec(height=1024, width=1024))
    channels: list[Channel] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_channels(self):
        names = [c.name for c in self.channels]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"plate.channels: duplicate channel name(s) {dup}")
        return self

    def grid(self) -> tuple[int, int]:
        if self.format == 96:
            return 8, 12
        if self.format == 384:
            return 16, 24
        return self.format.rows, self.format.cols


class EdgeEffect(_M):
    enabled: bool = False
    dim: float = Field(0.1, ge=0, lt=1)
    count_factor: float = Field(1.0, gt=0, le=1)


class ImageEffects(_M):
    background: Distribution = Field(default_factory=lambda: parse_distribution({"dist": "fixed", "value": 100.0}))
    noise: Noise = Field(default_factory=Noise)
    illumination: Illumination = Field(default_factory=Illumination)
    edge_effect: EdgeEffect = Field(default_factory=EdgeEffect)


class PlateEffect(_M):
    """What one plate does to every image on it. A channel missing from a dict has gain 1 and shift 0."""

    gain: dict[str, float] = Field(default_factory=dict)  # multiplies the blurred signal, not the background
    background_shift: dict[str, float] = Field(default_factory=dict)  # added to the drawn background
    count_factor: float = Field(1.0, gt=0)  # multiplies the expected cell count of every site


class WellEffects(_M):
    """Random variation between wells. Each non-blank well draws its own values; an SD of 0 leaves that part as is.

    Gain, count and size are lognormal factors with median 1 (the SD is on the log scale). The response multiplier
    is 1 + normal(0, SD), clipped at 0, and scales the strength of the well's main phenotype (not its side effect);
    the scaled strength is capped at 1, so a well never responds past the planted phenotype.
    """

    gain_sd: float = Field(0.0, ge=0)  # per channel, multiplies the blurred signal
    count_sd: float = Field(0.0, ge=0)  # multiplies the expected cell count of every site of the well
    size_sd: float = Field(0.0, ge=0)  # multiplies body and nucleus diameter
    response_sd: float = Field(0.0, ge=0)


class Shape(_M):
    diameter: Distribution
    eccentricity: Distribution
    lumpiness: Distribution


class Body(Shape):
    nucleus_offset: Distribution  # fraction of the body radius


class ChannelBase(_M):
    intensity: Distribution


class BaseCell(_M):
    count: Distribution
    overlap_fraction: float = Field(0.15, ge=0, le=1)
    nucleus: Shape
    body: Body
    channels: dict[str, ChannelBase] = Field(min_length=1)


class Phenotype(_M):
    """Multipliers (`*_factor`) and offsets (`*_add`) from the base cell, at strength 1. Neutral by default."""

    count_factor: float = 1.0
    nucleus_diameter_factor: float = 1.0
    body_diameter_factor: float = 1.0
    body_eccentricity_add: float = 0.0
    body_lumpiness_factor: float = 1.0
    dna_intensity_factor: float = 1.0
    dna_foci_add: float = 0.0
    nucleoli_count_factor: float = 1.0
    rna_intensity_factor: float = 1.0
    agp_fibre_factor: float = 1.0
    mito_length_factor: float = 1.0
    mito_count_factor: float = 1.0
    multinucleate_add: float = 0.0

    def at(self, strength: float) -> Phenotype:
        """The phenotype scaled by dose strength: a factor f becomes 1 + s(f-1), never below 0; an add a becomes s*a."""
        out: dict[str, float] = {}
        for name in type(self).model_fields:
            v = float(getattr(self, name))
            out[name] = strength * v if name.endswith("_add") else max(1.0 + strength * (v - 1.0), 0.0)
        return Phenotype(**out)

    def combined(self, other: Phenotype) -> Phenotype:
        """Both effects at once: factors multiply, adds add. Neutral on either side leaves the other unchanged."""
        out: dict[str, float] = {}
        for name in type(self).model_fields:
            a, b = float(getattr(self, name)), float(getattr(other, name))
            out[name] = a + b if name.endswith("_add") else a * b
        return Phenotype(**out)


class SideEffect(_M):
    """A second phenotype a compound carries, at `strength` times the dose strength (not scaled by potency)."""

    phenotype: str
    strength: float = Field(gt=0, le=1)


class Compound(_M):
    role: Literal["negative_control", "positive_control", "test"] = "test"
    phenotype: str | None = None
    dose: float | None = Field(None, ge=0)
    potency: float | None = Field(None, gt=0)  # scales the main phenotype's dose strength
    side_effect: SideEffect | None = None
    mechanism: str | None = None
    # each cell takes the main phenotype with this probability; the side effect reaches every cell
    responder_fraction: float | None = Field(None, gt=0, le=1)

    @model_validator(mode="after")
    def _role_fields(self):
        if self.role == "negative_control" and self.phenotype is not None:
            raise ValueError("a negative_control has no phenotype")
        if self.role != "negative_control" and self.phenotype is None:
            raise ValueError(f"a {self.role} compound needs a phenotype")
        if self.role == "positive_control" and self.dose is None:
            raise ValueError("a positive_control needs a dose (the strength it is applied at)")
        if self.role == "test" and self.dose is not None:
            raise ValueError("a test compound takes its strength from doses, not a dose")
        if self.role != "test":
            for field in ("potency", "side_effect", "mechanism", "responder_fraction"):
                if getattr(self, field) is not None:
                    raise ValueError(f"a {self.role} takes no {field}")
        return self


class Dose(_M):
    label: str
    strength: float = Field(ge=0)


class Constraints(_M):
    no_shared_row_or_col: bool = True
    doses_not_adjacent: bool = True


class Layout(_M):
    kind: Literal["controlled_random"] = "controlled_random"
    fixed: dict[str, str] = Field(default_factory=dict)  # well -> compound name
    blanks: int = Field(0, ge=0)
    replicates: int = Field(1, ge=1)
    constraints: Constraints = Field(default_factory=Constraints)


class BlurredSites(_M):
    n: int = Field(ge=0)
    psf_sigma_px: float = Field(gt=0)


class SaturatedDebris(_M):
    n: int = Field(ge=0)


class QCDefects(_M):
    blurred_sites: BlurredSites | None = None
    saturated_debris: SaturatedDebris | None = None


class Differs(_M):
    feature: str  # may be a glob, e.g. Nuclei_Texture_Contrast_DNA*
    direction: Literal["up", "down"]


class Expectations(_M):
    reference: str
    same: list[str] = Field(default_factory=list)
    differs: dict[str, list[Differs]] = Field(default_factory=dict)


class CellPlateDesign(_M):
    generator: Literal["cell_plate"] = "cell_plate"
    seed: int = Field(0, ge=0)
    plate: Plate
    image_effects: ImageEffects = Field(default_factory=ImageEffects)
    plate_effects: list[PlateEffect] | None = None  # one per plate, in plate order
    well_effects: WellEffects | None = None
    base_cell: BaseCell
    phenotypes: dict[str, Phenotype] = Field(default_factory=dict)
    compounds: dict[str, Compound] = Field(min_length=1)
    doses: dict[str, Dose] = Field(default_factory=dict)
    layout: Layout = Field(default_factory=Layout)
    qc_defects: QCDefects | None = None
    expectations: Expectations | None = None

    @model_validator(mode="after")
    def _consistent(self):
        channels = {c.name for c in self.plate.channels}
        if self.plate_effects is not None:
            if len(self.plate_effects) != self.plate.plates:
                raise ValueError(f"plate_effects: {len(self.plate_effects)} entries for {self.plate.plates} plate(s); "
                                 "give one per plate")
            for i, eff in enumerate(self.plate_effects):
                for field_, keys in (("gain", eff.gain), ("background_shift", eff.background_shift)):
                    bad = sorted(set(keys) - channels)
                    if bad:
                        raise ValueError(f"plate_effects[{i}].{field_}: unknown channel(s) {bad}; "
                                         f"known: {sorted(channels)}")
        missing = channels - set(self.base_cell.channels)
        if missing:
            raise ValueError(f"base_cell.channels: no intensity for plate channel(s) {sorted(missing)}")
        for name, c in self.compounds.items():
            if c.phenotype is not None and c.phenotype not in self.phenotypes:
                raise ValueError(f"compounds.{name}.phenotype: unknown phenotype {c.phenotype!r}; "
                                 f"known: {sorted(self.phenotypes)}")
            if c.side_effect is not None and c.side_effect.phenotype not in self.phenotypes:
                raise ValueError(f"compounds.{name}.side_effect.phenotype: unknown phenotype "
                                 f"{c.side_effect.phenotype!r}; known: {sorted(self.phenotypes)}")
        if any(c.role == "test" for c in self.compounds.values()) and not self.doses:
            raise ValueError("doses: test compounds need at least one dose")
        rows, cols = self.plate.grid()
        seen: dict[tuple[int, int], str] = {}
        for well, comp in self.layout.fixed.items():
            try:
                r, c = parse_well(well)
            except ValueError as exc:
                raise ValueError(f"layout.fixed: {exc}") from None
            if r >= rows or c >= cols:
                raise ValueError(f"layout.fixed: {well} is outside the {rows}x{cols} plate")
            if well != well_name(r, c):
                raise ValueError(f"layout.fixed: {well!r} is not canonical; write {well_name(r, c)!r}")
            if (r, c) in seen:
                raise ValueError(f"layout.fixed: {well!r} appears twice")
            seen[(r, c)] = well
            if comp not in self.compounds:
                raise ValueError(f"layout.fixed: {well} names unknown compound {comp!r}; "
                                 f"known: {sorted(self.compounds)}")
        n_free = sum(1 for c in self.compounds.values() if c.role == "test") * len(self.doses) * self.layout.replicates
        if len(seen) + self.layout.blanks + n_free > rows * cols:
            raise ValueError(f"layout: {len(seen)} fixed + {self.layout.blanks} blank + {n_free} compound wells "
                             f"> {rows * cols} wells on the plate")
        if self.expectations is not None:
            e = self.expectations
            bad = {e.reference, *e.same, *e.differs} - set(self.compounds)
            if bad:
                names = "reference" if e.reference in bad else "same/differs"
                raise ValueError(f"expectations.{names}: unknown compound(s) {sorted(bad)}; "
                                 f"known: {sorted(self.compounds)}")
        return self


def load_design(source: Any) -> CellPlateDesign:
    """Accept a design object, a dict, a YAML string or a path to a YAML file."""
    if isinstance(source, CellPlateDesign):
        return source
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source
                                    and source.endswith((".yaml", ".yml"))):
        source = yaml.safe_load(Path(source).read_text())
    elif isinstance(source, str):
        source = yaml.safe_load(source)
    return CellPlateDesign.model_validate(source)


def dump_design(design: CellPlateDesign) -> str:
    return yaml.safe_dump(design.model_dump(mode="json", exclude_none=True), sort_keys=False)


def bundled_designs() -> dict[str, Path]:
    return {p.stem: p for p in sorted(DESIGNS_DIR.glob("*.yaml"))}
