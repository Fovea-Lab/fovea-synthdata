"""Public Python interface for discovering, configuring, and running generators."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel

from synthdata.core import registry
from synthdata.core.generator import ProgressFn
from synthdata.core.provenance import Manifest, read_manifest

if TYPE_CHECKING:
    from PIL import Image
    from synthdata.generators.cell_plate.check import Report as CellReport
    from synthdata.generators.particle_plate.check import Report as ParticleReport
    from synthdata.report.build import ReportResult


def list_generators() -> list[str]:
    """Return the names of the available generators."""
    return registry.names()


def _preset_paths(generator: str) -> dict[str, Path]:
    gen = registry.get(generator)
    # Older programmatically registered generators may have no bundled presets.
    return gen.presets() if hasattr(gen, "presets") else {}


def list_presets(generator: str) -> list[str]:
    """Return qualified preset IDs for one generator."""
    return [f"{generator}/{name}" for name in sorted(_preset_paths(generator))]


def describe_generator(name: str) -> dict[str, Any]:
    """Describe the settings (JSON Schema), formats, presets, and operations."""
    gen = registry.get(name)
    capabilities = ["validate", "preview", "generate"]
    if hasattr(gen, "check"):
        capabilities.append("check")
    if hasattr(gen, "report_inputs"):
        capabilities.append("report")
    return {
        "name": gen.name,
        "version": gen.version,
        "presets": list_presets(name),
        "writers": sorted(gen.writers),
        "default_writer": gen.default_writer,
        "capabilities": capabilities,
        "settings_schema": gen.Design.model_json_schema(),
    }


def preset_yaml(preset: str) -> str:
    """Return the original bundled YAML text, including example comments."""
    generator, separator, name = preset.partition("/")
    if not separator or not generator or not name or "/" in name:
        raise ValueError("use a preset ID of the form generator/preset")
    presets = _preset_paths(generator)
    if name not in presets:
        raise KeyError(f"unknown preset {preset!r}; available for {generator}: {', '.join(sorted(presets))}")
    return presets[name].read_text(encoding="utf-8")


def load_preset(preset: str) -> dict[str, Any]:
    """Load an independent, editable copy of the settings in a bundled preset."""
    return yaml.safe_load(preset_yaml(preset))


def validate_design(design: dict | BaseModel) -> BaseModel:
    """Validate the settings and return the generator's typed design with defaults."""
    _, validated = registry.resolve_design(design)
    return validated


@contextmanager
def _dependencies(extra: str):
    try:
        yield
    except ModuleNotFoundError as exc:
        raise ImportError(f"missing dependency {exc.name!r}; install 'fovea-synthdata[{extra}]'") from exc


def preview(design: dict | BaseModel, *, per_condition: int = 1, tile: int = 256) -> Image.Image:
    """Render the configured generator's contact sheet in memory."""
    if per_condition < 1 or tile < 32:
        raise ValueError("per_condition must be at least 1 and tile must be at least 32")
    gen, validated = registry.resolve_design(design)
    with _dependencies(gen.name):
        return gen.contact_sheet(gen.preview(validated, per_condition), tile=tile)


def generate(
    design: dict | BaseModel,
    out: str | Path,
    *,
    writer: str | None = None,
    workers: int = 1,
    progress: ProgressFn | None = None,
    overwrite: bool = False,
) -> Manifest:
    """Write a dataset and return its manifest, including the seed and file checksums."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    gen, validated = registry.resolve_design(design)
    with _dependencies(gen.name):
        return gen.generate(validated, Path(out), writer=writer, workers=workers,
                            progress=progress, overwrite=overwrite)


def check(dataset: str | Path, *, write: bool = True) -> CellReport | ParticleReport:
    """Run the dataset generator's recoverability checks; inspect report.passed."""
    path = Path(dataset)
    gen = registry.get(read_manifest(path).generator)
    if not hasattr(gen, "check"):
        raise ValueError(f"no check is implemented for generator {gen.name!r}")
    with _dependencies(gen.name):
        return gen.check(path, write=write)


def report(dataset: str | Path, out: str | Path | None = None, *,
           sections: list[str] | None = None, overwrite: bool = False) -> ReportResult:
    """Build an HTML truth report for a generator that supports reporting."""
    path = Path(dataset)
    gen = registry.get(read_manifest(path).generator)
    if not hasattr(gen, "report_inputs"):
        raise ValueError(f"no report is implemented for generator {gen.name!r}")
    with _dependencies("report"):
        from synthdata.report import build_report

        return build_report(gen.report_inputs(path), Path(out) if out is not None else path / "report",
                            sections=sections, overwrite=overwrite)
