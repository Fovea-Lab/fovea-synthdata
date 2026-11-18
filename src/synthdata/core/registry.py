"""Name → generator. Built-ins import lazily, so `synthdata list` works without a generator's extras."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

_BUILTINS: dict[str, str] = {
    "particle_plate": "synthdata.generators.particle_plate:ParticlePlateGenerator",
    "cell_plate": "synthdata.generators.cell_plate:CellPlateGenerator",
}
_custom: dict[str, Any] = {}


def register(generator: Any) -> None:
    _custom[generator.name] = generator


def names() -> list[str]:
    return sorted(set(_BUILTINS) | set(_custom))


def get(name: str) -> Any:
    if name in _custom:
        return _custom[name]
    if name not in _BUILTINS:
        raise KeyError(f"unknown generator {name!r}; known: {', '.join(names())}")
    module, cls = _BUILTINS[name].split(":")
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"generator {name!r} needs extra dependencies: `uv sync --extra {name}` "
            f"(or `pip install 'fovea-synthdata[{name}]'`)"
        ) from exc
    return getattr(mod, cls)()


def resolve_design(design: dict | BaseModel) -> tuple[Any, BaseModel]:
    """Select the generator by its declared name and validate all settings."""
    if isinstance(design, BaseModel):
        design = design.model_dump(mode="json", exclude_none=True)
    if not isinstance(design, dict) or not isinstance(design.get("generator"), str):
        raise ValueError("a design needs a top-level 'generator:' string")
    gen = get(design["generator"])
    return gen, gen.load_design(design)


def load_design_file(path: str | Path) -> tuple[Any, Any]:
    """Read a design file; its top-level `generator:` key picks the generator that validates it."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    return resolve_design(data)
