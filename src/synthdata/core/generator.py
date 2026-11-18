"""The contract every generator implements, so the CLI and the app can drive any of them."""

from __future__ import annotations

import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel

from synthdata.core.provenance import MANIFEST_NAME, Manifest

ProgressFn = Callable[[int, int], None]

SENTINEL = ".synthdata"


class GenerationError(RuntimeError):
    """A generation run failed. The message names what failed; no manifest is written."""


class Writer(Protocol):
    name: str

    def open(self, out: Path, design: Any, layout: Any) -> None: ...
    def write(self, item: Any) -> None: ...
    def close(self) -> None: ...


def writer_factory(module: str, class_name: str) -> Callable[[], Writer]:
    """Load a writer only when selected. Catalog discovery needs no image I/O dependencies."""
    def create() -> Writer:
        return getattr(import_module(module), class_name)()

    return create


@runtime_checkable
class Generator(Protocol):
    name: str
    version: str
    Design: type[BaseModel]
    writers: Mapping[str, Callable[[], Writer]]
    default_writer: str

    def load_design(self, source: Any) -> BaseModel: ...
    def dump_design(self, design: BaseModel) -> str: ...
    def preview(self, design: BaseModel, n_per_condition: int = 1) -> Any: ...
    def contact_sheet(self, preview: Any, tile: int = 256) -> Any: ...
    def generate(self, design: BaseModel, out: Path, writer: str | None = None, workers: int = 1,
                 progress: ProgressFn | None = None, overwrite: bool = False) -> Manifest: ...


def prepare_output(out: Path, overwrite: bool) -> None:
    """Create ``out``. Refuse a non-empty folder unless ``overwrite`` AND it looks like a synthdata dataset."""
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise GenerationError(f"{out} exists and is not a folder")
    if out.is_dir() and any(out.iterdir()):
        if not overwrite:
            raise GenerationError(f"{out} is not empty; pass overwrite=True (CLI: --overwrite) to replace it")
        if not (out / SENTINEL).is_file():
            raise GenerationError(
                f"{out} is not empty and does not look like a synthdata dataset; refusing to delete it"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / SENTINEL).write_text("synthdata dataset folder — safe for `--overwrite` to replace\n")
