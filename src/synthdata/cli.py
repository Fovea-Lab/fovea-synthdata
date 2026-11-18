"""synthdata command line: list generators, generate, preview, check, and launch the app."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from synthdata import sdk
from synthdata.core import registry
from synthdata.core.generator import GenerationError

APP_MAIN = Path(__file__).parent / "app" / "main.py"

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Synthetic datasets with fully known ground truth.")
presets = typer.Typer(no_args_is_help=True, help="List and export editable generator designs.")
app.add_typer(presets, name="preset")


def default_workers() -> int:
    return min(8, os.cpu_count() or 1)


def _fail(msg: str, code: int) -> None:
    typer.secho(f"error: {msg}", err=True, fg="red")
    raise typer.Exit(code)


def _load(design: Path):
    try:
        return registry.load_design_file(design)[1]
    except ValidationError as exc:
        _fail(f"{design} is not a valid design:\n{exc}", 2)
    except (ValueError, KeyError, ImportError, OSError) as exc:
        _fail(str(exc), 2)


@app.command("list")
def list_cmd() -> None:
    """List the registered generators."""
    for name in sdk.list_generators():
        typer.echo(name)


@app.command()
def describe(generator: str) -> None:
    """Print a generator's settings schema, presets, supported operations and writers as JSON."""
    try:
        typer.echo(json.dumps(sdk.describe_generator(generator), indent=2))
    except (KeyError, ImportError) as exc:
        _fail(str(exc), 2)


@presets.command("list")
def preset_list(generator: Optional[str] = typer.Argument(None)) -> None:
    """List bundled preset IDs. Optionally restrict the list to one generator."""
    try:
        for name in [generator] if generator is not None else sdk.list_generators():
            for preset in sdk.list_presets(name):
                typer.echo(preset)
    except (KeyError, ImportError) as exc:
        _fail(str(exc), 2)


@presets.command("export")
def preset_export(preset: str, out: Path = typer.Option(..., "-o", "--out")) -> None:
    """Write an editable copy of a bundled YAML preset to a new file."""
    try:
        source = sdk.preset_yaml(preset)
        with out.open("x", encoding="utf-8") as dest:
            dest.write(source)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        _fail(str(exc), 2)
    typer.echo(f"preset → {out}")


@app.command()
def generate(
    design: Path = typer.Argument(..., exists=True, dir_okay=False, help="Design YAML file."),
    out: Path = typer.Option(..., "-o", "--out", help="Output folder (must be empty or absent)."),
    writer: Optional[str] = typer.Option(None, "--writer", help="Writer name; default is the generator's."),
    seed: Optional[int] = typer.Option(None, "--seed", min=0, help="Override the design's seed."),
    workers: int = typer.Option(default_workers(), "--workers", min=1, help="Parallel render processes."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace a previous synthdata dataset in OUT."),
) -> None:
    """Generate a dataset from a design file."""
    d = _load(design)
    if seed is not None:
        d = type(d).model_validate({**d.model_dump(mode="json", exclude_none=True), "seed": seed})
    state: dict = {}

    def progress(done: int, total: int) -> None:
        if "bar" not in state:
            state["bar"] = typer.progressbar(length=total, label="rendering", file=sys.stderr)
            state["bar"].__enter__()
            state["done"] = 0
        state["bar"].update(done - state["done"])
        state["done"] = done

    try:
        m = sdk.generate(d, out, writer=writer, workers=workers, progress=progress, overwrite=overwrite)
    except (GenerationError, ValueError, ImportError, OSError) as exc:
        _fail(str(exc), 1)
    finally:
        if "bar" in state:
            state["bar"].__exit__(None, None, None)
    counts = [f"{value} {key[2:].replace('_', ' ')}" for key, value in m.extra.items() if key.startswith("n_")]
    typer.echo(f"wrote {', '.join(counts)} → {out} (writer={m.writer}, seed={m.seed})")


@app.command()
def preview(
    design: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("preview.png"), "-o", "--out"),
    per_condition: int = typer.Option(1, "--per-condition", min=1),
    tile: int = typer.Option(256, "--tile", min=32),
) -> None:
    """Render a contact sheet: one row per condition, a few seconds."""
    d = _load(design)
    try:
        sdk.preview(d, per_condition=per_condition, tile=tile).save(out)
    except (ValueError, ImportError, OSError) as exc:
        _fail(str(exc), 1)
    typer.echo(f"preview → {out}")


@app.command()
def check(dataset: Path = typer.Argument(..., help="A generated dataset folder.")) -> None:
    """Run the recoverability self-check and write check/report.{md,json}. Exit 1 if it fails."""
    try:
        rep = sdk.check(dataset)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        _fail(str(exc), 1)
    for c in rep.stage1 + rep.stage2:
        status = getattr(c, "status", "pass" if c.passed else "fail")
        typer.echo(f"{ {'pass': 'ok  ', 'fail': 'FAIL', 'not_checked': 'not_checked'}[status] }  {c.name}")
    typer.echo(f"{'PASS' if rep.passed else 'FAIL'} — report: {Path(dataset) / 'check' / 'report.md'}")
    if not rep.passed:
        raise typer.Exit(1)


@app.command()
def report(
    dataset: Path = typer.Argument(..., help="A generated particle_plate dataset folder."),
    out: Optional[Path] = typer.Option(None, "-o", "--out", help="Output folder; default DATASET/report."),
    sections: Optional[str] = typer.Option(None, "--sections", help="Comma-separated section keys or numbers."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing report in OUT."),
) -> None:
    """Build the truth report (index.html, figures, tables, report.json) for a dataset."""
    keys = [k for k in sections.split(",") if k.strip()] if sections else None
    try:
        res = sdk.report(dataset, out, sections=keys, overwrite=overwrite)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        _fail(str(exc), 1)
    m = res.metrics
    f = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
    from synthdata.report.summary import twin_status

    typer.echo(f"ARI {f(m['ari'])} (controls merged {f(m.get('ari_control_merged'))}) · NMI {f(m['nmi'])} · mean mAP {f(m['mean_map'])} · "
               f"{twin_status(m)[1]} · {m['n_wells']} wells, {m['n_objects']} objects")
    typer.echo(f"report → {res.out / 'index.html'}")


@app.command("app")
def app_cmd() -> None:
    """Open the local Streamlit app in the browser."""
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP_MAIN)]))


if __name__ == "__main__":
    app()
