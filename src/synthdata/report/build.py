"""Orchestration: compute once, run the sections in order, write index.html and report.json last."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from typing import NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd

import synthdata
from synthdata.report import embed
from synthdata.report.context import FigureRecord, ReportContext, SectionResult, TableRecord
from synthdata.report.contract import ReportError, ReportInputs, inputs_sha256
from synthdata.report.html import render_html
from synthdata.report.sections import QC_SECTION, SECTIONS, caption_for, clustering, objects
from synthdata.report.stats import replicate_map


@dataclass
class ReportResult:
    out: Path
    sections: list[SectionResult]
    metrics: dict
    notes: list[str] = field(default_factory=list)


def resolve_sections(sections) -> list[str] | None:
    """Accept section keys or their 1-based numbers; None means all."""
    if sections is None:
        return None
    keys = [k for k, *_ in [*SECTIONS, QC_SECTION]]
    out = []
    for s in sections:
        s = str(s).strip()
        if s.isdigit() and 1 <= int(s) <= len(keys):
            s = keys[int(s) - 1]
        if s not in keys:
            raise ReportError(f"unknown section {s!r}; choose from {keys} or 1-{len(keys)}")
        out.append(s)
    return out


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


ARTEFACTS = ("index.html", "report.json", "figures", "tables")


def _check_out(out: Path, sections) -> bool:
    """Refuse unsafe targets without touching anything. True = partial rebuild over an existing report."""
    if out.exists() and not out.is_dir():
        raise ReportError(f"{out} exists and is not a folder")
    if not out.exists() or not any(out.iterdir()):
        return False
    if not (out / "report.json").is_file():
        raise ReportError(f"{out} is not empty and not a report (no report.json); refusing to touch it")
    return sections is not None


def _clear_report(out: Path) -> None:
    """Delete only the known report artefacts; every other file in OUT is left alone."""
    for name in ARTEFACTS:
        p = out / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def _load_previous(out: Path, inputs: ReportInputs) -> dict[str, SectionResult]:
    try:
        rj = json.loads((out / "report.json").read_text())
        prev = {}
        for s in rj["sections"]:
            prev[s["key"]] = SectionResult(
                s["key"], s["title"], s["caption"],
                [FigureRecord(**f) for f in s["figures"]], [TableRecord(**t) for t in s["tables"]],
                list(s["notes"]))
        stamp = (rj["inputs_sha256"], rj["synthdata_version"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise ReportError(f"{out / 'report.json'} is not a readable report ({type(exc).__name__}); "
                          "rebuild everything with --overwrite") from exc
    if stamp != (inputs_sha256(inputs), synthdata.__version__):
        raise ReportError("the existing report was built from a different dataset or synthdata version; "
                          "rebuild everything with --overwrite")
    return prev


def _safe_msg(exc: Exception, root: Path) -> str:
    """Exception text with the report folder and any other absolute directory stripped to bare names."""
    msg = f"{type(exc).__name__}: {exc}"
    msg = msg.replace(str(Path(root).resolve()) + "/", "").replace(str(root) + "/", "")
    return re.sub(r"(?<![\w.])/(?:[^\s/'\"]+/)+([^\s/'\"]+)", r"\1", msg)


def _metrics(ctx: ReportContext) -> dict:
    inp = ctx.inputs
    clus = ctx.clustering or {}
    emb = ctx.embedding or {}
    m = replicate_map(ctx.z, ctx.profiles["condition"], ctx.columns)
    map_ = {str(r.condition): float(r.mAP) for r in m.itertuples()}
    roles = inp.wells.drop_duplicates("condition").set_index(inp.wells.drop_duplicates("condition")["condition"]
                                                            .astype(str))["role"].astype(str).to_dict()
    distinct = [v for c, v in map_.items() if roles.get(c) != "control"] or list(map_.values())
    eff = ctx.effects
    flagged = {c: sorted(eff.loc[(eff["condition"] == c) & eff["flagged"], "column"].tolist())
               for c in ctx.conditions if c != inp.reference}
    twins = [c for c in flagged if roles.get(c) == "control"]
    testable = any(flagged[c] for c in flagged if roles.get(c) != "control")
    ari_merged = None
    if clus and ctx.embedding and ctx.embedding.get("ward") is not None:
        ctrl = sorted({c for c, r in roles.items() if r == "control"})
        merged = ctx.profiles["condition"].astype(str).map(lambda c: "__control__" if c in ctrl else c).to_numpy()
        k = len(ctx.conditions) - (len(ctrl) - 1 if ctrl else 0)
        sc = embed.cluster_scores(ctx.embedding["ward"][0], merged, k) if k >= 2 else None
        ari_merged = sc["ari"] if sc else None
    return {
        "ari": clus.get("ari"), "nmi": clus.get("nmi"), "k": emb.get("k"),
        "map": map_, "mean_map": float(np.mean(distinct)) if distinct else None,
        "flagged": flagged, "twin_flagged": any(flagged[c] for c in twins) if twins else None,
        "twin_testable": bool(testable), "ari_control_merged": ari_merged,
        "n_wells": int(len(ctx.profiles)), "n_objects": int(len(inp.objects)),
    }


def _null_metrics(ctx: ReportContext) -> dict:
    return {"ari": None, "nmi": None, "k": None, "map": {}, "mean_map": None, "flagged": {},
            "twin_flagged": None, "twin_testable": False, "ari_control_merged": None,
            "n_wells": int(len(ctx.profiles)), "n_objects": int(len(ctx.inputs.objects))}


class ReportMetrics(NamedTuple):
    metrics: dict
    effects: "pd.DataFrame"


def compute_metrics(inputs: ReportInputs) -> ReportMetrics:
    """The report's metrics and per-condition effects, with no files written (one path shared with build_report)."""
    ctx = ReportContext.create(inputs, Path("."))
    return _compute(ctx, [])


def _compute(ctx: ReportContext, notes: list[str], root: Path | None = None) -> ReportMetrics:
    root = root if root is not None else Path(".")
    try:
        clustering.compute(ctx)
    except Exception as exc:  # noqa: BLE001 - a failed analysis becomes a note, never a crash
        notes.append(f"embedding failed: {_safe_msg(exc, root)}")
        ctx.embedding, ctx.clustering = {}, {}
    try:
        metrics = _clean(_metrics(ctx))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"metrics failed: {_safe_msg(exc, root)}")
        metrics = _null_metrics(ctx)
    return ReportMetrics(metrics, ctx.effects.copy())


def build_report(inputs: ReportInputs, out: Path, sections=None, overwrite: bool = False) -> ReportResult:
    ctx = ReportContext.create(inputs, Path(out))  # validates first
    out = Path(out)
    wanted = resolve_sections(sections)
    in_place = _check_out(out, wanted)
    if out.exists() and any(out.iterdir()) and not in_place and not overwrite:
        raise ReportError(f"{out} already contains a report; pass --overwrite to replace it")
    if wanted and "qc" in wanted and ctx.inputs.qc is None:
        raise ReportError("section 'qc' needs a qc table, and these inputs have none")
    prev = _load_previous(out, ctx.inputs) if in_place else {}
    notes: list[str] = list(ctx.z_notes)

    # computed once, shared by sections 5-7 and the metrics
    metrics = _compute(ctx, notes, out).metrics
    try:
        objects.compute(ctx)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"object sample failed: {_safe_msg(exc, out)}")
        ctx.object_sample, ctx.object_embedding = None, None

    # only now is the old report replaced: a crash above leaves it intact
    out.mkdir(parents=True, exist_ok=True)
    if not in_place:
        _clear_report(out)
    results: list[SectionResult] = []
    for key, title, caption, fn in [*SECTIONS, *([QC_SECTION] if ctx.inputs.qc is not None else [])]:
        if wanted is not None and key not in wanted and key in prev:
            results.append(prev[key])
            continue
        for sub in ("figures", "tables"):
            shutil.rmtree(out / sub / key, ignore_errors=True)
        try:
            res = fn(ctx)
        except Exception as exc:  # noqa: BLE001
            res = SectionResult(key, title, caption_for(ctx.inputs, key),
                                notes=[f"{key} failed: {_safe_msg(exc, out)}"])
        results.append(res)

    (out / "index.html").write_text(render_html(ctx, results, metrics), encoding="utf-8")
    doc = {
        "synthdata_version": synthdata.__version__,
        "source": inputs.source,
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "inputs_sha256": inputs_sha256(ctx.inputs),
        "reference": inputs.reference,
        "features": [{"name": f.name, "kind": f.kind, "label": f.label, "unit": f.unit} for f in inputs.features],
        "notes": notes,
        "sections": [{"key": r.key, "title": r.title, "caption": r.caption,
                      "figures": [f.__dict__ for f in r.figures], "tables": [t.__dict__ for t in r.tables],
                      "notes": r.notes} for r in results],
        "metrics": metrics,
    }
    (out / "report.json").write_text(json.dumps(_clean(doc), sort_keys=True, indent=2, allow_nan=False))
    return ReportResult(out=out, sections=results, metrics=metrics, notes=notes)
