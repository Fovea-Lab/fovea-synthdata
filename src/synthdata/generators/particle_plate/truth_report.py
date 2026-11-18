"""Adapter: a particle_plate dataset's ground truth as report inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthdata.core.provenance import read_manifest, sha256_bytes
from synthdata.generators.particle_plate.design import load_design
from synthdata.generators.particle_plate.reader import read_image
from synthdata.report.contract import DEFAULT_FEATURES, ReportInputs

FEATURES = ("diameter", "eccentricity", "orientation", "intensity")


NOTES_FOR_READER = {
    "distributions": "Only aligned should show a lobe in the orientation roses.",
    "fingerprints": "The twin (control_twin) should show no dot.",
    "clustering": "The twin should sit inside the control.",
    "objects": ("Look for two separated diameter peaks in two_sizes, a high-eccentricity cloud for elongated and "
                "aligned, and large shifted to bigger diameters."),
}


def truth_inputs(dataset: Path) -> ReportInputs:
    dataset = Path(dataset)
    design = load_design(dataset / "design.yaml")
    particles = pd.read_parquet(dataset / "truth" / "particles.parquet")
    wells = pd.read_parquet(dataset / "truth" / "wells.parquet")
    manifest = read_manifest(dataset)

    objects = particles.rename(columns={"particle_id": "object_id"})[
        ["plate", "well", "site", "object_id", "x", "y", *FEATURES]].reset_index(drop=True)
    wells = wells[["plate", "well", "row", "col", "condition", "role", "replicate"]]
    rank = {c: i for i, c in enumerate(design.conditions)}
    wells = (wells.assign(_rank=wells["condition"].map(rank))
             .sort_values(["_rank", "plate", "row", "col"], kind="stable")
             .drop(columns="_rank").reset_index(drop=True))

    if design.expectations is not None:
        reference = design.expectations.reference
    else:
        reference = next(c for c, cond in design.conditions.items() if cond.role == "control")

    intended = {f: {c: getattr(design.particle_for(c), f) for c in design.conditions} for f in FEATURES}

    images = {}
    for c in design.conditions:
        first = wells[wells["condition"] == c].iloc[0]
        images[c] = read_image(dataset, first["plate"], first["well"], 1)[0]

    qc = None
    qc_path = dataset / "truth" / "qc.parquet"
    if design.qc_defects is not None and qc_path.exists():  # only designs with a defect layer get a QC section
        img = pd.read_parquet(qc_path)
        assert (img.groupby(["plate", "well"])["well_label"].nunique() == 1).all()  # well_label is constant per well
        qc = (img.groupby(["plate", "well"], sort=True)
              .agg(label=("well_label", "first"), degree=("degree", "max")).reset_index())

    return ReportInputs(
        objects=objects, wells=wells, features=DEFAULT_FEATURES, reference=reference,
        title="particle_plate truth report",
        source=f"{dataset.resolve().name} · {manifest.generator} {manifest.generator_version} · seed {manifest.seed}",
        site_area_px=float(design.plate.image.height * design.plate.image.width),
        n_sites=design.plate.sites_per_well, intended=intended, images=images,
        source_manifest_sha256=sha256_bytes((dataset / "manifest.json").read_bytes()),
        grid=design.plate.grid(), notes_for_reader=dict(NOTES_FOR_READER),
        qc=qc, qc_title="Quality control (true defect labels)" if qc is not None else None,
    )
