"""Report sections. Each module exposes `build(ctx) -> SectionResult`."""

from synthdata.report.sections import (clustering, distributions, fingerprints, heatmaps, objects, overview,
                                     qc, replicates)

CAPTIONS = {
    "overview": "The plate layout, one example image per condition and the counts behind the report.",
    "heatmaps": ("Each tile is one well; a pattern that follows the plate layout is a condition effect, and an edge "
                 "ring would be a plate artefact."),
    "distributions": ("Object-level distributions per condition. In the orientation roses 0° is horizontal and "
                      "angles run clockwise, as in the image; orientation is axial, so each bin is mirrored."),
    "fingerprints": ("Each tile is a condition's median robust z against the reference; the dot marks a "
                     "significant effect (q < 0.05 and |d| ≥ 1)."),
    "clustering": ("Wells cluster by their profiles: colour is the true condition, so a tight single-colour group "
                   "is a condition the profile recovers."),
    "replicates": ("Replicate wells of a condition should sit together; mean average precision near 1 means a well's "
                   "nearest profiles are its own replicates."),
    "objects": ("Individual objects coloured by condition; separated peaks, distinct clouds and shifted "
                "distributions show how conditions differ at the object level."),
}

CAPTIONS["qc"] = "Each well's quality call; amber and red wells need review. Grey wells were not assessed."

_INTENDED = ("the dark line is the intended distribution, so a violin that follows it is a condition rendered "
             "as designed.")
_TWIN = (" A control-role condition (a twin) should show no flags; with few wells per condition its |d| can "
         "approach 1 by chance, so a flag also needs q < 0.05.")
_ARI = (" Conditions that are identical by design (such as a control and its twin) cannot be told apart, so a "
        "perfect recovery cannot reach ARI 1; the ARI with the control conditions merged is shown beside it.")
_NOISE = (" Control-role z-profiles sit near zero, so the stripes in their similarity and their low mAP are noise "
          "by design and expected.")


def caption_for(inputs, key: str) -> str:
    """The design-agnostic caption for a section, plus the adapter's optional `notes_for_reader` sentence."""
    text = CAPTIONS[key]
    w = inputs.wells
    n_control = w.loc[w["role"].astype(str) == "control", "condition"].astype(str).nunique()
    has_twin = bool(((w["role"].astype(str) == "control") & (w["condition"].astype(str) != inputs.reference)).any())
    if key == "distributions" and inputs.intended:
        text = text.replace("per condition. ", f"per condition; {_INTENDED} ", 1)
    elif key == "fingerprints" and has_twin:
        text += _TWIN
    elif key == "clustering" and n_control >= 2:
        text += _ARI + _NOISE
    elif key == "replicates" and n_control >= 2:
        text += _NOISE
    extra = (getattr(inputs, "notes_for_reader", None) or {}).get(key)
    return f"{text} {extra}".strip() if extra else text


SECTIONS = [
    ("overview", "Overview", CAPTIONS["overview"], overview.build),
    ("heatmaps", "Plate heatmaps", CAPTIONS["heatmaps"], heatmaps.build),
    ("distributions", "Distributions", CAPTIONS["distributions"], distributions.build),
    ("fingerprints", "Fingerprints", CAPTIONS["fingerprints"], fingerprints.build),
    ("clustering", "Well profiles and clustering", CAPTIONS["clustering"], clustering.build),
    ("replicates", "Replicate consistency", CAPTIONS["replicates"], replicates.build),
    ("objects", "Object-level views", CAPTIONS["objects"], objects.build),
]

# Optional, always last: built only when the inputs carry a `qc` table (kept out of SECTIONS so a report without qc is unchanged).
QC_SECTION = ("qc", "Quality control", CAPTIONS["qc"], qc.build)
