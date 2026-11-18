from synthdata.generators.particle_plate.truth_report import truth_inputs
from synthdata.report.contract import validate


def test_truth_inputs(small_truth_dataset):
    inp = truth_inputs(small_truth_dataset)
    assert len(inp.wells) == 32
    assert inp.reference == "control"
    assert len(inp.wells["condition"].unique()) == 8
    assert not inp.objects.empty and "object_id" in inp.objects.columns
    assert len(inp.images) == 8
    assert all(im.shape == (384, 384) for im in inp.images.values())
    assert inp.grid == (8, 4)
    assert inp.site_area_px == 384 * 384
    assert set(inp.intended) == {"diameter", "eccentricity", "orientation", "intensity"}
    assert len(inp.source_manifest_sha256) == 64
    v = validate(inp)
    from synthdata.generators.particle_plate.design import load_design
    design = load_design(small_truth_dataset / "design.yaml")
    assert list(v.wells["condition"].cat.categories) == list(design.conditions)
