import time

import pytest

from synthdata.generators.cell_plate import generator as gen
from synthdata.generators.cell_plate.check import check_dataset
from synthdata.generators.cell_plate.design import bundled_designs, load_design


@pytest.mark.slow
def test_default_plate_check_passes_and_is_in_budget(tmp_path):
    default = load_design(bundled_designs()["default"])
    t = time.monotonic()
    gen.CellPlateGenerator().generate(default, tmp_path, writer="tiff", workers=8)
    assert time.monotonic() - t < 300
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 3e9
    assert check_dataset(tmp_path).passed


def test_tuning_design_matches_default_except_seed_and_sites():
    default = load_design(bundled_designs()["default"]).model_dump()
    tuning = load_design(bundled_designs()["tuning"]).model_dump()
    assert tuning["seed"] == 4242 and tuning["plate"]["sites_per_well"] == 1
    for d in (default, tuning):
        d.pop("seed"); d["plate"].pop("sites_per_well")
    assert default == tuning
