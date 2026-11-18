import warnings

import pytest

from synthdata.generators.cell_plate.design import bundled_designs, load_design
from synthdata.generators.cell_plate.generator import CellPlateGenerator


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    """The `small` design written once with the tiff writer; shared by every test that needs a dataset."""
    out = tmp_path_factory.mktemp("cell_small")
    design = load_design(bundled_designs()["small"].read_text())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="cell_plate", category=UserWarning)  # the small design is dense at a few sites
        CellPlateGenerator().generate(design, out, writer="tiff")
    return out
