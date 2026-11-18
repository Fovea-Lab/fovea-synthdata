import pytest

from synthdata.generators.particle_plate import ParticlePlateGenerator
from synthdata.generators.particle_plate.design import bundled_designs, load_design


@pytest.fixture(scope="session")
def small_truth_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("truth") / "small"
    ParticlePlateGenerator().generate(load_design(bundled_designs()["small"]), out, writer="tiff")
    return out


@pytest.fixture(scope="session")
def replicate_truth_dataset(tmp_path_factory):
    """Small design on a 96-well plate: 12 replicate wells per condition, one site per well."""
    import yaml

    spec = yaml.safe_load(bundled_designs()["small"].read_text())
    spec["plate"]["format"] = 96
    spec["plate"]["sites_per_well"] = 1
    out = tmp_path_factory.mktemp("truth96") / "replicates"
    ParticlePlateGenerator().generate(load_design(spec), out, writer="tiff")
    return out
