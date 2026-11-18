import copy

import pytest
import yaml

from synthdata.generators.particle_plate.design import bundled_designs, load_design


@pytest.fixture
def small_dict():
    return copy.deepcopy(yaml.safe_load(bundled_designs()["small"].read_text()))


@pytest.fixture
def small_design(small_dict):
    return load_design(small_dict)
