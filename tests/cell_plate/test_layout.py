import warnings
from collections import Counter, defaultdict

import pytest
import yaml

from synthdata.generators.cell_plate.design import bundled_designs, load_design
from pydantic import ValidationError

from synthdata.generators.cell_plate.layout import assign


def _default(**seed):
    data = yaml.safe_load(bundled_designs()["default"].read_text())
    data.update(seed)
    return load_design(data)


def test_counts_and_fixed_controls():
    slots = assign(_default())
    assert len(slots) == 96
    assert [(s.row, s.col) for s in slots] == sorted((s.row, s.col) for s in slots)
    assert Counter(s.role for s in slots) == {"negative_control": 8, "positive_control": 8, "test": 72, "blank": 8}
    assert {s.well for s in slots if s.compound == "DMSO"} == {"A01", "B06", "C11", "D03", "E08", "F12", "G02", "H07"}
    assert {s.well for s in slots if s.compound == "POS-CTRL"} == {"A07", "B12", "C04", "D09", "E01", "F05", "G10", "H03"}
    pos = next(s for s in slots if s.compound == "POS-CTRL")
    assert (pos.dose_label, pos.strength, pos.phenotype, pos.condition) == ("10 uM", 1.0, "apoptotic", "POS-CTRL")
    dmso = next(s for s in slots if s.compound == "DMSO")
    assert (dmso.dose_label, dmso.strength, dmso.phenotype) == ("", 0.0, None)
    blank = next(s for s in slots if s.role == "blank")
    assert (blank.compound, blank.condition, blank.phenotype, blank.strength) == ("BLANK", "BLANK", None, 0.0)
    assert slots[0].plate_name == "plate01"


def test_replicates_never_share_row_or_column():
    groups = defaultdict(list)
    for s in assign(_default()):
        if s.role == "test":
            groups[s.condition].append(s)
    assert len(groups) == 18 and "CPD-001|1 uM" in groups
    for g in groups.values():
        assert len(g) == 4
        assert len({s.row for s in g}) == 4 and len({s.col for s in g}) == 4
        assert [s.replicate for s in g] == [1, 2, 3, 4]


def test_doses_of_a_compound_not_adjacent():
    cells = defaultdict(dict)
    for s in assign(_default()):
        if s.role == "test":
            cells[s.compound][(s.row, s.col)] = s.dose_label
    for grid in cells.values():
        for (r, c), label in grid.items():
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                assert grid.get(nb, label) == label


def _blank_only(rows, cols, blanks, seed):
    data = yaml.safe_load(bundled_designs()["default"].read_text())
    data["seed"] = seed
    data["plate"]["format"] = {"rows": rows, "cols": cols}
    data["compounds"] = {"DMSO": data["compounds"]["DMSO"]}
    data["doses"] = {}
    data["layout"] = {"fixed": {}, "blanks": blanks, "replicates": 1}
    data.pop("expectations")
    return load_design(data)


def _one_edge(slots, rows, cols):
    return (all(s.row == 0 for s in slots) or all(s.row == rows - 1 for s in slots)
            or all(s.col == 0 for s in slots) or all(s.col == cols - 1 for s in slots))


def test_blanks_not_all_on_one_edge():
    # Two blanks on a 4x4 grid share an edge in about one draw in five when unconstrained.
    for seed in range(60):
        blanks = [s for s in assign(_blank_only(4, 4, 2, seed)) if s.role == "blank"]
        assert len(blanks) == 2
        assert not _one_edge(blanks, 4, 4), f"seed {seed}: {[b.well for b in blanks]}"
    default = [s for s in assign(_default()) if s.role == "blank"]
    assert len(default) == 8 and not _one_edge(default, 8, 12)


def test_blanks_edge_constraint_relaxes_with_warning():
    # A single blank on a 2x2 grid is always on an edge: no compliant draw exists.
    with pytest.warns(UserWarning, match="blanks"):
        slots = assign(_blank_only(2, 2, 1, 0))
    assert [s.role for s in slots] == ["blank"]


def test_deterministic_and_seed_dependent():
    a, b, c = assign(_default()), assign(_default()), assign(_default(seed=7))
    assert a == b
    assert a != c


def _small_grid(replicates, doses=2, compounds=1):
    data = yaml.safe_load(bundled_designs()["default"].read_text())
    data["plate"]["format"] = {"rows": 4, "cols": 4}
    data["compounds"] = {"DMSO": data["compounds"]["DMSO"],
                         **{f"CPD-{i}": {"phenotype": "no_effect"} for i in range(compounds)}}
    data["doses"] = {f"d{i}": {"label": f"{i} uM", "strength": 0.5} for i in range(doses)}
    data["layout"] = {"fixed": {}, "blanks": 0, "replicates": replicates,
                      "constraints": {"no_shared_row_or_col": True, "doses_not_adjacent": True}}
    data.pop("expectations")
    return load_design(data)


def test_unsatisfiable_layout_relaxes_then_raises():
    # Three doses x four replicates on a 4x4 grid are three disjoint permutation matrices;
    # there is no room to keep a compound's doses apart, so doses_not_adjacent must be relaxed.
    design = _small_grid(replicates=4, doses=3, compounds=1)
    with pytest.warns(UserWarning, match="doses_not_adjacent"):
        slots = assign(design)
    assert len(slots) == 12 and {s.role for s in slots} == {"test"}
    # Five replicates cannot avoid a shared row or column on four rows: relaxation does not help.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no_shared_row_or_col"):
            assign(_small_grid(replicates=5, doses=1))


def design(plates=1):
    """The small design (96 wells, one site), with its plate count overridden."""
    data = yaml.safe_load(bundled_designs()["small"].read_text())
    data["plate"]["plates"] = plates
    return load_design(data)


def two_plate_design():
    return design(plates=2)


def test_two_plates_have_their_own_layout():
    slots = assign(two_plate_design())
    a, b = [{(s.row, s.col): s.condition for s in slots if s.plate == p} for p in (0, 1)]
    assert len(slots) == 192 and a != b
    assert sorted(a.values()) == sorted(b.values())
    assert [(s.plate, s.row, s.col) for s in slots] == sorted((s.plate, s.row, s.col) for s in slots)


def test_plate_zero_matches_the_one_plate_layout():
    one, two = assign(design(plates=1)), assign(design(plates=2))
    assert [s for s in two if s.plate == 0] == one


def test_replicate_numbers_restart_per_plate():
    reps = {p: sorted((s.condition, s.replicate) for s in assign(two_plate_design()) if s.plate == p)
            for p in (0, 1)}
    assert reps[0] == reps[1]


def test_plates_over_two_is_allowed_and_zero_is_refused():
    assert design(plates=3).plate.plates == 3
    with pytest.raises(ValidationError):
        design(plates=0)
