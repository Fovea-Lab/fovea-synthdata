import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from synthdata.cli import app
from synthdata.generators.cell_plate.check import check_dataset, count_within_tolerance, null_verdicts, segment_site
from synthdata.generators.cell_plate.reader import read_site, read_truth_masks

runner = CliRunner()


def _wells(out):
    return pd.read_parquet(out / "truth" / "wells.parquet")


def _defect_wells(out):
    qc = pd.read_parquet(out / "truth" / "qc.parquet")
    return set(qc.loc[qc.defect != "none", "well"])


def test_blank_site_segments_to_nothing(small_dataset):
    blanks = _wells(small_dataset).query("role == 'blank'")
    assert len(blanks) == 8
    for r in blanks.itertuples():
        nuclei, cells = segment_site(read_site(small_dataset, r.plate, r.well, 1))
        assert nuclei.max() == 0 and cells.max() == 0


def test_counts_within_tolerance(small_dataset):
    wells = _wells(small_dataset)
    skip = _defect_wells(small_dataset)
    checked = 0
    for r in wells.itertuples():
        if r.role == "blank" or r.well in skip:
            continue
        truth_n = len(np.unique(read_truth_masks(small_dataset, r.plate, r.well, 1)[0])) - 1
        nuclei, _ = segment_site(read_site(small_dataset, r.plate, r.well, 1))
        found = len(np.unique(nuclei)) - 1
        assert abs(found - truth_n) <= max(0.25 * truth_n, 1), (r.well, found, truth_n)
        checked += 1
    assert checked > 60


def test_seeded_cells_share_nucleus_ids(small_dataset):
    r = _wells(small_dataset).query("role == 'negative_control'").iloc[0]
    nuclei, cells = segment_site(read_site(small_dataset, r.plate, r.well, 1))
    assert nuclei.max() > 0
    assert set(np.unique(cells)) == set(np.unique(nuclei))
    assert nuclei.shape == cells.shape


def test_segment_site_works_with_dna_only():
    rng = np.random.default_rng(0)
    img = rng.normal(200, 8, (128, 128))
    img[40:70, 40:70] += 3000
    nuclei, cells = segment_site({"DNA": img})
    assert nuclei.max() == 1 and set(np.unique(cells)) == {0, 1}


def test_check_passes_on_small(small_dataset):
    rep = check_dataset(small_dataset)
    failed = [c.name for c in rep.stage1 + rep.stage2 if not c.passed]
    assert not failed, failed
    assert rep.passed and rep.asserted
    assert (small_dataset / "check" / "report.json").is_file()
    assert (small_dataset / "check" / "report.md").is_file()


def test_check_marks_unsupported_features_not_checked(small_dataset):
    rep = check_dataset(small_dataset, write=False)
    by_name = {c.name: c for c in rep.stage2}
    for compound in ("CPD-005", "CPD-006", "CPD-009"):
        hit = [c for n, c in by_name.items() if compound in n]
        assert hit and all(c.status == "not_checked" and c.passed for c in hit)
    supported = [c for c in rep.stage2 if c.status != "not_checked"]
    assert any("CPD-002" in c.name for c in supported)


def test_check_flags_a_broken_dataset(small_dataset, tmp_path):
    import shutil
    broken = tmp_path / "broken"
    shutil.copytree(small_dataset, broken)
    (broken / "images" / "plate01_A02_s1_DNA.tif").unlink()
    with pytest.raises(Exception):
        check_dataset(broken, write=False)


def _copy(small_dataset, tmp_path):
    import shutil
    dst = tmp_path / "damaged"
    shutil.copytree(small_dataset, dst)
    return dst


def test_swapped_condition_labels_fail(small_dataset, tmp_path):
    dst = _copy(small_dataset, tmp_path)
    w = pd.read_parquet(dst / "truth" / "wells.parquet")
    a = w.index[w.condition == "CPD-002|10 uM"]
    b = w.index[w.condition == "DMSO"]
    for col in ("condition", "role", "phenotype", "strength"):
        ia, ib = w.columns.get_loc(col), w.columns.get_loc(col)
        for x, y in zip(a, b):
            va, vb = w.iat[x, ia], w.iat[y, ib]
            w.iat[x, ia], w.iat[y, ib] = vb, va
    w.to_parquet(dst / "truth" / "wells.parquet")
    assert not check_dataset(dst, write=False).passed


def test_swapped_truth_masks_fail(small_dataset, tmp_path):
    dst = _copy(small_dataset, tmp_path)
    lab = dst / "truth" / "labels"
    x, y = lab / "plate01_A01_s1_nuclei.tif", lab / "plate01_A02_s1_nuclei.tif"
    tmp = lab / "swap.tif"
    x.rename(tmp)
    y.rename(x)
    tmp.rename(y)
    rep = check_dataset(dst, write=False)
    assert not rep.passed
    assert any(not c.passed for c in rep.stage1)


@pytest.mark.parametrize("condition", ["CPD-004|10 uM", "DMSO"])
def test_cell_area_tracks_truth(small_dataset, condition):
    """High-coverage (CPD-004, ~50-60% of the frame) and ordinary (DMSO) wells: segmented median cell area vs truth.

    The segmenter sweep (3 seeds, small design) gave 0.91 to 1.13 for DMSO and 0.91 to 1.10 for CPD-004 at the
    high dose; the band leaves room around that. CPD-008 is left out: its bodies hold 2 to 3 nuclei and a seeded
    segmentation splits them, by design.
    """
    from skimage.measure import regionprops
    wells = _wells(small_dataset)
    skip = _defect_wells(small_dataset)
    rows = wells[(wells.condition == condition) & ~wells.well.isin(skip)]
    assert len(rows) >= 2
    for r in rows.itertuples():
        _, tc = read_truth_masks(small_dataset, r.plate, r.well, 1)
        _, cells = segment_site(read_site(small_dataset, r.plate, r.well, 1))
        ratio = np.median([p.area for p in regionprops(cells)]) / np.median([p.area for p in regionprops(tc)])
        assert 0.8 < ratio < 1.25, (r.well, ratio)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1234, 4321])
def test_check_passes_at_other_seeds(tmp_path, seed):
    import warnings
    from synthdata.generators.cell_plate.design import bundled_designs, load_design
    from synthdata.generators.cell_plate.generator import CellPlateGenerator
    design = load_design(bundled_designs()["small"].read_text()).model_copy(update={"seed": seed})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="cell_plate", category=UserWarning)
        CellPlateGenerator().generate(design, tmp_path / "d", writer="tiff")
    rep = check_dataset(tmp_path / "d", write=False)
    assert rep.passed, [c.name for c in rep.stage1 + rep.stage2 if not c.passed]


def test_only_saturated_debris_sites_are_excluded(small_dataset):
    rep = check_dataset(small_dataset, write=False)
    qc = pd.read_parquet(small_dataset / "truth" / "qc.parquet")
    n_debris = int((qc.defect == "saturated_debris").sum())
    counts = next(c for c in rep.stage2 if c.name.startswith("per-well nuclei count"))
    assert counts.detail["excluded_sites"] == n_debris
    assert any("n_a" in c.detail for c in rep.stage2)
    assert rep.exclusions["saturated_debris"] and "n_per_condition" in rep.exclusions


def test_cli_check_dispatches(small_dataset):
    res = runner.invoke(app, ["check", str(small_dataset)])
    assert res.exit_code == 0, res.output
    assert "PASS" in res.output


def test_cli_report_says_not_implemented(small_dataset):
    res = runner.invoke(app, ["report", str(small_dataset)])
    assert res.exit_code == 1
    assert "no report is implemented for generator 'cell_plate'" in res.output


@pytest.mark.parametrize("effects, expected", [
    ([(0.2, 0.5), (0.5, 0.2), (0.9, 0.04), (0.1, 0.9)], [True] * 4),                      # all under the bound
    ([(0.2, 0.5), (1.5, 0.01), (0.9, 0.2), (0.1, 0.9)], [True] * 4),                       # one marginal feature is chance
    ([(0.2, 0.5), (1.5, 0.01), (1.5, 0.02), (0.1, 0.9)], [True, False, False, True]),      # two marginal fail both
    ([(0.2, 0.5), (8.0, 0.2), (0.9, 0.2), (0.1, 0.9)], [True, False, True, True]),         # past the ceiling fails alone, p or not
    ([(0.2, 0.5), (2.99, 0.01), (0.9, 0.2), (0.1, 0.9)], [True] * 4),                      # just under the ceiling
    ([(1.5, 0.11), (1.5, 0.11), (1.5, 0.11), (0.1, 0.9)], [True] * 4),                     # d=1.5 at p=0.11 would not be called an effect
    ([(1.5, 0.11), (1.5, 0.01), (1.5, 0.02), (0.1, 0.9)], [True, False, False, True]),     # only the significant ones are marginal
])
def test_null_verdicts(effects, expected):
    assert null_verdicts(effects) == expected


@pytest.mark.parametrize("found, truth, ok", [
    (4, 6, True),      # two nuclei short of 6: a granularity miss, allowed
    (3, 6, False),     # off by 3 of 6 still fails
    (9, 6, False),
    (7, 6, True),
    (75, 100, True),   # large counts keep the 25% rule
    (74, 100, False),
    (0, 2, True),
    (0, 3, False),
])
def test_count_tolerance_is_relative_with_a_two_nucleus_floor(found, truth, ok):
    assert count_within_tolerance(found, truth) is ok
