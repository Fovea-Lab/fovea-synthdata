import json

import numpy as np
import pytest

from synthdata.core import registry
from synthdata.core.generator import SENTINEL, GenerationError, prepare_output
from synthdata.core.provenance import MANIFEST_NAME, hash_tree, is_complete, read_manifest, write_manifest
from synthdata.core.rng import stream


def test_stream_is_keyed_not_ordered():
    a = stream(42, 1, 0, 3, 2).random(5)
    stream(42, 9, 9).random(100)  # unrelated draws in between change nothing
    b = stream(42, 1, 0, 3, 2).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, stream(42, 1, 0, 3, 3).random(5))
    assert not np.array_equal(a, stream(43, 1, 0, 3, 2).random(5))


def test_stream_rejects_negative_keys():
    with pytest.raises(ValueError):
        stream(1, -1)


def test_manifest_round_trip(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\x00\x01")
    assert not is_complete(tmp_path)
    m = write_manifest(tmp_path, generator="g", generator_version="1", seed=3,
                       design_yaml="a: 1\n", writer="w", extra={"n": 2})
    assert is_complete(tmp_path)
    assert set(m.files) == {"a.txt", "sub/b.bin"}
    back = read_manifest(tmp_path)
    assert back == m
    assert json.loads((tmp_path / MANIFEST_NAME).read_text())["seed"] == 3
    assert MANIFEST_NAME not in hash_tree(tmp_path)


def test_read_manifest_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a complete synthdata dataset"):
        read_manifest(tmp_path)


def test_prepare_output_creates_and_refuses(tmp_path):
    out = tmp_path / "new" / "nested"
    prepare_output(out, overwrite=False)
    assert out.is_dir()
    assert (out / SENTINEL).is_file()
    (out / "design.yaml").write_text("x")
    with pytest.raises(GenerationError, match="not empty"):
        prepare_output(out, overwrite=False)
    prepare_output(out, overwrite=True)  # looks like a synthdata dataset → replaced
    assert out.is_dir() and list(out.iterdir()) == [out / SENTINEL]


def test_prepare_output_refuses_foreign_folder(tmp_path):
    (tmp_path / "precious.txt").write_text("keep me")
    with pytest.raises(GenerationError, match="does not look like a synthdata dataset"):
        prepare_output(tmp_path, overwrite=True)
    assert (tmp_path / "precious.txt").read_text() == "keep me"


def test_prepare_output_refuses_a_file(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(GenerationError, match="not a folder"):
        prepare_output(f, overwrite=True)


def test_prepare_output_refuses_generic_markers(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "design.yaml").write_text("x")
    with pytest.raises(GenerationError, match="does not look like a synthdata dataset"):
        prepare_output(tmp_path, overwrite=True)
    assert (tmp_path / "manifest.json").read_text() == "{}"
    assert (tmp_path / "design.yaml").read_text() == "x"


def test_manifest_atomic_write(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    write_manifest(tmp_path, generator="g", generator_version="1", seed=3,
                   design_yaml="a: 1\n", writer="w", extra={"n": 2})
    assert not (tmp_path / (MANIFEST_NAME + ".tmp")).exists()


def test_registry_lists_builtin_and_rejects_unknown():
    assert "particle_plate" in registry.names()
    with pytest.raises(KeyError, match="unknown generator 'nope'"):
        registry.get("nope")


def test_registry_register_custom(monkeypatch):
    class Dummy:
        name = "dummy_test_gen"

    monkeypatch.setattr(registry, "_custom", {})
    registry.register(Dummy())
    assert "dummy_test_gen" in registry.names()
    assert registry.get("dummy_test_gen").name == "dummy_test_gen"


def test_load_design_file_needs_generator_key(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("seed: 1\n")
    with pytest.raises(ValueError, match="generator"):
        registry.load_design_file(p)
