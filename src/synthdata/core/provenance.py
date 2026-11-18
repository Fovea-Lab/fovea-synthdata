"""Dataset manifest: what produced a dataset, plus a checksum of every file in it."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from synthdata import __version__

MANIFEST_NAME = "manifest.json"


@dataclass
class Manifest:
    generator: str
    generator_version: str
    synthdata_version: str
    seed: int
    design_sha256: str
    writer: str
    created_at: str
    files: dict[str, str] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / MANIFEST_NAME and not p.name.endswith(".tmp")
    }


def write_manifest(out: Path, *, generator: str, generator_version: str, seed: int, design_yaml: str,
                   writer: str, extra: dict | None = None) -> Manifest:
    """Hash everything in ``out`` and write the manifest. Call this LAST: it marks the dataset complete."""
    out = Path(out)
    m = Manifest(
        generator=generator,
        generator_version=generator_version,
        synthdata_version=__version__,
        seed=int(seed),
        design_sha256=sha256_bytes(design_yaml.encode()),
        writer=writer,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        files=hash_tree(out),
        extra=extra or {},
    )
    tmp_path = out / (MANIFEST_NAME + ".tmp")
    tmp_path.write_text(json.dumps(asdict(m), indent=2) + "\n")
    os.replace(tmp_path, out / MANIFEST_NAME)
    return m


def read_manifest(out: Path) -> Manifest:
    p = Path(out) / MANIFEST_NAME
    if not p.is_file():
        raise FileNotFoundError(f"{out} has no {MANIFEST_NAME}; it is not a complete synthdata dataset")
    return Manifest(**json.loads(p.read_text()))


def is_complete(out: Path) -> bool:
    return (Path(out) / MANIFEST_NAME).is_file()
