"""Deterministic random streams keyed by *what* is generated, not by the order of calls."""

import numpy as np


def stream(seed: int, *key: int) -> np.random.Generator:
    """An independent generator for ``key`` (e.g. namespace, plate, row, col, site) under ``seed``."""
    if any(int(k) < 0 for k in key):
        raise ValueError(f"stream keys must be non-negative integers, got {key}")
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(int(k) for k in key)))
