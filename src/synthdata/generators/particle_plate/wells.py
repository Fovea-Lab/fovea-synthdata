"""Plate, well and site names shared by every particle_plate module."""

import re

ROW_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def well_name(row: int, col: int) -> str:
    return f"{ROW_LETTERS[row]}{col + 1:02d}"


def parse_well(name: str) -> tuple[int, int]:
    m = re.fullmatch(r"([A-Z])(\d{1,2})", name.strip().upper())
    if not m:
        raise ValueError(f"bad well name {name!r}; expected e.g. A01")
    return ROW_LETTERS.index(m[1]), int(m[2]) - 1


def plate_name(plate: int) -> str:
    return f"plate{plate + 1:02d}"
