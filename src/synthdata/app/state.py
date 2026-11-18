"""Session state shared by the app pages: the current design is one YAML string."""

from __future__ import annotations

import streamlit as st
import yaml
from pydantic import ValidationError

from synthdata.core import registry
from synthdata.generators.particle_plate.design import bundled_designs

KEY, REV = "design_yaml", "design_rev"


def ensure_state() -> None:
    if KEY not in st.session_state:
        st.session_state[KEY] = bundled_designs()["default"].read_text()
    st.session_state.setdefault(REV, 0)


def get_yaml() -> str:
    ensure_state()
    return st.session_state[KEY]


def rev() -> int:
    ensure_state()
    return st.session_state[REV]


def set_yaml(text: str, bump: bool = True) -> None:
    """Replace the design. ``bump`` re-keys the form widgets so they pick up the new values."""
    st.session_state[KEY] = text
    if bump:
        st.session_state[REV] = rev() + 1


def current_design():
    """(generator, design, None) on success, (None, None, message) on failure."""
    try:
        data = yaml.safe_load(get_yaml())
        if not isinstance(data, dict) or "generator" not in data:
            return None, None, "the design needs a top-level 'generator:' key"
        gen = registry.get(data["generator"])
        return gen, gen.load_design(data), None
    except (ValidationError, yaml.YAMLError, KeyError, ValueError) as exc:
        return None, None, str(exc)
