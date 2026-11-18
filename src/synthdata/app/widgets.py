"""Form widgets for distribution specs and the other small design blocks."""

from __future__ import annotations

import streamlit as st

TYPES = ["normal", "lognormal", "uniform", "poisson", "beta_mean", "fixed", "mixture"]
PARAMS = {"normal": ["mean", "sd"], "lognormal": ["mean", "sd"], "uniform": ["low", "high"],
          "poisson": ["mean"], "beta_mean": ["mean", "sd"], "fixed": ["value"]}
DEFAULTS = {"normal": {"mean": 1.0, "sd": 0.1}, "lognormal": {"mean": 1.0, "sd": 0.1},
            "uniform": {"low": 0.0, "high": 1.0}, "poisson": {"mean": 10.0},
            "beta_mean": {"mean": 0.5, "sd": 0.1}, "fixed": {"value": 1.0}}


def dist_editor(label: str, spec: dict, key: str, allow_mixture: bool = True) -> dict:
    types = TYPES if allow_mixture else TYPES[:-1]
    current = spec.get("dist", "normal")
    cols = st.columns([1.3, 1, 1])
    kind = cols[0].selectbox(label, types, index=types.index(current) if current in types else 0, key=f"{key}_dist")
    if kind == "mixture":
        comps = spec.get("components") if current == "mixture" else [
            {"weight": 0.5, **{"dist": "normal", "mean": 1.0, "sd": 0.1}},
            {"weight": 0.5, **{"dist": "normal", "mean": 2.0, "sd": 0.1}}]
        n = int(st.number_input(f"{label}: components", 2, 5, len(comps), key=f"{key}_n"))
        out = []
        for i in range(n):
            c = comps[i] if i < len(comps) else {"weight": 0.1, "dist": "normal", "mean": 1.0, "sd": 0.1}
            w = st.number_input(f"{label} · component {i + 1} weight", 0.0, 1.0, float(c["weight"]), key=f"{key}_w{i}")
            sub = dist_editor(f"{label} · component {i + 1}", {k: v for k, v in c.items() if k != "weight"},
                              f"{key}_c{i}", allow_mixture=False)
            out.append({"weight": w, **sub})
        return {"dist": "mixture", "components": out}
    vals = {}
    for j, p in enumerate(PARAMS[kind]):
        default = spec.get(p, DEFAULTS[kind][p]) if current == kind else DEFAULTS[kind][p]
        vals[p] = cols[1 + j].number_input(p, value=float(default), key=f"{key}_{p}", format="%g")
    return {"dist": kind, **vals}


def arrangement_editor(spec: dict, key: str) -> dict:
    c = st.columns(3)
    kind = c[0].selectbox("arrangement", ["uniform", "clumped"], index=["uniform", "clumped"].index(spec.get("kind", "uniform")), key=f"{key}_kind")
    if kind == "uniform":
        return {"kind": "uniform"}
    return {"kind": "clumped",
            "clusters": int(c[1].number_input("clusters", 1, 100, int(spec.get("clusters", 5)), key=f"{key}_cl")),
            "spread": float(c[2].number_input("spread (px)", 1.0, 1000.0, float(spec.get("spread", 50.0)), key=f"{key}_sp"))}


def profile_editor(spec: dict, key: str) -> dict:
    c = st.columns(2)
    kinds = ["flat", "gaussian", "ring"]
    kind = c[0].selectbox("profile", kinds, index=kinds.index(spec.get("kind", "flat")), key=f"{key}_kind")
    if kind == "gaussian":
        return {"kind": kind, "sigma_frac": float(c[1].number_input("sigma_frac", 0.01, 5.0, float(spec.get("sigma_frac", 0.35)), key=f"{key}_sf"))}
    if kind == "ring":
        return {"kind": kind, "width_frac": float(c[1].number_input("width_frac", 0.01, 1.0, float(spec.get("width_frac", 0.3)), key=f"{key}_wf"))}
    return {"kind": "flat"}
