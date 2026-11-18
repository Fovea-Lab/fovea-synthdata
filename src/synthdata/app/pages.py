"""The four app pages. Each calls the same generator/check functions as the CLI."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from synthdata.app.state import current_design, get_yaml, rev, set_yaml
from synthdata.app.widgets import arrangement_editor, dist_editor, profile_editor
from synthdata.cli import default_workers
from synthdata.core.provenance import read_manifest
from synthdata.generators.particle_plate.design import DIST_FIELDS, MEASURED_PARAMS, bundled_designs, dump_design
from synthdata.generators.particle_plate.layout import assign
from synthdata.generators.particle_plate.summaries import (
    condition_colours, histogram_frame, intended_pdf_frame, layout_grid, plate_map, render_tile,
    styled_plate,
)


# ---------- Design ----------

def design_colours(design) -> dict[str, str]:
    """Condition -> colour, from the design's condition order (same on Preview and Inspect)."""
    return condition_colours(list(design.conditions))


def _inspect_order(out: Path, pw: pd.DataFrame) -> list[str]:
    """Condition order of the dataset's own design.yaml; falls back to order of appearance."""
    try:
        from synthdata.generators.particle_plate.design import load_design
        return list(load_design(out / "design.yaml").conditions)
    except Exception:
        return list(dict.fromkeys(pw.condition))


def _design_form(d: dict) -> dict:
    d = copy.deepcopy(d)
    k = f"r{rev()}"
    with st.expander("Plate", expanded=True):
        c = st.columns(4)
        d["seed"] = int(c[0].number_input("seed", 0, 2**31 - 1, int(d["seed"]), key=f"{k}_seed"))
        fmt = d["plate"]["format"]
        choice = c[1].selectbox("format", ["96", "384", "custom"],
                                index=["96", "384", "custom"].index(str(fmt) if fmt in (96, 384) else "custom"), key=f"{k}_fmt")
        if choice == "custom":
            cur = fmt if isinstance(fmt, dict) else {"rows": 8, "cols": 12}
            d["plate"]["format"] = {"rows": int(c[2].number_input("rows", 1, 26, int(cur["rows"]), key=f"{k}_rows")),
                                    "cols": int(c[3].number_input("cols", 1, 99, int(cur["cols"]), key=f"{k}_cols"))}
        else:
            d["plate"]["format"] = int(choice)
        c = st.columns(4)
        d["plate"]["plates"] = int(c[0].number_input("plates", 1, 50, int(d["plate"]["plates"]), key=f"{k}_plates"))
        d["plate"]["sites_per_well"] = int(c[1].number_input("sites per well", 1, 100, int(d["plate"]["sites_per_well"]), key=f"{k}_sites"))
        d["plate"]["image"]["height"] = int(c[2].number_input("height", 16, 8192, int(d["plate"]["image"]["height"]), key=f"{k}_h"))
        d["plate"]["image"]["width"] = int(c[3].number_input("width", 16, 8192, int(d["plate"]["image"]["width"]), key=f"{k}_w"))
    with st.expander("Image effects"):
        fx = d["image_effects"]
        fx["background"] = dist_editor("background", fx["background"], f"{k}_bg")
        c = st.columns(4)
        fx["noise"]["poisson"] = c[0].checkbox("Poisson noise", fx["noise"]["poisson"], key=f"{k}_pn")
        fx["noise"]["gaussian_sd"] = float(c[1].number_input("read noise sd", 0.0, 1e4, float(fx["noise"]["gaussian_sd"]), key=f"{k}_gn"))
        kinds = ["none", "radial", "linear"]
        fx["illumination"]["kind"] = c[2].selectbox("illumination", kinds, index=kinds.index(fx["illumination"]["kind"]), key=f"{k}_il")
        fx["illumination"]["strength"] = float(c[3].number_input("strength", 0.0, 0.99, float(fx["illumination"]["strength"]), key=f"{k}_is"))
        c = st.columns(2)
        fx["edge_effect"]["enabled"] = c[0].checkbox("edge effect (outer wells dimmer)", fx["edge_effect"]["enabled"], key=f"{k}_ee")
        fx["edge_effect"]["dim"] = float(c[1].number_input("edge dim", 0.0, 0.99, float(fx["edge_effect"]["dim"]), key=f"{k}_ed"))
    base = d["base_particle"]
    with st.expander("Base particle (every condition inherits this)"):
        for f in DIST_FIELDS:
            base[f] = dist_editor(f, base[f], f"{k}_base_{f}")
        base["arrangement"] = arrangement_editor(base["arrangement"], f"{k}_base_arr")
        base["profile"] = profile_editor(base["profile"], f"{k}_base_prof")
    st.subheader("Conditions")
    for name in list(d["conditions"]):
        cond = d["conditions"][name]
        with st.expander(name):
            cond["role"] = st.selectbox("role", ["control", "test"], index=["control", "test"].index(cond.get("role", "test")), key=f"{k}_{name}_role")
            for f in (*DIST_FIELDS, "arrangement", "profile"):
                if st.checkbox(f"override {f}", f in cond, key=f"{k}_{name}_ov_{f}"):
                    cur = cond.get(f, base[f])
                    if f == "arrangement":
                        cond[f] = arrangement_editor(cur, f"{k}_{name}_{f}")
                    elif f == "profile":
                        cond[f] = profile_editor(cur, f"{k}_{name}_{f}")
                    else:
                        cond[f] = dist_editor(f, cur, f"{k}_{name}_{f}")
                else:
                    cond.pop(f, None)
            c = st.columns(2)
            if c[0].button("Duplicate", key=f"{k}_{name}_dup"):
                new, i = f"{name}_copy", 2
                while new in d["conditions"]:
                    new, i = f"{name}_copy{i}", i + 1
                d["conditions"][new] = copy.deepcopy(cond)
                if d["layout"].get("order"):
                    d["layout"]["order"].append(new)
                set_yaml(yaml.safe_dump(d, sort_keys=False))
                st.rerun()
            if c[1].button("Remove", key=f"{k}_{name}_rm"):
                d["conditions"].pop(name)
                if d["layout"].get("order"):
                    d["layout"]["order"] = [o for o in d["layout"]["order"] if o != name]
                set_yaml(yaml.safe_dump(d, sort_keys=False))
                st.rerun()
    c = st.columns([3, 1])
    new_name = c[0].text_input("New condition name", key=f"{k}_newname")
    if c[1].button("Add condition", key=f"{k}_add") and new_name and new_name not in d["conditions"]:
        d["conditions"][new_name] = {"role": "test"}
        if d["layout"].get("order"):
            d["layout"]["order"].append(new_name)
        set_yaml(yaml.safe_dump(d, sort_keys=False))
        st.rerun()
    with st.expander("Layout"):
        kinds = ["by_row", "by_column", "explicit", "randomized"]
        d["layout"]["kind"] = st.selectbox("kind", kinds, index=kinds.index(d["layout"]["kind"]), key=f"{k}_lk")
        order = d["layout"].get("order") or list(d["conditions"])
        d["layout"]["order"] = st.multiselect("order", list(d["conditions"]), default=[o for o in order if o in d["conditions"]], key=f"{k}_lo")
    return d


def design_page() -> None:
    st.title("Design")
    bundled = bundled_designs()
    c = st.columns([2, 1, 3])
    pick = c[0].selectbox("Start from a bundled design", list(bundled), key="design_pick")
    if c[1].button("Load", key="design_load"):
        set_yaml(bundled[pick].read_text())
        st.rerun()
    path = c[2].text_input("…or a design file path", key="design_path")
    if path and st.button("Load file", key="design_load_file"):
        try:
            set_yaml(Path(path).expanduser().read_text())
            st.rerun()
        except OSError as exc:
            st.error(str(exc))
    gen, design, err = current_design()
    tab_form, tab_yaml = st.tabs(["Form", "YAML"])
    with tab_form:
        if err:
            st.info("Fix the YAML first; the form needs a valid design.")
        else:
            base = design.model_dump(mode="json", exclude_none=True)
            edited = _design_form(base)
            if edited != base:
                set_yaml(yaml.safe_dump(edited, sort_keys=False), bump=False)
                gen, design, err = current_design()
    with tab_yaml:
        text = st.text_area("design.yaml", get_yaml(), height=520, key=f"yaml_{hashlib.sha1(get_yaml().encode()).hexdigest()[:12]}")
        if st.button("Apply YAML", key="yaml_apply"):
            set_yaml(text)
            st.rerun()
    if err:
        st.error(err)
    else:
        n_wells = len(assign(design))
        st.success(f"valid · {n_wells} wells · {n_wells * design.plate.sites_per_well} images")
    c = st.columns([3, 1])
    save = c[0].text_input("Save as", "designs/my_design.yaml", key="design_save_path")
    if c[1].button("Save", key="design_save"):
        p = Path(save).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(get_yaml())
        st.toast(f"saved {p}")


# ---------- Preview ----------

@st.cache_data(show_spinner="rendering preview…", max_entries=8)
def _sheet(yaml_text: str, n: int):
    from synthdata.core import registry

    data = yaml.safe_load(yaml_text)
    gen = registry.get(data["generator"])
    return gen.contact_sheet(gen.preview(gen.load_design(data), n))


def preview_page() -> None:
    st.title("Preview")
    gen, design, err = current_design()
    if err:
        st.error(err)
        return
    n = st.slider("Sites per condition", 1, 4, 1, key="preview_n")
    st.image(_sheet(get_yaml(), n), caption="one row per condition · contrast shared across tiles")
    st.subheader("Intended distributions")
    param = st.selectbox("Parameter", list(DIST_FIELDS), index=1, key="preview_param")
    st.line_chart(intended_pdf_frame(design, param))
    st.caption("Fixed values have no density and draw as a flat line.")
    st.subheader("Plate layout (plate 1)")
    st.dataframe(styled_plate(layout_grid(design), design_colours(design)), width="stretch")


# ---------- Generate ----------

def generate_page() -> None:
    st.title("Generate")
    gen, design, err = current_design()
    if err:
        st.error(err)
        return
    out = st.text_input("Output folder", str(Path.home() / "synthdata-out" / "particle_plate"), key="gen_out")
    writers = list(gen.writers)
    writer = st.selectbox("Writer", writers, index=writers.index(gen.default_writer), key="gen_writer")
    c = st.columns(2)
    seed = int(c[0].number_input("Seed", 0, 2**31 - 1, int(design.seed), key=f"gen_seed_{design.seed}"))
    workers = int(c[1].number_input("Workers", 1, 64, default_workers(), key="gen_workers"))
    overwrite = st.checkbox("Overwrite if the folder holds a previous synthdata dataset", key="gen_overwrite")
    if st.button("Generate", type="primary", key="gen_go"):
        bar = st.progress(0.0, text="rendering…")

        def progress(done: int, total: int) -> None:
            bar.progress(done / total, text=f"{done}/{total} images")

        try:
            m = gen.generate(type(design).model_validate({**design.model_dump(mode="json", exclude_none=True), "seed": seed}), Path(out).expanduser(), writer=writer,
                             workers=workers, progress=progress, overwrite=overwrite)
        except Exception as exc:  # shown to the user; the CLI prints the same messages
            st.error(str(exc))
            return
        st.session_state["dataset_path"] = str(Path(out).expanduser())
        st.success(f"wrote {m.extra['n_images']} images, {m.extra['n_particles']} particles → {out}")
        st.json({k: v for k, v in asdict(m).items() if k != "files"})


# ---------- Inspect ----------

def inspect_page() -> None:
    st.title("Inspect")
    path = st.text_input("Dataset folder", st.session_state.get("dataset_path", ""), key="inspect_path")
    if not path:
        st.info("Generate a dataset first, or paste a dataset folder.")
        return
    out = Path(path).expanduser()
    try:
        m = read_manifest(out)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    from synthdata.generators.particle_plate.check import check_dataset
    from synthdata.generators.particle_plate.reader import read_image

    for name in ("wells", "images"):
        f = out / "truth" / f"{name}.parquet"
        if not f.exists():
            st.error(f"{f} is missing — not a complete particle_plate dataset")
            return
    wells = pd.read_parquet(out / "truth" / "wells.parquet")
    images = pd.read_parquet(out / "truth" / "images.parquet")
    st.caption(f"{m.generator} {m.generator_version} · writer {m.writer} · seed {m.seed} · {m.extra}")
    plates = sorted(wells.plate.unique())
    plate = st.selectbox("Plate", plates, key="inspect_plate") if len(plates) > 1 else plates[0]
    pw = wells[wells.plate == plate]
    st.dataframe(styled_plate(plate_map(pw), condition_colours(_inspect_order(m, pw))), width="stretch")
    well = st.selectbox("Well", pw.well.tolist(), key="inspect_well")
    overlay = st.toggle("Mask outlines", key="inspect_overlay")
    sites = images[(images.plate == plate) & (images.well == well)].site.tolist()
    cols = st.columns(max(len(sites), 1))
    for col, site in zip(cols, sites):
        img, mask = read_image(out, plate, well, int(site))
        col.image(render_tile(img, mask if overlay else None), caption=f"{well} s{site}", width="stretch")
    st.subheader("Truth distributions by condition")
    param = st.selectbox("Parameter", list(MEASURED_PARAMS), key="inspect_param")
    particles = pd.read_parquet(out / "truth" / "particles.parquet", columns=["condition", param])
    st.bar_chart(histogram_frame(particles, param))
    st.subheader("Recoverability")
    if st.button("Run check", key="inspect_check"):
        with st.spinner("checking…"):
            check_dataset(out)
    report = out / "check" / "report.md"
    if report.exists():
        st.markdown(report.read_text())


def _inside(base: Path, rel: str | None) -> Path | None:
    """`base / rel` if it stays inside `base` (report.json paths are relative), else None."""
    if not rel:
        return None
    p = (base / rel).resolve()
    return p if p.is_relative_to(base.resolve()) and p.is_file() else None


def _build_report_ui(ds: Path, label: str, key: str) -> None:
    from synthdata.generators.particle_plate.truth_report import truth_inputs
    from synthdata.report import build_report

    if st.button(label, key=key):
        try:
            with st.spinner("Building report..."):
                build_report(truth_inputs(ds), ds / "report", overwrite=True)
        except Exception as e:  # surface any build failure in the page
            st.error(f"Report build failed: {e}")
            return
        st.rerun()


def report_page() -> None:
    import json

    st.header("Report")
    path = st.text_input("Dataset path", value=st.session_state.get("dataset_path", ""), key="report_dataset")
    if not path.strip():
        st.info("Enter the path of a generated dataset to view or build its truth report.")
        return
    ds = Path(path).expanduser()
    base = ds / "report"
    rj = base / "report.json"
    if not rj.exists():
        st.info(f"No report at {base} yet.")
        _build_report_ui(ds, "Build report", "report_build")
        return
    try:
        rep = json.loads(rj.read_text())
        sections = {s["key"]: s for s in rep["sections"]}
    except (OSError, ValueError, KeyError, TypeError) as e:
        st.error(f"{rj} is unreadable ({type(e).__name__}: {e}). Rebuild the report below, or run "
                 f"`synthdata report {ds} --overwrite`.")
        if st.checkbox("Replace the existing report", key="report_rebuild_confirm"):
            _build_report_ui(ds, "Rebuild report", "report_rebuild")
        return
    m = rep.get("metrics", {})
    labels = [("ari", "ARI"), ("ari_control_merged", "ARI (controls merged)"), ("nmi", "NMI"), ("k", "k"), ("mean_map", "Mean mAP"),
              ("n_wells", "Wells"), ("n_objects", "Objects")]
    shown = [(lab, m[k]) for k, lab in labels if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)]
    if "twin_flagged" in m:
        from synthdata.report.summary import twin_status

        kind = twin_status(m)[0]
        shown.append(("Twin", {"none": "not applicable", "untestable": "not testable (no condition reaches "
                               "significance)", "flagged": "flagged", "ok": "indistinguishable"}[kind]))
    if shown:
        for col, (lab, v) in zip(st.columns(len(shown)), shown):
            col.metric(lab, f"{v:.3g}" if isinstance(v, float) else v)
    for n in rep.get("notes", []):
        st.caption(n)
    key = st.selectbox("Section", list(sections), key="report_section", format_func=lambda k: sections[k]["title"])
    sec = sections[key]
    st.caption(sec.get("caption", ""))
    for n in sec.get("notes", []):
        st.caption(n)
    for fig in sec.get("figures", []):
        spec, png = _inside(base, fig.get("spec")), _inside(base, fig.get("png"))
        if spec:
            st.vega_lite_chart(json.loads(spec.read_text()), width="stretch")
        elif png:
            st.image(str(png), caption=fig.get("title"))
        else:
            st.caption(f"{fig.get('name', 'figure')}: file missing or outside the report folder")
    idx = base / "index.html"
    if idx.exists():
        st.download_button("Download index.html", idx.read_bytes(), file_name="index.html", mime="text/html")
    with st.expander("Tables"):
        tables = [t for s in rep["sections"] for t in s.get("tables", []) if _inside(base, t.get("parquet"))]
        if tables:
            t = st.selectbox("Table", tables, format_func=lambda t: t["name"], key="report_table")
            st.dataframe(pd.read_parquet(_inside(base, t["parquet"])), width="stretch")
    with st.expander("Rebuild"):
        st.caption("Recompute the report from the dataset; the current report is replaced.")
        if st.checkbox("Replace the existing report", key="report_rebuild_confirm"):
            _build_report_ui(ds, "Rebuild report", "report_rebuild")
