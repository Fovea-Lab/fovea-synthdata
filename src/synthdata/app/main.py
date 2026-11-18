"""synthdata app entry point — `synthdata app` runs `streamlit run` on this file."""

import streamlit as st

from synthdata.app import pages
from synthdata.app.state import ensure_state

st.set_page_config(page_title="synthdata", layout="wide")
ensure_state()
st.navigation([
    st.Page(pages.design_page, title="Design", url_path="design", default=True),
    st.Page(pages.preview_page, title="Preview", url_path="preview"),
    st.Page(pages.generate_page, title="Generate", url_path="generate"),
    st.Page(pages.inspect_page, title="Inspect", url_path="inspect"),
    st.Page(pages.report_page, title="Report", url_path="report"),
]).run()
