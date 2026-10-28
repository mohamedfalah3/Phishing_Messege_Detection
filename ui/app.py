"""Phishing Message Detection - web UI.

Run from the project folder:   streamlit run ui/app.py
(or double-click ui/run_ui.bat on Windows)
"""
import streamlit as st

from components import inject_css

st.set_page_config(page_title="Phishing Message Detector", page_icon=":material/phishing:", layout="wide")
inject_css()

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/analyze.py", title="Analyze a message", icon=":material/search:"),
    st.Page("views/batch.py", title="Batch scan", icon=":material/upload_file:"),
    st.Page("views/performance.py", title="Model performance", icon=":material/monitoring:"),
    st.Page("views/errors.py", title="Error explorer", icon=":material/bug_report:"),
    st.Page("views/about.py", title="Dataset & method", icon=":material/school:"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### Settings")
    st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01, key="threshold",
              help="A message is flagged as phishing when P(phishing) is at or above this value.")
    st.caption("Lower threshold → catches more phishing (fewer false negatives) but flags more safe "
               "mail (more false positives). Used by every page.")
    st.divider()
    st.caption("AI course project · Deep learning text classification (PyTorch)")

nav.run()
