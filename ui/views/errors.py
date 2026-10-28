import pandas as pd
import streamlit as st

from backend import LONG_NAMES, SHORT_NAMES, attention, model_names, test_predictions
from components import attention_html, model_scores

st.title("Error explorer")
st.caption("Every test message the chosen model gets wrong at the current threshold. "
           "Click a row to read the full message and see what the BiLSTM focused on.")

thr = st.session_state.get("threshold", 0.5)
df = test_predictions()
names = model_names()

c1, c2, c3 = st.columns([1.3, 1.3, 2])
name = c1.selectbox("Model", names, index=len(names) - 1, format_func=lambda n: LONG_NAMES[n])
kind = c2.segmented_control("Error type", ["False positives", "False negatives"], default="False positives")
query = c3.text_input("Search in messages", placeholder="e.g. order, password, http")

pred = df[name] >= thr
fp_mask, fn_mask = pred & (df.label == 0), ~pred & (df.label == 1)
only_all = st.toggle("Only errors that every model makes",
                     help="Messages misclassified by all models at this threshold")
if only_all:
    for n in names:
        p = df[n] >= thr
        fp_mask &= p & (df.label == 0)
        fn_mask &= ~p & (df.label == 1)

m1, m2 = st.columns(2)
m1.metric("False positives (safe → flagged)", int(fp_mask.sum()), border=True)
m2.metric("False negatives (phishing → missed)", int(fn_mask.sum()), border=True)

mask = fp_mask if kind == "False positives" else fn_mask
errs = df[mask].copy()
if query:
    errs = errs[errs.text.str.contains(query, case=False, regex=False)]
errs = errs.sort_values(name, ascending=(kind == "False negatives"))
if errs.empty:
    st.success("No errors of this type for this selection.")
    st.stop()

table = pd.DataFrame({"P(phishing)": errs[name].values,
                      "true label": ["phishing" if v else "safe" for v in errs.label],
                      "message": errs.text.str.slice(0, 220).str.replace(r"\s+", " ", regex=True).values})
event = st.dataframe(table, hide_index=True, width="stretch", height=360, on_select="rerun",
                     selection_mode="single-row", column_config={
                         "P(phishing)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                         "message": st.column_config.TextColumn(width="large")})

rows = event.selection.rows if event and event.selection else []
if not rows:
    st.info("Select a row to inspect the message.")
    st.stop()

row = errs.iloc[rows[0]]
st.subheader("Selected message")
st.markdown(f"**True label:** {'phishing' if row.label else 'safe'} · "
            f"**{SHORT_NAMES[name]}** says P(phishing) = {row[name]:.3f}")
model_scores({n: float(row[n]) for n in names}, thr)
with st.container(border=True):
    st.text(row.text[:4000])
pairs = attention(row.text)
if pairs:
    st.markdown("**BiLSTM attention** (darker = more important)")
    st.html(attention_html(pairs))
