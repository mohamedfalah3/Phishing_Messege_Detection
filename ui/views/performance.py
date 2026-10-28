import pandas as pd
import streamlit as st

from backend import (COST_FN, COST_FP, LONG_NAMES, confusion, cost_threshold, figure, model_names,
                     test_predictions, threshold_sweep)
from components import confusion_chart, sweep_chart

st.title("Model performance")
st.caption("All numbers are computed live on the held-out test set (2,627 messages) at the threshold "
           "chosen in the sidebar.")

df = test_predictions()
y = df.label.values
names = model_names()

name = st.selectbox("Model", names, index=len(names) - 1, format_func=lambda n: LONG_NAMES[n])
cost_thr = cost_threshold(name)


def use_threshold(value):
    st.session_state.threshold = value


b1, b2, _ = st.columns([1, 1.6, 3])
b1.button("Reset to 0.50", on_click=use_threshold, args=(0.5,), width="stretch")
if cost_thr is not None:
    b2.button(f"Use cost-tuned threshold ({cost_thr:.2f})", on_click=use_threshold, args=(cost_thr,),
              width="stretch", help=f"Chosen on the validation set to minimise {COST_FN:g}×FN + {COST_FP:g}×FP")

thr = st.session_state.get("threshold", 0.5)
c = confusion(y, df[name].values, thr)
cols = st.columns(6)
cols[0].metric("Accuracy", f"{c['accuracy']:.2%}", border=True)
cols[1].metric("Precision", f"{c['precision']:.2%}", border=True)
cols[2].metric("Recall", f"{c['recall']:.2%}", border=True)
cols[3].metric("F1", f"{c['f1']:.4f}", border=True)
cols[4].metric("False positives", f"{c['FP']} ({c['fpr']:.1%})", help="Safe messages flagged (false positive rate)",
               border=True)
cols[5].metric("False negatives", f"{c['FN']} ({c['fnr']:.1%})", help="Phishing messages missed (false negative rate)",
               border=True)

left, right = st.columns([1, 1.5])
with left:
    st.markdown(f"**Confusion matrix** · threshold {thr:.2f}")
    st.altair_chart(confusion_chart(c), width="stretch")
with right:
    st.markdown("**False positive / false negative trade-off** (dashed line = current threshold)")
    st.altair_chart(sweep_chart(threshold_sweep(y, df[name].values), thr), width="stretch")

st.subheader(f"All models at threshold {thr:.2f}")
rows = []
for n in names:
    m = confusion(y, df[n].values, thr)
    rows.append({"Model": LONG_NAMES[n], "Accuracy": m["accuracy"], "Precision": m["precision"],
                 "Recall": m["recall"], "F1": m["f1"], "False positives": m["FP"], "False negatives": m["FN"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
    k: st.column_config.NumberColumn(format="percent") for k in ["Accuracy", "Precision", "Recall"]} | {
    "F1": st.column_config.NumberColumn(format="%.4f")})

st.subheader("Saved evaluation figures")
tabs = st.tabs(["ROC & precision-recall", "Training curves", "Error counts", "Probability distribution"])
figs = [["roc_pr_curves.png"], ["training_curves_cnn.png", "training_curves_bilstm.png"],
        ["error_counts.png", "threshold_tradeoff.png"], ["probability_hist_cnn.png"]]
for tab, files in zip(tabs, figs):
    with tab:
        for f in files:
            path = figure(f)
            if path:
                st.image(path, width="stretch")
