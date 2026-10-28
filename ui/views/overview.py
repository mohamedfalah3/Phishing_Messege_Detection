import pandas as pd
import streamlit as st

from backend import LONG_NAMES, metrics

st.title("Phishing Message Detection with Deep Learning")
st.markdown(
    "This app uses text classifiers trained on **17,511 labelled emails** to estimate how likely a message "
    "is to be **phishing**. It compares two deep learning models (**TextCNN** and **BiLSTM + Attention**) "
    "with a classical **TF-IDF + Logistic Regression** baseline, and looks closely at their "
    "**false positives** (safe mail flagged) and **false negatives** (phishing missed).")

m = metrics()
if not m:
    st.warning("No results found. Run the training pipeline first: `python run_all.py`")
    st.stop()

best = m.get("ensemble", m.get("baseline"))["threshold_0.5"]
st.subheader("Headline results on the test set")
st.caption("2,627 messages the models never saw during training · threshold 0.5 · ensemble of all models")
c = st.columns(5)
c[0].metric("Accuracy", f"{best['accuracy']:.2%}", border=True)
c[1].metric("F1 score", f"{best['f1']:.3f}", border=True)
c[2].metric("ROC-AUC", f"{best['roc_auc']:.4f}", border=True)
c[3].metric("False positives", best["FP"], help="Safe messages wrongly flagged", border=True)
c[4].metric("False negatives", best["FN"], help="Phishing messages missed", border=True)

rows = []
for name in [k for k in m if not k.startswith("_")]:
    r = m[name]["threshold_0.5"]
    rows.append({"Model": LONG_NAMES[name], "Accuracy": r["accuracy"], "Precision": r["precision"],
                 "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"],
                 "False positives": r["FP"], "False negatives": r["FN"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
    c: st.column_config.NumberColumn(format="percent") for c in ["Accuracy", "Precision", "Recall"]} | {
    "F1": st.column_config.NumberColumn(format="%.4f"), "ROC-AUC": st.column_config.NumberColumn(format="%.4f")})

st.subheader("What you can do here")
cards = [
    ("views/analyze.py", ":material/search:", "Analyze a message",
     "Paste any message and see each model's verdict, the words the BiLSTM focused on, and common phishing signals."),
    ("views/batch.py", ":material/upload_file:", "Batch scan",
     "Upload a CSV or TXT file of messages, classify them all and download the results."),
    ("views/performance.py", ":material/monitoring:", "Model performance",
     "Move the threshold and watch accuracy, false positives and false negatives change live."),
    ("views/errors.py", ":material/bug_report:", "Error explorer",
     "Read the messages each model gets wrong and see why."),
]
cols = st.columns(4)
for col, (page, icon, title, desc) in zip(cols, cards):
    with col.container(border=True, height="stretch"):
        st.page_link(page, label=title, icon=icon)
        st.caption(desc)
