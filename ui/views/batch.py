import pandas as pd
import streamlit as st

from backend import LONG_NAMES, PROCESSED_DIR, SHORT_NAMES, confusion, model_names, predict

MAX_ROWS = 5000
thr = st.session_state.get("threshold", 0.5)

st.title("Batch scan")
st.caption("Classify many messages at once. Upload a **CSV** (one message per row) or a **TXT** file "
           "(one message per line). If the CSV has a `label` column (1/0 or phishing/safe), "
           "accuracy and errors are computed too.")

src = st.radio("Source", ["Upload a file", "Use a random sample of the test set"], horizontal=True)
df = None
if src == "Upload a file":
    up = st.file_uploader("Messages file", type=["csv", "txt"])
    if up is not None:
        if up.name.lower().endswith(".txt"):
            lines = [line.strip() for line in up.getvalue().decode("utf-8", errors="ignore").splitlines()]
            df = pd.DataFrame({"text": [line for line in lines if line]})
        else:
            df = pd.read_csv(up)
else:
    n = st.slider("Sample size", 50, 1000, 200, 50)
    df = pd.read_csv(PROCESSED_DIR / "test.csv").sample(n, random_state=1).reset_index(drop=True)

if df is None or df.empty:
    st.stop()

text_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
text_col = st.selectbox("Column containing the message text", text_cols,
                        index=text_cols.index("text") if "text" in text_cols else 0)
if len(df) > MAX_ROWS:
    st.warning(f"Only the first {MAX_ROWS:,} rows are scanned.")
    df = df.head(MAX_ROWS)

names = model_names()
decider = st.selectbox("Model for the verdict", names, index=len(names) - 1, format_func=lambda n: LONG_NAMES[n])

with st.spinner(f"Scanning {len(df):,} messages..."):
    probs = predict(df[text_col].fillna("").astype(str))
out = pd.DataFrame({"message": df[text_col].astype(str)})
for name in names:
    out[f"P({SHORT_NAMES[name]})"] = probs[name].values
out["verdict"] = ["phishing" if p >= thr else "safe" for p in probs[decider]]

labels = None
if "label" in df.columns:
    raw = df["label"].astype(str).str.lower().str.strip()
    labels = raw.isin(["1", "phishing", "phishing email", "true"]).astype(int).values
    out["true label"] = ["phishing" if v else "safe" for v in labels]
    out["correct"] = out["verdict"] == out["true label"]

flagged = int((out.verdict == "phishing").sum())
c = st.columns(4)
c[0].metric("Messages scanned", f"{len(out):,}", border=True)
c[1].metric("Flagged as phishing", f"{flagged:,} ({flagged / len(out):.1%})", border=True)
if labels is not None:
    m = confusion(labels, probs[decider].values, thr)
    c[2].metric("Accuracy", f"{m['accuracy']:.2%}", border=True)
    c[3].metric("Errors", f"{m['FP']} FP · {m['FN']} FN", help="False positives · false negatives", border=True)

view = st.segmented_control("Show", ["All", "Flagged only", "Errors only"] if labels is not None
                            else ["All", "Flagged only"], default="All")
shown = out
if view == "Flagged only":
    shown = out[out.verdict == "phishing"]
elif view == "Errors only":
    shown = out[~out["correct"]]

front = [c for c in ["verdict", "true label", "correct"] if c in shown.columns]
shown = shown[front + [c for c in shown.columns if c not in front]]
st.dataframe(shown, width="stretch", hide_index=True, height=420, column_config={
    "verdict": st.column_config.TextColumn("Verdict", width="small"),
    "true label": st.column_config.TextColumn("True label", width="small"),
    "correct": st.column_config.CheckboxColumn("Correct", width="small"),
    "message": st.column_config.TextColumn("Message", width="large"),
    **{f"P({SHORT_NAMES[n]})": st.column_config.ProgressColumn(SHORT_NAMES[n], min_value=0, max_value=1,
                                                                format="%.3f", width="small") for n in names}})

st.download_button("Download results as CSV", out.to_csv(index=False).encode("utf-8"),
                   file_name="phishing_scan_results.csv", mime="text/csv", icon=":material/download:")
