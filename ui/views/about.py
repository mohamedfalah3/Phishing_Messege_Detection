import streamlit as st

from backend import figure

st.title("Dataset & method")

st.header("1. Dataset")
st.markdown(
    "**Source:** [Phishing Email Dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) "
    "(18,650 emails labelled *Safe* or *Phishing*).\n\n"
    "**Cleaning:** removed empty rows and **577 exact duplicates** (so no message appears in both training "
    "and test data), plus messages under 3 tokens → **17,511 messages**: 10,975 safe and 6,536 phishing (37%).\n\n"
    "**Split** (stratified): 70% train (12,257) · 15% validation (2,627) · 15% test (2,627).")
if figure("eda_overview.png"):
    st.image(figure("eda_overview.png"), width="stretch")

st.header("2. Preprocessing")
c = st.columns(4)
steps = [("Normalise", "Lower-case; links → `urltoken`, emails → `emailtoken`, money → `moneytoken`, "
                       "numbers → `numtoken`"),
         ("Tokenise", "Split into words; keep `! ? $ %` as tokens"),
         ("Vocabulary", "30,000 most frequent words from the training set only; unknown → `<unk>`"),
         ("Fixed length", "First 300 tokens (covers the whole message for 75% of emails)")]
for col, (i, (title, desc)) in zip(c, enumerate(steps, 1)):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{i}. {title}**")
        st.caption(desc)

st.header("3. Models")
m = st.columns(3)
with m[0].container(border=True, height="stretch"):
    st.markdown("**TF-IDF + Logistic Regression** · baseline")
    st.markdown("- TF-IDF weights for up to 60k words and word pairs\n- Logistic regression: one weight per "
                "feature\n- Trains in seconds")
with m[1].container(border=True, height="stretch"):
    st.markdown("**TextCNN** · deep learning")
    st.markdown("- 128-d word embeddings\n- 3 × 128 convolution filters over 3, 4 and 5 words\n"
                "- Max-over-time pooling → dropout → dense → sigmoid\n- 4.04M parameters")
with m[2].container(border=True, height="stretch"):
    st.markdown("**BiLSTM + Attention** · deep learning")
    st.markdown("- 128-d word embeddings\n- Bidirectional LSTM, 128 units per direction\n"
                "- Attention layer weights each word (explainable)\n- 4.10M parameters")

st.header("4. Training")
st.markdown("Binary cross-entropy loss · AdamW (lr 0.001) · batch 64 · dropout · gradient clipping · "
            "early stopping after 3 epochs without validation improvement · CPU only.")
for f in ["training_curves_cnn.png", "training_curves_bilstm.png"]:
    if figure(f):
        st.image(figure(f), width="stretch")

st.header("5. Evaluation")
st.markdown(
    "- Metrics on the **test set only**: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC, "
    "false positive rate and false negative rate.\n"
    "- Two thresholds: **0.5** and a **cost-tuned** threshold chosen on validation data, assuming a missed "
    "phishing message costs 3× a false alarm.\n"
    "- **Error analysis:** misclassified messages, error rate by length, errors shared by all models, and "
    "BiLSTM attention words.\n\n"
    "The full write-up is in `report/REPORT.md`.")
