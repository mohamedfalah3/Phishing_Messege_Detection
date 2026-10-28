import streamlit as st

from backend import EXAMPLES, LONG_NAMES, attention, model_names, predict, signals
from components import attention_html, model_scores, top_words_chart, verdict_card

thr = st.session_state.get("threshold", 0.5)

st.title("Analyze a message")
st.caption("Paste an email or SMS, or start from an example. Everything runs locally with the trained models.")


def load_example():
    choice = st.session_state.example
    if choice in EXAMPLES:
        st.session_state.msg = EXAMPLES[choice]


left, right = st.columns([3, 1])
with right:
    st.selectbox("Example messages", ["(write your own)"] + list(EXAMPLES), key="example",
                 on_change=load_example)
    names = model_names()
    decider = st.selectbox("Model for the verdict", names, index=len(names) - 1,
                           format_func=lambda n: LONG_NAMES[n])
with left:
    text = st.text_area("Message", key="msg", height=190,
                        placeholder="e.g. Your account has been suspended. Verify now at http://...")
    st.button("Analyze", type="primary", icon=":material/search:")

if not text or not text.strip():
    st.info("Enter a message above and press **Analyze**.")
    st.stop()

probs = predict([text]).iloc[0].to_dict()
verdict_card(probs[decider], thr, LONG_NAMES[decider])

st.subheader("Every model's opinion")
model_scores(probs, thr)

st.subheader("Why? What the BiLSTM paid attention to")
pairs = attention(text)
if pairs:
    a, b = st.columns([2, 1])
    with a:
        st.html(attention_html(pairs))
        st.caption("Darker = more attention. Text is shown after preprocessing: links → `urltoken`, "
                   "numbers → `numtoken`, money → `moneytoken`, emails → `emailtoken`. "
                   "Hover a word to see its weight.")
        if len(pairs) >= 300:
            st.warning("Long message: the deep models only read the first 300 tokens.")
    with b:
        st.altair_chart(top_words_chart(pairs), width="stretch")
else:
    st.caption("The BiLSTM model is not available, or the message has no recognisable words.")

st.subheader("Common phishing signals in the text")
found = signals(text)
if found:
    for name, hits in found.items():
        chips = "".join(f'<span class="chip">{h}</span>' for h in hits[:8])
        st.html(f"<b>{name}</b>&nbsp; {chips}")
else:
    st.write("None of the common signals were found.")
st.caption("These rule-based signals help a human read the message. The models do not use them; "
           "they learned their own patterns from data.")
