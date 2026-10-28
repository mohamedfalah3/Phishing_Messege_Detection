"""Reusable display pieces for the UI pages."""
import html

import altair as alt
import pandas as pd
import streamlit as st

from backend import COLORS, SHORT_NAMES

CSS = """
<style>
.verdict {padding: 1.1rem 1.4rem; border-radius: 14px; color: #fff; margin-bottom: .5rem;}
.verdict.phishing {background: #eb6834;}
.verdict.safe {background: #2a78d6;}
.verdict .label {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.verdict .sub {font-size: 1rem; opacity: .95;}
.attn {line-height: 2.1; font-size: 1.02rem;}
.attn span {padding: 2px 4px; border-radius: 5px; margin: 0 1px;}
.chip {display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px;
       background: rgba(235,104,52,.14); color: inherit; font-size: .9rem;}
.tag {display: inline-block; padding: 1px 8px; border-radius: 6px; color: #fff; font-size: .8rem;
      font-weight: 600;}
.tag.phishing {background: #eb6834;} .tag.safe {background: #2a78d6;}
</style>
"""


def inject_css():
    st.html(CSS)


def label_of(p, thr):
    return "phishing" if p >= thr else "safe"


def verdict_card(p, thr, model_name):
    kind = label_of(p, thr)
    title = "Likely PHISHING" if kind == "phishing" else "Looks SAFE"
    st.html(f'<div class="verdict {kind}"><div class="label">{title}</div>'
            f'<div class="sub">{html.escape(model_name)}: P(phishing) = {p:.3f} '
            f'&nbsp;·&nbsp; threshold {thr:.2f}</div></div>')


def tag(kind):
    return f'<span class="tag {kind}">{kind.upper()}</span>'


def model_scores(probs: dict, thr):
    """One bordered tile per model with its probability and verdict."""
    cols = st.columns(len(probs))
    for col, (name, p) in zip(cols, probs.items()):
        with col.container(border=True):
            st.markdown(f"**{SHORT_NAMES[name]}**")
            st.html(f'<div style="font-size:1.6rem;font-weight:700">{p:.3f}</div>{tag(label_of(p, thr))}')
            st.progress(float(min(max(p, 0.0), 1.0)))


def attention_html(pairs):
    """Highlight each token by its attention weight (darker = more important)."""
    top = max(w for _, w in pairs) or 1.0
    spans = []
    for tok, w in pairs:
        a = 0.08 + 0.85 * (w / top) if w / top > 0.05 else 0
        style = f"background: rgba(235,104,52,{a:.2f});" if a else ""
        weight = "font-weight:700;" if w / top > 0.5 else ""
        spans.append(f'<span style="{style}{weight}" title="attention {w:.3f}">{html.escape(tok)}</span>')
    return f'<div class="attn">{" ".join(spans)}</div>'


def top_words_chart(pairs, k=10):
    df = pd.DataFrame(pairs, columns=["word", "weight"]).groupby("word", as_index=False)["weight"].sum()
    df = df.nlargest(k, "weight")
    return (alt.Chart(df).mark_bar(color=COLORS["phishing"], cornerRadiusEnd=4)
            .encode(x=alt.X("weight:Q", title="Attention weight"),
                    y=alt.Y("word:N", sort="-x", title=None),
                    tooltip=["word", alt.Tooltip("weight:Q", format=".3f")])
            .properties(height=28 * len(df) + 20))


def confusion_chart(c):
    df = pd.DataFrame([
        {"Actual": "Safe", "Predicted": "Safe", "n": c["TN"], "cell": "TN"},
        {"Actual": "Safe", "Predicted": "Phishing", "n": c["FP"], "cell": "FP (false alarm)"},
        {"Actual": "Phishing", "Predicted": "Safe", "n": c["FN"], "cell": "FN (missed)"},
        {"Actual": "Phishing", "Predicted": "Phishing", "n": c["TP"], "cell": "TP"},
    ])
    order = ["Safe", "Phishing"]
    base = alt.Chart(df).encode(x=alt.X("Predicted:N", sort=order, axis=alt.Axis(labelAngle=0, orient="top")),
                                y=alt.Y("Actual:N", sort=order))
    heat = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("n:Q", scale=alt.Scale(type="log", range=["#eef4fc", "#0d366b"]), legend=None),
        tooltip=["cell", "n"])
    labels = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text=alt.Text("label:N"),
        color=alt.condition(alt.datum.n > 300, alt.value("white"), alt.value("#0b0b0b")),
    ).transform_calculate(label="datum.cell + ': ' + format(datum.n, ',')")
    return (heat + labels).properties(height=260)


def sweep_chart(sweep, thr):
    long = sweep.melt("threshold", var_name="Error type", value_name="Messages")
    lines = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("threshold:Q", title="Decision threshold"),
        y=alt.Y("Messages:Q", title="Errors on the test set (axis capped at 150)", scale=alt.Scale(domain=[0, 150], clamp=True)),
        color=alt.Color("Error type:N", scale=alt.Scale(domain=["False positives", "False negatives"],
                                                        range=[COLORS["safe"], COLORS["phishing"]]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["threshold", "Error type", "Messages"])
    rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(strokeDash=[5, 4], color="#52514e").encode(x="t:Q")
    return (lines + rule).properties(height=320)
