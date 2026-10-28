"""Model loading and scoring for the Streamlit UI. Reuses the project's own
code in ../src so the UI gives exactly the same predictions as the scripts."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (COST_FN, COST_FP, FIGURES_DIR, MODEL_NAMES, PROCESSED_DIR,  # noqa: E402
                    RESULTS_DIR)
from inference import available_models, load, score  # noqa: E402
from text_utils import MessageDataset, collate, tokenize  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
ERRORS_DIR = RESULTS_DIR / "errors"

SHORT_NAMES = {"baseline": "TF-IDF + LogReg", "cnn": "TextCNN", "bilstm": "BiLSTM + Attention",
               "ensemble": "Ensemble (average)"}
LONG_NAMES = {**MODEL_NAMES, "ensemble": "Ensemble (average of all models)"}
COLORS = {"safe": "#2a78d6", "phishing": "#eb6834", "baseline": "#2a78d6", "cnn": "#eb6834",
          "bilstm": "#1baf7a", "ensemble": "#0f1b2d"}

# Rule-based signals shown next to the model output, to help a human read the
# message. They are NOT used by the models.
SIGNALS = {
    "Link / URL": r"https?://|www\.|\b[\w-]+\.(com|net|org|info|biz|ru|cn|tk)\b",
    "Money amount": r"[$€£]\s?\d[\d,.]*\d|[$€£]\s?\d|\b\d+\s?(dollars|usd|million)\b",
    "Urgency": r"\b(urgent|immediately|within \d+ hours|asap|act now|expires?|final notice|limited time)\b",
    "Account / login": r"\b(account|login|log in|sign in|password|username|verify|verification|suspended|locked)\b",
    "Financial request": r"\b(bank|credit card|wire|transfer|payment|invoice|bitcoin|gift card)\b",
    "Prize / reward": r"\b(congratulations|winner|won|prize|reward|free|claim)\b",
    "Click request": r"\b(click|tap|open the attachment|follow the link)\b",
}

EXAMPLES = {
    "Account suspended (phishing)": "Dear customer, we detected unusual activity on your account. Your access has been "
        "suspended. Click here http://secure-verify-account.com/login to verify your identity within 24 hours or "
        "your account will be closed.",
    "Gift card prize (phishing)": "Congratulations! You have been selected to receive a $1,000 gift card. Claim your "
        "prize now by providing your name, address and credit card number for shipping.",
    "Inheritance scam (phishing)": "I am a lawyer representing the estate of a deceased client who left $15.5 million. "
        "Kindly reply with your bank details so we can transfer your share of the funds.",
    "Mailbox full (phishing, hard)": "URGENT: Your mailbox has exceeded its storage limit. Login at "
        "www.mail-upgrade.net with your username and password to avoid losing your emails.",
    "Meeting moved (safe)": "Hi team, the meeting about the quarterly report has been moved to Thursday at 3pm. "
        "Please bring your notes on the project schedule. Thanks, John",
    "Document review (safe)": "Thanks for sending the draft. I made a few comments on section 2 and attached the "
        "revised version. Let me know if you have questions before Friday.",
    "Order confirmation (safe, hard)": "Thank you for shopping with us. Your order number is 11871528 and is pending "
        "credit card approval. You can track your order at http://www.example-store.com/orders.",
}


@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    return {name: load(name) for name in available_models()}


def model_names():
    names = list(get_models())
    return names + (["ensemble"] if len(names) > 1 else [])


def predict(texts) -> pd.DataFrame:
    """P(phishing) from every model (+ ensemble) for a list of texts."""
    texts = [str(t) for t in texts]
    out = {}
    for name, (model, vocab) in get_models().items():
        out[name], _ = score(name, model, vocab, texts)
    df = pd.DataFrame(out)
    if len(out) > 1:
        df["ensemble"] = df.mean(axis=1)
    return df


@torch.no_grad()
def attention(text):
    """(token, weight) pairs from the BiLSTM attention layer, or None."""
    models = get_models()
    if "bilstm" not in models:
        return None
    model, vocab = models["bilstm"]
    tokens = tokenize(text)[:300]
    if not tokens:
        return None
    x, lengths, _ = collate([MessageDataset([text], [0], vocab)[0]])
    model(x, lengths)
    weights = model.last_attention[0].numpy()
    return list(zip(tokens, weights[: len(tokens)]))


def signals(text):
    found = {}
    for name, pattern in SIGNALS.items():
        hits = [m.group(0) for m in re.finditer(pattern, text, flags=re.I)]
        if hits:
            found[name] = sorted(set(h.strip().lower() for h in hits))
    return found


@st.cache_data(show_spinner="Scoring the test set with every model (first run only)...")
def test_predictions() -> pd.DataFrame:
    """Test-set texts, labels and every model's probability. Cached on disk
    because the BiLSTM takes a while on a CPU."""
    cache = CACHE_DIR / "test_predictions.csv"
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    if cache.exists():
        cached = pd.read_csv(cache)
        if len(cached) == len(test) and set(available_models()) <= set(cached.columns):
            return cached
    preds = predict(test.text)
    df = pd.concat([test.reset_index(drop=True), preds], axis=1)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def confusion(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0}


def threshold_sweep(y, p):
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    return pd.DataFrame({"threshold": grid,
                         "False positives": [int(((p >= t) & (y == 0)).sum()) for t in grid],
                         "False negatives": [int(((p < t) & (y == 1)).sum()) for t in grid]})


@st.cache_data
def metrics():
    path = RESULTS_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def cost_threshold(name):
    m = metrics().get(name, {})
    return m.get("cost_tuned", {}).get("threshold")


def errors(name, kind):
    path = ERRORS_DIR / f"{name}_{kind}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def figure(name):
    path = FIGURES_DIR / name
    return str(path) if path.exists() else None


__all__ = ["COST_FN", "COST_FP", "COLORS", "EXAMPLES", "LONG_NAMES", "SHORT_NAMES", "PROCESSED_DIR",
           "attention", "confusion", "cost_threshold", "errors", "figure", "get_models", "metrics",
           "model_names", "predict", "signals", "test_predictions", "threshold_sweep"]
