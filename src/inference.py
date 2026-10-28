"""Load trained models and score messages (shared by evaluate.py and predict.py)."""
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import MODELS_DIR
from models import build_model
from text_utils import MessageDataset, Vocab, collate


def available_models():
    names = []
    if (MODELS_DIR / "baseline.joblib").exists():
        names.append("baseline")
    names += [n for n in ("cnn", "bilstm") if (MODELS_DIR / f"{n}.pt").exists()]
    return names


def load(name):
    if name == "baseline":
        return joblib.load(MODELS_DIR / "baseline.joblib"), None
    ckpt = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu")
    vocab = Vocab.load(MODELS_DIR / "vocab.json")
    model = build_model(ckpt["model"], len(vocab), ckpt["params"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab


@torch.no_grad()
def score(name, model, vocab, texts, top_k=0):
    """Return P(phishing) for each text. For the BiLSTM, top_k > 0 also returns
    the top_k most-attended words of each message as an explanation."""
    texts = list(texts)
    if name == "baseline":
        return model.predict_proba(texts)[:, 1], None
    loader = DataLoader(MessageDataset(texts, [0] * len(texts), vocab), batch_size=256,
                        collate_fn=collate)
    probs, explanations = [], []
    for x, lengths, _ in loader:
        probs.append(torch.sigmoid(model(x, lengths)).numpy())
        if top_k and getattr(model, "last_attention", None) is not None:
            for row_ids, row_w in zip(x, model.last_attention):
                order = torch.argsort(row_w, descending=True).tolist()
                words = []
                for i in order:
                    w = vocab.itos[row_ids[i]]
                    if row_ids[i] != 0 and w not in words:
                        words.append(w)
                    if len(words) == top_k:
                        break
                explanations.append(words)
    return np.concatenate(probs), (explanations or None)
