"""Text normalisation, tokenisation, vocabulary and PyTorch dataset helpers."""
import json
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

from config import MAX_LEN, MAX_VOCAB, MIN_FREQ

PAD, UNK = "<pad>", "<unk>"

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(com|net|org|info|biz|ru|cn|tk)\b\S*)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MONEY_RE = re.compile(r"[$€£]\s?\d[\d,.]*")
NUM_RE = re.compile(r"\b\d+\b")
TOKEN_RE = re.compile(r"[a-z]+|[!?$%]")


def normalize(text: str) -> str:
    """Lower-case and replace URLs, e-mail addresses, money and numbers with
    placeholder words, so the model learns *that* a link is present rather than
    memorising individual links."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, min_freq=MIN_FREQ, max_size=MAX_VOCAB):
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls([PAD, UNK] + words)

    def encode(self, tokens: list[str], max_len=MAX_LEN) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens[:max_len]] or [unk]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


class MessageDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab):
        self.ids = [vocab.encode(tokenize(t)) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    """Pad a batch to its longest sequence; also return true lengths."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s)
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)
