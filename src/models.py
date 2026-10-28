"""Deep learning text classifiers. Both output one logit per message
(sigmoid(logit) = probability the message is phishing)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCNN(nn.Module):
    """Kim (2014) style CNN: parallel 1-D convolutions with different window
    sizes act as learned n-gram detectors ("verify your account", "click here"),
    followed by max-over-time pooling."""

    def __init__(self, vocab_size, embed_dim=128, num_filters=128,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
                                   for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x, lengths):
        mask = (x != 0).unsqueeze(1)                       # (B, 1, T)
        emb = self.embedding(x).transpose(1, 2)            # (B, E, T)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))[:, :, : x.size(1)]
            h = h.masked_fill(~mask, float("-inf"))         # ignore padding
            pooled.append(h.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM reads the message in both directions; an attention
    layer then weights each word's contribution. The attention weights tell us
    which words drove the decision - useful for explaining FPs and FNs."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, 1)
        self.last_attention = None

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))  # (B, T, 2H)
        scores = self.attn(torch.tanh(out)).squeeze(2)                               # (B, T)
        scores = scores.masked_fill(x == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        self.last_attention = weights.detach()
        context = (weights.unsqueeze(2) * out).sum(dim=1)                            # (B, 2H)
        return self.fc(self.dropout(context)).squeeze(1)


def build_model(name, vocab_size, params):
    cls = {"cnn": TextCNN, "bilstm": BiLSTMAttention}[name]
    return cls(vocab_size, **params)
