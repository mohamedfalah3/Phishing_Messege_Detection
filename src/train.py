"""Step 2 - train a deep learning classifier.

Usage:  python train.py --model cnn      (TextCNN)
        python train.py --model bilstm   (BiLSTM + attention)
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import plot_style  # noqa: F401
import matplotlib.pyplot as plt
from config import (BATCH_SIZE, EPOCHS, FIGURES_DIR, GRAD_CLIP, LR, MODEL_COLORS,
                    MODEL_NAMES, MODEL_PARAMS, MODELS_DIR, PATIENCE, PROCESSED_DIR,
                    SEED, WEIGHT_DECAY)
from models import build_model
from text_utils import MessageDataset, Vocab, collate, tokenize


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    probs, labels, total_loss = [], [], 0.0
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    for x, lengths, y in loader:
        logits = model(x, lengths)
        total_loss += loss_fn(logits, y).item()
        probs.append(torch.sigmoid(logits))
        labels.append(y)
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return probs, labels, total_loss / len(labels)


def plot_history(history, name):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    color = MODEL_COLORS[name]
    axes[0].plot(epochs, [h["train_loss"] for h in history], color=color, label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], color=color, linestyle="--", label="Validation")
    axes[0].set_title("Loss (binary cross-entropy)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [h["train_acc"] for h in history], color=color, label="Train accuracy")
    axes[1].plot(epochs, [h["val_acc"] for h in history], color=color, linestyle="--",
                 label="Validation accuracy")
    axes[1].plot(epochs, [h["val_f1"] for h in history], color="#52514e", linestyle=":", label="Validation F1")
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(f"Training curves - {MODEL_NAMES[name]}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"training_curves_{name}.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "bilstm"], default="bilstm")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    vocab_path = MODELS_DIR / "vocab.json"
    vocab = Vocab.build(tokenize(t) for t in train_df.text)  # built from TRAIN only
    vocab.save(vocab_path)
    print(f"Vocabulary size: {len(vocab):,}")

    train_dl = DataLoader(MessageDataset(train_df.text, train_df.label, vocab),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(MessageDataset(val_df.text, val_df.label, vocab),
                        batch_size=256, collate_fn=collate)

    params = MODEL_PARAMS[args.model]
    model = build_model(args.model, len(vocab), params)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAMES[args.model]}  ({n_params:,} trainable parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, bad_epochs, history = float("inf"), 0, []
    ckpt_path = MODELS_DIR / f"{args.model}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, lengths, y in train_dl:
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            run_loss += loss.item() * len(y)
            correct += ((logits > 0).float() == y).sum().item()
            seen += len(y)

        val_probs, val_labels, val_loss = predict_proba(model, val_dl)
        val_pred = (val_probs >= 0.5).astype(int)
        rec = {"epoch": epoch, "train_loss": run_loss / seen, "train_acc": correct / seen,
               "val_loss": val_loss, "val_acc": float((val_pred == val_labels).mean()),
               "val_f1": float(f1_score(val_labels, val_pred)), "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"epoch {epoch:2d} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
              f"val loss {val_loss:.4f} acc {rec['val_acc']:.4f} f1 {rec['val_f1']:.4f} | {rec['seconds']}s")
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            torch.save({"model": args.model, "params": params, "state_dict": model.state_dict(),
                        "epoch": epoch}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping: no validation improvement for {PATIENCE} epochs.")
                break

    with open(MODELS_DIR / f"{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, args.model)
    print(f"Best checkpoint (val loss {best_val:.4f}) saved to {ckpt_path}")


if __name__ == "__main__":
    main()
