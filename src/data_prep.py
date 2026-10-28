"""Step 1 - download, clean, explore and split the labelled message dataset.

Output: data/processed/{train,val,test}.csv with columns `text`, `label`
(1 = phishing, 0 = safe) plus a few exploratory figures.
"""
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import plot_style  # noqa: F401  (applies figure style)
import matplotlib.pyplot as plt
from config import (DATA_URL, FIGURES_DIR, MODEL_COLORS, PROCESSED_DIR, RAW_CSV,
                    SEED, TEST_SIZE, VAL_SIZE)
from text_utils import tokenize


def download():
    if RAW_CSV.exists():
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, RAW_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Email Text": "text", "Email Type": "label_name"})[["text", "label_name"]]
    n0 = len(df)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["text"].str.lower() != "empty"]
    n1 = len(df)
    # Remove exact duplicates so the same message can't appear in train AND test.
    df = df.drop_duplicates(subset="text")
    n2 = len(df)
    df["n_tokens"] = df["text"].map(lambda t: len(tokenize(t)))
    df = df[df["n_tokens"] >= 3]  # near-empty messages carry no signal
    n3 = len(df)
    df["label"] = (df["label_name"] == "Phishing Email").astype(int)
    print(f"Rows: raw={n0}  after removing empty={n1}  after dedup={n2}  "
          f"after removing <3 tokens={n3}")
    return df.reset_index(drop=True)


def plot_eda(df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    counts = df["label"].value_counts().sort_index()
    ax = axes[0]
    bars = ax.bar(["Safe", "Phishing"], counts.values, width=0.55,
                  color=[MODEL_COLORS["baseline"], MODEL_COLORS["cnn"]])
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=3)
    ax.set_title("Class balance after cleaning")
    ax.set_ylabel("Messages")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    bins = np.logspace(np.log10(3), np.log10(5000), 40)
    for lab, name, key in [(0, "Safe", "baseline"), (1, "Phishing", "cnn")]:
        ax.hist(df.loc[df.label == lab, "n_tokens"].clip(upper=5000), bins=bins,
                histtype="step", linewidth=2, label=name, color=MODEL_COLORS[key])
    ax.axvline(300, color="#52514e", linestyle="--", linewidth=1)
    ax.text(310, ax.get_ylim()[1] * 0.9, "MAX_LEN = 300", color="#52514e")
    ax.set_xscale("log")
    ax.set_title("Message length (tokens)")
    ax.set_xlabel("Tokens per message (log scale)")
    ax.set_ylabel("Messages")
    ax.legend()
    fig.savefig(FIGURES_DIR / "eda_overview.png")
    plt.close(fig)


def main():
    download()
    df = clean(pd.read_csv(RAW_CSV))
    print(df["label_name"].value_counts().to_string())
    print(df.groupby("label_name")["n_tokens"].describe().round(1).to_string())
    plot_eda(df)

    train_val, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["label"],
                                       random_state=SEED)
    train, val = train_test_split(train_val, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                  stratify=train_val["label"], random_state=SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["text", "label"]].to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):6d} messages, phishing share = {part.label.mean():.3f}")


if __name__ == "__main__":
    main()
