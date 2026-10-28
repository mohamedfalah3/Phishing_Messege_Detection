"""Step 3 - evaluate every trained model on the held-out TEST set, with a
focus on false positives (safe messages flagged as phishing) and false
negatives (phishing messages that slip through).

Outputs (in results/):
  metrics.json, results_summary.md, figures/*.png,
  errors/<model>_false_positives.csv, errors/<model>_false_negatives.csv
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import plot_style  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from config import (COST_FN, COST_FP, FIGURES_DIR, MODEL_COLORS, MODEL_NAMES,
                    PROCESSED_DIR, RESULTS_DIR, TEXT, TEXT_MUTED)
from inference import available_models, load, score
from text_utils import tokenize

BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ERRORS_DIR = RESULTS_DIR / "errors"


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 3),
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "specificity": tn / (tn + fp), "mcc": matthews_corrcoef(y, pred),
        "false_positive_rate": fp / (fp + tn), "false_negative_rate": fn / (fn + tp),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
    }


def cost_threshold(y, p):
    """Threshold that minimises COST_FN*FN + COST_FP*FP (chosen on VALIDATION)."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = [COST_FN * ((p < t) & (y == 1)).sum() + COST_FP * ((p >= t) & (y == 0)).sum()
             for t in grid]
    return float(grid[int(np.argmin(costs))])


# ---------------------------------------------------------------- figures
def plot_confusion(results, y):
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.9))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y, (results[name]["p"] >= 0.5).astype(int))
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
        cells = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cells[i][j]}\n{cm[i, j]:,}\n({share[i, j]:.1%})", ha="center",
                        va="center", fontsize=10, color="white" if share[i, j] > 0.5 else TEXT)
        ax.set_xticks([0, 1], ["Safe", "Phishing"])
        ax.set_yticks([0, 1], ["Safe", "Phishing"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        ax.set_title(MODEL_NAMES[name].split(" (")[0], fontsize=11)
    fig.suptitle("Confusion matrices on the test set (threshold 0.5)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png")
    plt.close(fig)


def plot_curves(results, y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y, r["p"])
        ax1.plot(fpr, tpr, color=MODEL_COLORS[name],
                 label=f"{MODEL_NAMES[name].split(' (')[0]} (AUC {r['m05']['roc_auc']:.4f})")
        prec, rec, _ = precision_recall_curve(y, r["p"])
        ax2.plot(rec, prec, color=MODEL_COLORS[name],
                 label=f"{MODEL_NAMES[name].split(' (')[0]} (AP {r['m05']['pr_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], color=TEXT_MUTED, linestyle=":", linewidth=1)
    ax1.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate (recall)")
    ax2.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    # Zoomed insets: all models are near-perfect, the interesting part is the corner
    for ax, xl, yl in [(ax1, (0, 0.1), (0.9, 1.0)), (ax2, (0.9, 1.0), (0.9, 1.0))]:
        ins = ax.inset_axes([0.42, 0.2, 0.5, 0.45], zorder=5)
        for name, r in results.items():
            if ax is ax1:
                a, b, _ = roc_curve(y, r["p"])
            else:
                b, a, _ = precision_recall_curve(y, r["p"])
            ins.plot(a, b, color=MODEL_COLORS[name], linewidth=1.5)
        ins.set_xlim(*xl)
        ins.set_ylim(*yl)
        ins.tick_params(labelsize=7)
        ins.set_title("zoom", fontsize=8, color=TEXT_MUTED, loc="right")
    ax1.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1, -0.02))
    ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png")
    plt.close(fig)


def plot_threshold_tradeoff(results, y):
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    grid = np.linspace(0.01, 0.99, 99)
    for ax, name in zip(axes, names):
        p = results[name]["p"]
        fps = [((p >= t) & (y == 0)).sum() for t in grid]
        fns = [((p < t) & (y == 1)).sum() for t in grid]
        ax.plot(grid, fps, color=MODEL_COLORS["baseline"], label="False positives")
        ax.plot(grid, fns, color=MODEL_COLORS["cnn"], label="False negatives")
        ax.axvline(0.5, color=TEXT_MUTED, linestyle=":", linewidth=1)
        ax.axvline(results[name]["thr_cost"], color=TEXT, linestyle="--", linewidth=1)
        ax.set_title(MODEL_NAMES[name].split(" (")[0], fontsize=11)
        ax.set_xlabel("Decision threshold")
    axes[0].set_ylabel("Errors on test set (axis capped at 150)")
    axes[0].set_ylim(0, 150)
    for ax, name in zip(axes, names):
        t = results[name]["thr_cost"]
        ax.text(t + 0.015, 138, f"cost-tuned {t:.2f}", fontsize=8, color=TEXT)
        ax.text(0.515, 125, "0.5", fontsize=8, color=TEXT_MUTED)
    fig.suptitle("False positive / false negative trade-off as the threshold moves",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2,
               bbox_to_anchor=(0.99, 1.0))
    fig.savefig(FIGURES_DIR / "threshold_tradeoff.png")
    plt.close(fig)


def plot_error_bars(results):
    names = list(results)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    groups = [("FP", "m05", "False positives\n@0.5"), ("FN", "m05", "False negatives\n@0.5"),
              ("FP", "mcost", "False positives\n@cost-tuned"), ("FN", "mcost", "False negatives\n@cost-tuned")]
    width = 0.8 / len(names)
    x = np.arange(len(groups))
    for k, name in enumerate(names):
        vals = [results[name][m][key] for key, m, _ in groups]
        bars = ax.bar(x + (k - (len(names) - 1) / 2) * width, vals, width * 0.92,
                      color=MODEL_COLORS[name], label=MODEL_NAMES[name].split(" (")[0])
        ax.bar_label(bars, padding=2, fontsize=8, color=TEXT_MUTED)
    ax.set_xticks(x, [g[2] for g in groups])
    ax.set_ylabel("Messages (test set)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Error counts by model")
    ax.legend(loc="upper left", ncol=len(names), fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.savefig(FIGURES_DIR / "error_counts.png")
    plt.close(fig)


def plot_probability_hist(results, y, name):
    p = results[name]["p"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, color=MODEL_COLORS["baseline"], histtype="step", linewidth=2,
            label="Actually safe")
    ax.hist(p[y == 1], bins=bins, color=MODEL_COLORS["cnn"], histtype="step", linewidth=2,
            label="Actually phishing")
    ax.set_yscale("log")
    ax.axvline(0.5, color=TEXT_MUTED, linestyle=":", linewidth=1)
    ax.set(title=f"Predicted phishing probability - {MODEL_NAMES[name].split(' (')[0]}",
           xlabel="P(phishing)", ylabel="Messages (log scale)")
    ax.legend(loc="upper center")
    fig.savefig(FIGURES_DIR / f"probability_hist_{name}.png")
    plt.close(fig)


# ---------------------------------------------------------- error analysis
def save_errors(name, test, r, thr=0.5):
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    df = test.assign(p_phishing=r["p"].round(4), n_tokens=test.text.map(lambda t: len(tokenize(t))))
    if r["expl"] is not None:
        df["top_attention_words"] = [", ".join(w) for w in r["expl"]]
    df["text"] = df.text.str.slice(0, 600).str.replace(r"\s+", " ", regex=True)
    fp = df[(df.label == 0) & (df.p_phishing >= thr)].sort_values("p_phishing", ascending=False)
    fn = df[(df.label == 1) & (df.p_phishing < thr)].sort_values("p_phishing")
    fp.to_csv(ERRORS_DIR / f"{name}_false_positives.csv", index=False)
    fn.to_csv(ERRORS_DIR / f"{name}_false_negatives.csv", index=False)
    return df, fp, fn


def error_profile(df, fp, fn):
    """Simple descriptive analysis of what the misclassified messages look like."""
    def stats(part):
        toks = part.text.map(tokenize)
        return {"count": len(part),
                "median_tokens": float(part.n_tokens.median()) if len(part) else 0,
                "share_with_url": float(toks.map(lambda t: "urltoken" in t).mean()) if len(part) else 0,
                "mean_confidence_wrong": float((part.p_phishing - 0.5).abs().mean() * 2) if len(part) else 0}
    correct = df[~df.index.isin(fp.index) & ~df.index.isin(fn.index)]
    return {"false_positives": stats(fp), "false_negatives": stats(fn),
            "correct_safe": stats(correct[correct.label == 0]),
            "correct_phishing": stats(correct[correct.label == 1])}


def attention_words(fp, fn):
    out = {}
    for key, part in [("false_positives", fp), ("false_negatives", fn)]:
        if "top_attention_words" in part:
            c = Counter(w for s in part.top_attention_words for w in s.split(", ") if w)
            out[key] = c.most_common(15)
    return out


def length_buckets(df, name):
    buckets = pd.cut(df.n_tokens, [0, 30, 100, 300, 1000, np.inf],
                     labels=["<30", "30-100", "100-300", "300-1000", ">1000"])
    pred = (df.p_phishing >= 0.5).astype(int)
    g = pd.DataFrame({"bucket": buckets, "err": pred != df.label,
                      "fp": (pred == 1) & (df.label == 0), "fn": (pred == 0) & (df.label == 1)})
    t = g.groupby("bucket", observed=False).agg(messages=("err", "size"), error_rate=("err", "mean"),
                                                FP=("fp", "sum"), FN=("fn", "sum"))
    t["model"] = name
    return t.reset_index()


# -------------------------------------------------------------------- main
def fmt_table(rows, cols):
    head = "| " + " | ".join(c for c, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(f(r) for _, f in cols) + " |\n" for r in rows)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    yv, yt = val.label.values, test.label.values

    results = {}
    for name in available_models():
        model, vocab = load(name)
        pv, _ = score(name, model, vocab, val.text)
        pt, expl = score(name, model, vocab, test.text, top_k=6)
        thr = cost_threshold(yv, pv)
        results[name] = {"p": pt, "expl": expl, "thr_cost": thr,
                         "m05": metrics_at(yt, pt, 0.5), "mcost": metrics_at(yt, pt, thr)}
        print(f"{MODEL_NAMES[name]:38s} test F1@0.5={results[name]['m05']['f1']:.4f}  "
              f"FP={results[name]['m05']['FP']}  FN={results[name]['m05']['FN']}  | "
              f"cost thr={thr:.2f}: FP={results[name]['mcost']['FP']} FN={results[name]['mcost']['FN']}")

    # Ensemble = average probability of all models (reported in tables only)
    ens_p = np.mean([r["p"] for r in results.values()], axis=0)
    ens_pv = np.mean([score(n, *load(n), val.text)[0] for n in results], axis=0)
    ens_thr = cost_threshold(yv, ens_pv)
    ensemble = {"m05": metrics_at(yt, ens_p, 0.5), "mcost": metrics_at(yt, ens_p, ens_thr)}

    plot_confusion(results, yt)
    plot_curves(results, yt)
    plot_threshold_tradeoff(results, yt)
    plot_error_bars(results)
    deep = [n for n in results if n != "baseline"]
    best_deep = max(deep, key=lambda n: results[n]["m05"]["f1"]) if deep else "baseline"
    plot_probability_hist(results, yt, best_deep)

    profiles, attn, buckets = {}, {}, []
    error_sets = {}
    for name, r in results.items():
        df, fp, fn = save_errors(name, test, r)
        profiles[name] = error_profile(df, fp, fn)
        attn[name] = attention_words(fp, fn)
        buckets.append(length_buckets(df, name))
        error_sets[name] = set(fp.index) | set(fn.index)
    buckets = pd.concat(buckets)

    names = list(results)
    shared_all = set.intersection(*error_sets.values()) if error_sets else set()
    overlap = {"errors_shared_by_all_models": len(shared_all),
               "errors_per_model": {n: len(s) for n, s in error_sets.items()}}

    out = {n: {"threshold_0.5": r["m05"], "cost_tuned": r["mcost"]} for n, r in results.items()}
    out["ensemble"] = {"threshold_0.5": ensemble["m05"], "cost_tuned": ensemble["mcost"]}
    out["_error_profiles"] = profiles
    out["_attention_words"] = attn
    out["_error_overlap"] = overlap
    out["_cost_weights"] = {"COST_FN": COST_FN, "COST_FP": COST_FP}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # Markdown summary
    pct = lambda k: (lambda r: f"{r[k]:.2%}")  # noqa: E731
    num = lambda k, d=4: (lambda r: f"{r[k]:.{d}f}")  # noqa: E731
    cols = [("Model", lambda r: r["name"]), ("Thr", num("threshold", 2)), ("Accuracy", pct("accuracy")),
            ("Precision", pct("precision")), ("Recall", pct("recall")), ("F1", num("f1")),
            ("ROC-AUC", num("roc_auc")), ("PR-AUC", num("pr_auc")), ("MCC", num("mcc")),
            ("FP", lambda r: str(r["FP"])), ("FN", lambda r: str(r["FN"])),
            ("FPR", pct("false_positive_rate")), ("FNR", pct("false_negative_rate"))]
    all_names = {**MODEL_NAMES, "ensemble": "Ensemble (average of all)"}
    rows05 = [{"name": all_names[n], **out[n]["threshold_0.5"]} for n in names + ["ensemble"]]
    rowsc = [{"name": all_names[n], **out[n]["cost_tuned"]} for n in names + ["ensemble"]]
    md = ["# Results summary (test set, %d messages: %d safe, %d phishing)\n"
          % (len(yt), (yt == 0).sum(), (yt == 1).sum()),
          "## Default threshold 0.5\n", fmt_table(rows05, cols),
          f"\n## Cost-tuned threshold (chosen on validation, FN cost = {COST_FN:g} x FP cost)\n",
          fmt_table(rowsc, cols),
          "\n## Error profile (threshold 0.5)\n"]
    for n in names:
        p = profiles[n]
        md.append(f"**{MODEL_NAMES[n]}**\n")
        md.append(fmt_table(
            [{"g": g, **p[g]} for g in p],
            [("Group", lambda r: r["g"].replace("_", " ")), ("Count", lambda r: str(r["count"])),
             ("Median tokens", lambda r: f"{r['median_tokens']:.0f}"),
             ("Contains URL", lambda r: f"{r['share_with_url']:.1%}"),
             ("Mean confidence", lambda r: f"{r['mean_confidence_wrong']:.2f}")]))
        md.append("")
    md.append("## Error rate by message length (threshold 0.5)\n")
    piv = buckets.pivot_table(index="bucket", columns="model", values="error_rate", observed=False)
    md.append(fmt_table([{"b": b, **row} for b, row in piv.iterrows()],
                        [("Length (tokens)", lambda r: str(r["b"]))]
                        + [(n, (lambda n: lambda r: f"{r[n]:.2%}")(n)) for n in piv.columns]))
    md.append("\n\nMessages per bucket: " + ", ".join(
        f"{b}: {m}" for b, m in buckets[buckets.model == names[0]][["bucket", "messages"]].values))
    md.append(f"\n\n## Error overlap\n\n{overlap['errors_shared_by_all_models']} test messages are "
              f"misclassified by **every** model (errors per model: "
              + ", ".join(f"{n} {c}" for n, c in overlap["errors_per_model"].items()) + ").\n")
    for n, a in attn.items():
        for k, words in a.items():
            md.append(f"\n**{MODEL_NAMES[n]} - most-attended words in {k.replace('_', ' ')}:** "
                      + ", ".join(f"{w} ({c})" for w, c in words) + "\n")
    (RESULTS_DIR / "results_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {RESULTS_DIR / 'metrics.json'} and {RESULTS_DIR / 'results_summary.md'}")
    print(f"Error examples in {ERRORS_DIR}")


if __name__ == "__main__":
    main()
