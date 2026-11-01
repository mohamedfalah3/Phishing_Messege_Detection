import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHeaderView, QLabel, QScrollArea, QTabWidget, QTableView, QVBoxLayout

import app_core
from config import FIGURES_DIR
from pages.base import Page
from theme import BLUE, ORANGE
from widgets import DataFrameModel, Tile, hbox, label, page_header, pct


def card(title, body):
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 12, 16, 14)
    lay.addWidget(label(f"<b>{title}</b>"))
    lay.addWidget(label(body))
    return frame


class OverviewPage(Page):
    def __init__(self, state):
        super().__init__(state)
        page_header(self.lay, "Phishing Message Detection with Deep Learning",
                    "Text classifiers trained on 17,511 labelled e-mails estimate how likely a message is to be "
                    "phishing. Two deep learning models (TextCNN, BiLSTM + Attention) are compared with a "
                    "TF-IDF + Logistic Regression baseline, with a focus on false positives and false negatives.")

        m = state.metrics
        best = m.get("ensemble", m.get("baseline", {})).get("threshold_0.5")
        if best:
            self.lay.addWidget(label("Headline results on the test set", "section"))
            self.lay.addWidget(label("2,627 messages never seen during training · threshold 0.5 · ensemble of "
                                     "all models", "caption"))
            tiles = [Tile("Accuracy"), Tile("F1 score"), Tile("ROC-AUC"),
                     Tile("False positives", "Safe messages wrongly flagged"),
                     Tile("False negatives", "Phishing messages missed")]
            tiles[0].set(pct(best["accuracy"]))
            tiles[1].set(f"{best['f1']:.3f}")
            tiles[2].set(f"{best['roc_auc']:.4f}")
            tiles[3].set(best["FP"], BLUE)
            tiles[4].set(best["FN"], ORANGE)
            self.lay.addWidget(hbox(*tiles))

            rows = [{"Model": app_core.LONG_NAMES[n], "Accuracy": r["accuracy"], "Precision": r["precision"],
                     "Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"], "False positives": r["FP"],
                     "False negatives": r["FN"]}
                    for n, v in m.items() if not n.startswith("_") for r in [v["threshold_0.5"]]]
            model = DataFrameModel(pd.DataFrame(rows), formats={
                "Accuracy": pct, "Precision": pct, "Recall": pct, "F1": lambda v: f"{v:.4f}",
                "ROC-AUC": lambda v: f"{v:.4f}"})
            table = QTableView()
            table.setModel(model)
            table.verticalHeader().hide()
            table.setFixedHeight(40 + 32 * len(rows))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.lay.addWidget(table)
        else:
            self.lay.addWidget(label("No results found. Run the training pipeline first: python run_all.py"))

        self.lay.addWidget(label("How it works", "section"))
        self.lay.addWidget(hbox(
            card("1. Data", "Phishing Email Dataset (18,650 e-mails). Empty rows and 577 duplicates removed → "
                            "17,511 messages, 37% phishing. Split 70 / 15 / 15 (train / validation / test)."),
            card("2. Preprocessing", "Lower-case; links, e-mails, money and numbers replaced by placeholder "
                                     "tokens; 30,000-word vocabulary from training data; first 300 tokens."),
            card("3. Models", "TF-IDF + Logistic Regression (baseline) · TextCNN (convolutions over 3-5 words) · "
                              "BiLSTM + Attention (explainable). Ensemble = average of all three."),
            card("4. Evaluation", "Test-set accuracy, precision, recall, F1, ROC-AUC; false positive / false "
                                  "negative analysis; cost-tuned threshold (a miss costs 3× a false alarm).")))

        self.lay.addWidget(label("Saved evaluation figures", "section"))
        tabs = QTabWidget()
        for title, files in [("Dataset", ["eda_overview.png"]), ("Training", ["training_curves_cnn.png",
                             "training_curves_bilstm.png"]), ("Confusion matrices", ["confusion_matrices.png"]),
                             ("ROC && PR", ["roc_pr_curves.png"]), ("Errors", ["error_counts.png",
                             "threshold_tradeoff.png"])]:
            tabs.addTab(self.figure_tab(files), title)
        tabs.setMinimumHeight(520)
        self.lay.addWidget(tabs)
        self.finish()

    @staticmethod
    def figure_tab(files):
        area = QScrollArea()
        area.setWidgetResizable(True)
        body = QFrame()
        lay = QVBoxLayout(body)
        for f in files:
            path = FIGURES_DIR / f
            if path.exists():
                img = QLabel()
                img.setPixmap(QPixmap(str(path)).scaledToWidth(1050, Qt.SmoothTransformation))
                img.setAlignment(Qt.AlignHCenter)
                lay.addWidget(img)
        lay.addStretch(1)
        area.setWidget(body)
        return area
