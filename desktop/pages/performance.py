import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PySide6.QtWidgets import QComboBox, QHeaderView, QPushButton, QTableView, QWidget, QHBoxLayout

import app_core
from config import COST_FN, COST_FP
from pages.base import Page
from theme import BLUE, MUTED, NAVY, ORANGE
from widgets import Canvas, DataFrameModel, Tile, hbox, label, page_header, pct

BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


class PerformancePage(Page):
    def __init__(self, state):
        super().__init__(state)
        page_header(self.lay, "Model performance",
                    "Live results on the held-out test set (2,627 messages) at the threshold in the sidebar.")
        self.loading = label("Scoring the test set with every model (first run only, about a minute)…",
                             "caption")
        self.lay.addWidget(self.loading)

        self.model = QComboBox()
        for n in state.names:
            self.model.addItem(app_core.LONG_NAMES[n], n)
        self.model.setCurrentIndex(len(state.names) - 1)
        self.model.currentIndexChanged.connect(self.render)
        reset = QPushButton("Threshold 0.50")
        reset.clicked.connect(lambda: state.set_threshold(0.5))
        self.cost_btn = QPushButton()
        self.cost_btn.setToolTip(f"Chosen on the validation set to minimise {COST_FN:g}×FN + {COST_FP:g}×FP")
        self.cost_btn.clicked.connect(self.use_cost_threshold)
        self.lay.addWidget(hbox(label("Model:", wrap=False), self.model, reset, self.cost_btn, stretch_last=True))

        names = ["Accuracy", "Precision", "Recall", "F1", "False positives", "False negatives"]
        self.tiles = {n: Tile(n) for n in names}
        self.tiles["False positives"].setToolTip("Safe messages flagged (false positive rate)")
        self.tiles["False negatives"].setToolTip("Phishing messages missed (false negative rate)")
        self.lay.addWidget(hbox(*self.tiles.values()))

        charts = QWidget()
        row = QHBoxLayout(charts)
        row.setContentsMargins(0, 0, 0, 0)
        self.cm = Canvas(height=3.5, width=4.2)
        self.sweep = Canvas(height=3.8, width=6.5)
        row.addWidget(self.cm, 2)
        row.addWidget(self.sweep, 3)
        self.lay.addWidget(charts)

        self.compare_title = label("", "section")
        self.lay.addWidget(self.compare_title)
        self.table_model = DataFrameModel(formats={"Accuracy": pct, "Precision": pct, "Recall": pct,
                                                   "F1": lambda v: f"{v:.4f}"})
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.verticalHeader().hide()
        self.table.setFixedHeight(40 + 32 * len(state.names))
        self.lay.addWidget(self.table)
        self.finish()

        self.setEnabled(False)
        state.thresholdChanged.connect(self.render)
        state.testReady.connect(self.on_ready)

    def on_ready(self):
        self.loading.hide()
        self.setEnabled(True)
        self.render()

    def use_cost_threshold(self):
        thr = app_core.cost_threshold(self.model.currentData())
        if thr is not None:
            self.state.set_threshold(thr)

    def render(self, *_):
        df = self.state.test_df
        if df is None:
            return
        name, thr = self.model.currentData(), self.state.threshold
        cost = app_core.cost_threshold(name)
        self.cost_btn.setText(f"Cost-tuned threshold ({cost:.2f})" if cost is not None else "Cost-tuned threshold")
        self.cost_btn.setEnabled(cost is not None)
        y = df.label.values
        c = app_core.confusion(y, df[name].values, thr)
        self.tiles["Accuracy"].set(pct(c["accuracy"]))
        self.tiles["Precision"].set(pct(c["precision"]))
        self.tiles["Recall"].set(pct(c["recall"]))
        self.tiles["F1"].set(f"{c['f1']:.4f}")
        self.tiles["False positives"].set(f"{c['FP']} ({c['fpr']:.1%})", BLUE)
        self.tiles["False negatives"].set(f"{c['FN']} ({c['fnr']:.1%})", ORANGE)
        self.draw_confusion(c, thr)
        self.draw_sweep(app_core.threshold_sweep(y, df[name].values), thr)

        self.compare_title.setText(f"All models at threshold {thr:.2f}")
        rows = []
        for n in self.state.names:
            m = app_core.confusion(y, df[n].values, thr)
            rows.append({"Model": app_core.LONG_NAMES[n], "Accuracy": m["accuracy"], "Precision": m["precision"],
                         "Recall": m["recall"], "F1": m["f1"], "False positives": m["FP"],
                         "False negatives": m["FN"]})
        self.table_model.set_frame(pd.DataFrame(rows))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)

    def draw_confusion(self, c, thr):
        ax = self.cm.axes()
        cm = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
        names = [["TN", "FP (false alarm)"], ["FN (missed)", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}\n({share[i, j]:.1%})", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white" if share[i, j] > 0.5 else NAVY)
        ax.set_xticks([0, 1], ["Safe", "Phishing"])
        ax.set_yticks([0, 1], ["Safe", "Phishing"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Confusion matrix · threshold {thr:.2f}", loc="left", fontsize=10, fontweight="bold")
        self.cm.draw()

    def draw_sweep(self, sweep, thr):
        ax = self.sweep.axes()
        ax.plot(sweep.threshold, sweep["False positives"], color=BLUE, lw=2.2, label="False positives (safe flagged)")
        ax.plot(sweep.threshold, sweep["False negatives"], color=ORANGE, lw=2.2,
                label="False negatives (phishing missed)")
        ax.axvline(thr, color=MUTED, ls="--", lw=1.2)
        ax.text(thr + 0.01, 140, f"current {thr:.2f}", color=MUTED, fontsize=9)
        ax.set_ylim(0, 150)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Errors on test set (capped at 150)")
        ax.grid(color="#edf0f5")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax.set_title("False positive / false negative trade-off", loc="left", fontsize=10, fontweight="bold")
        self.sweep.draw()
