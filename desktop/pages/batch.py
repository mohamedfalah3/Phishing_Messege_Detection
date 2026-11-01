from pathlib import Path

import pandas as pd
from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHeaderView, QMessageBox, QPushButton, QSpinBox,
                               QTableView)

import app_core
from config import PROCESSED_DIR
from pages.base import Page
from state import run_in_background
from theme import ORANGE
from widgets import DataFrameModel, Tile, hbox, label, page_header, pct, prob, verdict_color

MAX_ROWS = 20000


class BatchPage(Page):
    def __init__(self, state):
        super().__init__(state)
        page_header(self.lay, "Batch scan",
                    "Classify many messages at once: a CSV (one message per row) or a TXT file (one message "
                    "per line). If the CSV has a 'label' column (1/0 or phishing/safe), accuracy and errors "
                    "are shown too.")

        open_btn = QPushButton("Open CSV / TXT…")
        open_btn.setObjectName("primary")
        open_btn.clicked.connect(self.open_file)
        self.sample_size = QSpinBox()
        self.sample_size.setRange(50, 2627)
        self.sample_size.setSingleStep(50)
        self.sample_size.setValue(200)
        sample_btn = QPushButton("Scan a test-set sample")
        sample_btn.clicked.connect(self.sample)
        self.export_btn = QPushButton("Export results…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export)
        self.lay.addWidget(hbox(open_btn, label("or", wrap=False), self.sample_size, sample_btn, None,
                                self.export_btn))

        self.column = QComboBox()
        self.column.setMinimumWidth(160)
        self.column.activated.connect(lambda _: self.score())
        self.decider = QComboBox()
        for n in state.names:
            self.decider.addItem(app_core.LONG_NAMES[n], n)
        self.decider.setCurrentIndex(len(state.names) - 1)
        self.decider.currentIndexChanged.connect(self.render)
        self.view = QComboBox()
        self.view.addItems(["All messages", "Flagged as phishing", "Errors only"])
        self.view.currentIndexChanged.connect(self.render)
        self.lay.addWidget(hbox(label("Text column:", wrap=False), self.column, label("Verdict from:", wrap=False),
                                self.decider, label("Show:", wrap=False), self.view, stretch_last=True))

        self.status = label("No messages loaded yet.", "caption")
        self.lay.addWidget(self.status)
        self.tiles = [Tile("Messages scanned"), Tile("Flagged as phishing"), Tile("Accuracy"),
                      Tile("Errors (FP · FN)", "False positives · false negatives")]
        self.lay.addWidget(hbox(*self.tiles))

        self.model = DataFrameModel(formats={}, colors={"verdict": verdict_color, "true label": verdict_color})
        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(self.model)
        proxy.setSortRole(Qt.UserRole)
        self.table = QTableView()
        self.table.setModel(proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(380)
        self.lay.addWidget(self.table)
        self.finish()

        self.df, self.probs, self.labels, self.out = None, None, None, None
        state.thresholdChanged.connect(self.render)

    # ---------------------------------------------------------------- loading
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open messages", "", "Messages (*.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            if path.lower().endswith(".txt"):
                lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
                df = pd.DataFrame({"text": [line.strip() for line in lines if line.strip()]})
            else:
                df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not read file", str(exc))
            return
        self.load(df, Path(path).name)

    def sample(self):
        df = pd.read_csv(PROCESSED_DIR / "test.csv").sample(self.sample_size.value(), random_state=1)
        self.load(df.reset_index(drop=True), "test-set sample")

    def load(self, df, source):
        if df.empty:
            QMessageBox.information(self, "Empty file", "The file contains no messages.")
            return
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
        self.df, self.source = df, source
        text_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
        self.column.clear()
        self.column.addItems([str(c) for c in text_cols])
        if "text" in text_cols:
            self.column.setCurrentText("text")
        self.labels = app_core.parse_labels(df["label"]) if "label" in df.columns else None
        self.score()

    def score(self):
        if self.df is None or not self.column.currentText():
            return
        texts = self.df[self.column.currentText()].fillna("").astype(str).tolist()
        self.status.setText(f"Scanning {len(texts):,} messages from {self.source}…")
        self.setEnabled(False)
        run_in_background(lambda: app_core.predict(self.state.models, texts), self.scored, self.failed)

    def scored(self, probs):
        self.probs = probs
        self.setEnabled(True)
        self.status.setText(f"{len(probs):,} messages from {self.source}. Click a column header to sort.")
        self.export_btn.setEnabled(True)
        self.render()

    def failed(self, message):
        self.setEnabled(True)
        self.status.setText("Scan failed.")
        QMessageBox.warning(self, "Scan failed", message)

    # -------------------------------------------------------------- rendering
    def render(self, *_):
        if self.probs is None:
            return
        thr = self.state.threshold
        decider = self.decider.currentData()
        out = pd.DataFrame({"verdict": ["phishing" if p >= thr else "safe" for p in self.probs[decider]]})
        if self.labels is not None:
            out["true label"] = ["phishing" if v else "safe" for v in self.labels]
            out["correct"] = (out["verdict"] == out["true label"]).map({True: "yes", False: "NO"})
        for n in self.state.names:
            out[app_core.SHORT_NAMES[n]] = self.probs[n].values
        out["message"] = self.df[self.column.currentText()].astype(str).str.replace(r"\s+", " ", regex=True).values
        self.out = out

        flagged = int((out.verdict == "phishing").sum())
        self.tiles[0].set(f"{len(out):,}")
        self.tiles[1].set(f"{flagged:,} ({flagged / len(out):.1%})", ORANGE if flagged else None)
        if self.labels is not None:
            c = app_core.confusion(self.labels, self.probs[decider].values, thr)
            self.tiles[2].set(pct(c["accuracy"]))
            self.tiles[3].set(f"{c['FP']} · {c['FN']}")
        else:
            self.tiles[2].set("no labels")
            self.tiles[3].set("—")

        shown = out
        if self.view.currentIndex() == 1:
            shown = out[out.verdict == "phishing"]
        elif self.view.currentIndex() == 2:
            shown = out[out["correct"] == "NO"] if "correct" in out else out.iloc[0:0]
        self.model.formats = {app_core.SHORT_NAMES[n]: prob for n in self.state.names}
        self.model.set_frame(shown)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(shown.columns) - 1, QHeaderView.Stretch)

    def export(self):
        if self.out is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", "phishing_scan_results.csv", "CSV (*.csv)")
        if path:
            self.out.to_csv(path, index=False)
            self.status.setText(f"Saved {len(self.out):,} rows to {path}")
