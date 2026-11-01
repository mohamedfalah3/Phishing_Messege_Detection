import html

import pandas as pd
from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QHeaderView, QLineEdit, QPushButton,
                               QSplitter, QTableView, QTextBrowser, QVBoxLayout, QWidget)

import app_core
from pages.base import Page
from widgets import DataFrameModel, ProbCard, attention_html, hbox, label, page_header, prob, verdict_color


class ErrorsPage(Page):
    def __init__(self, state):
        super().__init__(state)
        page_header(self.lay, "Error explorer",
                    "Every test message the chosen model gets wrong at the current threshold. "
                    "Select a row to read the full message and see what the BiLSTM focused on.")
        self.loading = label("Scoring the test set with every model (first run only, about a minute)…",
                             "caption")
        self.lay.addWidget(self.loading)

        self.model = QComboBox()
        for n in state.names:
            self.model.addItem(app_core.LONG_NAMES[n], n)
        self.model.setCurrentIndex(len(state.names) - 1)
        self.model.currentIndexChanged.connect(self.render)
        self.fp_btn, self.fn_btn = QPushButton("False positives"), QPushButton("False negatives")
        group = QButtonGroup(self)
        for b in (self.fp_btn, self.fn_btn):
            b.setCheckable(True)
            group.addButton(b)
        self.fp_btn.setChecked(True)
        group.buttonClicked.connect(self.render)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search in messages (e.g. order, password, http)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.render)
        self.all_models = QCheckBox("Only errors every model makes")
        self.all_models.toggled.connect(self.render)
        self.lay.addWidget(hbox(label("Model:", wrap=False), self.model, self.fp_btn, self.fn_btn,
                                self.search, self.all_models))
        self.counts = label("", "caption")
        self.lay.addWidget(self.counts)

        split = QSplitter(Qt.Vertical)
        self.table_model = DataFrameModel(formats={"P(phishing)": prob}, colors={"true label": verdict_color})
        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(self.table_model)
        proxy.setSortRole(Qt.UserRole)
        self.table = QTableView()
        self.table.setModel(proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.selectionModel().currentRowChanged.connect(self.show_row)
        split.addWidget(self.table)

        detail = QWidget()
        d = QVBoxLayout(detail)
        d.setContentsMargins(0, 8, 0, 0)
        self.detail_title = label("Select a row to inspect the message.", "section")
        d.addWidget(self.detail_title)
        self.cards = {n: ProbCard(n) for n in state.names}
        d.addWidget(hbox(*self.cards.values()))
        self.full_text = QTextBrowser()
        self.full_text.setMinimumHeight(110)
        d.addWidget(self.full_text)
        d.addWidget(label("<b>BiLSTM attention</b> (darker = more important)"))
        self.attn = QTextBrowser()
        self.attn.setMinimumHeight(110)
        d.addWidget(self.attn)
        split.addWidget(detail)
        split.setSizes([330, 470])
        split.setMinimumHeight(820)
        self.lay.addWidget(split)
        self.finish()

        self.errs = pd.DataFrame()
        self.setEnabled(False)
        state.thresholdChanged.connect(self.render)
        state.testReady.connect(self.on_ready)

    def on_ready(self):
        self.loading.hide()
        self.setEnabled(True)
        self.render()

    def render(self, *_):
        df = self.state.test_df
        if df is None:
            return
        name, thr = self.model.currentData(), self.state.threshold
        pred = df[name] >= thr
        fp, fn = pred & (df.label == 0), ~pred & (df.label == 1)
        if self.all_models.isChecked():
            for n in self.state.names:
                p = df[n] >= thr
                fp &= p & (df.label == 0)
                fn &= ~p & (df.label == 1)
        self.counts.setText(f"<b>{int(fp.sum())}</b> false positives (safe → flagged) · "
                            f"<b>{int(fn.sum())}</b> false negatives (phishing → missed)")
        want_fp = self.fp_btn.isChecked()
        errs = df[fp if want_fp else fn]
        query = self.search.text().strip()
        if query:
            errs = errs[errs.text.str.contains(query, case=False, regex=False)]
        self.errs = errs.sort_values(name, ascending=not want_fp).reset_index(drop=True)
        table = pd.DataFrame({"P(phishing)": self.errs[name].values,
                              "true label": ["phishing" if v else "safe" for v in self.errs.label],
                              "message": self.errs.text.str.slice(0, 300).str.replace(r"\s+", " ", regex=True)
                              .values})
        self.table_model.set_frame(table)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        if len(table):
            self.table.selectRow(0)
        else:
            self.detail_title.setText("No errors of this type for this selection.")
            self.full_text.clear()
            self.attn.clear()

    def show_row(self, current, _previous):
        if not current.isValid() or self.errs.empty:
            return
        src = self.table.model().mapToSource(current).row()
        row = self.errs.iloc[src]
        name = self.model.currentData()
        self.detail_title.setText(f"True label: {'phishing' if row.label else 'safe'} · "
                                  f"{app_core.SHORT_NAMES[name]} says P(phishing) = {row[name]:.3f}")
        for n, card in self.cards.items():
            card.set(float(row[n]), self.state.threshold)
        self.full_text.setHtml(f"<p>{html.escape(row.text[:5000])}</p>")
        self.attn.setHtml(attention_html(app_core.attention(self.state.models, row.text)))
