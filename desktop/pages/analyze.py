from email import policy
from email.parser import BytesParser
from pathlib import Path

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QPlainTextEdit, QPushButton, QTextBrowser,
                               QWidget)

import app_core
from pages.base import Page
from theme import ORANGE
from widgets import (Canvas, ProbCard, VerdictBanner, attention_html, hbox, label, page_header,
                     signals_html)


def read_message_file(path):
    """Text of a .txt file, or subject + body of an .eml e-mail."""
    path = Path(path)
    if path.suffix.lower() == ".eml":
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        body = msg.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body else ""
        return f"{msg.get('subject', '')}\n\n{text}".strip()
    return path.read_text(encoding="utf-8", errors="ignore")


class AnalyzePage(Page):
    def __init__(self, state):
        super().__init__(state)
        page_header(self.lay, "Analyze a message",
                    "Paste an e-mail or SMS, open a .txt / .eml file, or pick an example. "
                    "Press Ctrl+Enter to analyze.")

        self.examples = QComboBox()
        self.examples.addItem("Example messages…")
        self.examples.addItems(list(app_core.EXAMPLES))
        self.examples.currentTextChanged.connect(self.load_example)
        self.decider = QComboBox()
        for n in state.names:
            self.decider.addItem(app_core.LONG_NAMES[n], n)
        self.decider.setCurrentIndex(len(state.names) - 1)
        self.decider.currentIndexChanged.connect(self.refresh_verdicts)
        open_btn = QPushButton("Open file…")
        open_btn.clicked.connect(self.open_file)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        run = QPushButton("Analyze")
        run.setObjectName("primary")
        run.clicked.connect(self.analyze)
        self.lay.addWidget(hbox(self.examples, label("Verdict from:", wrap=False), self.decider, None,
                                open_btn, clear_btn, run))

        self.edit = QPlainTextEdit()
        self.edit.setPlaceholderText("e.g. Your account has been suspended. Verify now at http://…")
        self.edit.setMinimumHeight(150)
        self.lay.addWidget(self.edit)
        QShortcut(QKeySequence("Ctrl+Return"), self, self.analyze)
        QShortcut(QKeySequence("Ctrl+Enter"), self, self.analyze)

        self.hint = label("Enter a message above and press Analyze.", "caption")
        self.lay.addWidget(self.hint)

        self.results = QWidget()
        r = self.results_layout = QGridLayout(self.results)
        r.setContentsMargins(0, 0, 0, 0)
        r.setVerticalSpacing(10)
        self.banner = VerdictBanner()
        r.addWidget(self.banner, 0, 0, 1, 2)
        r.addWidget(label("Every model's opinion", "section"), 1, 0, 1, 2)
        self.cards = {n: ProbCard(n) for n in state.names}
        r.addWidget(hbox(*self.cards.values()), 2, 0, 1, 2)
        r.addWidget(label("Why? What the BiLSTM paid attention to", "section"), 3, 0, 1, 2)
        self.attn = QTextBrowser()
        self.attn.setMinimumHeight(170)
        self.words = Canvas(height=2.6, width=3.6)
        self.words.setMinimumWidth(320)
        r.addWidget(self.attn, 4, 0)
        r.addWidget(self.words, 4, 1)
        r.setColumnStretch(0, 3)
        r.setColumnStretch(1, 2)
        r.addWidget(label("Darker = more attention. Words are shown after preprocessing: links → urltoken, "
                          "numbers → numtoken, money → moneytoken, e-mail addresses → emailtoken.", "caption"),
                    5, 0, 1, 2)
        r.addWidget(label("Common phishing signals in the text", "section"), 6, 0, 1, 2)
        self.signals = QTextBrowser()
        self.signals.setFixedHeight(150)
        r.addWidget(self.signals, 7, 0, 1, 2)
        r.addWidget(label("These rule-based signals help a human read the message. The models do not use them; "
                          "they learned their own patterns from data.", "caption"), 8, 0, 1, 2)
        self.results.hide()
        self.lay.addWidget(self.results)
        self.finish()

        self.probs = None
        state.thresholdChanged.connect(self.refresh_verdicts)

    def load_example(self, name):
        if name in app_core.EXAMPLES:
            self.edit.setPlainText(app_core.EXAMPLES[name])
            self.analyze()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open a message", "",
                                              "Messages (*.txt *.eml);;All files (*)")
        if path:
            self.edit.setPlainText(read_message_file(path))
            self.analyze()

    def clear(self):
        self.edit.clear()
        self.examples.setCurrentIndex(0)
        self.results.hide()
        self.hint.show()

    def analyze(self):
        text = self.edit.toPlainText().strip()
        if not text:
            self.hint.setText("Enter a message above and press Analyze.")
            return
        models = self.state.models
        self.probs = app_core.predict(models, [text]).iloc[0].to_dict()
        pairs = app_core.attention(models, text)
        self.attn.setHtml(attention_html(pairs) + ("<p><i>Long message: the deep models only read the first "
                                                   "300 tokens.</i></p>" if pairs and len(pairs) >= 300 else ""))
        self.draw_words(pairs)
        self.signals.setHtml(signals_html(app_core.signals(text)))
        self.refresh_verdicts()
        self.hint.hide()
        self.results.show()

    def draw_words(self, pairs):
        ax = self.words.axes()
        if pairs:
            weights = {}
            for tok, w in pairs:
                weights[tok] = weights.get(tok, 0) + float(w)
            top = sorted(weights.items(), key=lambda kv: kv[1])[-10:]
            ax.barh([t for t, _ in top], [w for _, w in top], color=ORANGE, height=0.65)
            ax.set_xlabel("Attention weight")
            ax.grid(axis="x", color="#edf0f5")
            ax.set_axisbelow(True)
        ax.set_title("Most-attended words", loc="left", fontsize=10, fontweight="bold")
        self.words.draw()

    def refresh_verdicts(self, *_):
        if self.probs is None:
            return
        thr = self.state.threshold
        decider = self.decider.currentData()
        self.banner.set(self.probs[decider], thr, app_core.LONG_NAMES[decider])
        for n, card in self.cards.items():
            card.set(self.probs[n], thr)
