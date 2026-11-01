from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox, QPushButton,
                               QSlider, QStackedWidget, QVBoxLayout, QWidget)

from pages.analyze import AnalyzePage
from pages.batch import BatchPage
from pages.errors import ErrorsPage
from pages.overview import OverviewPage
from pages.performance import PerformancePage


class MainWindow(QMainWindow):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setWindowTitle("Phishing Message Detector")
        self.resize(1380, 900)
        self.setMinimumSize(1100, 700)

        pages = [("Overview", OverviewPage), ("Analyze a message", AnalyzePage), ("Batch scan", BatchPage),
                 ("Model performance", PerformancePage), ("Error explorer", ErrorsPage)]
        self.stack = QStackedWidget()
        self.stack.setObjectName("content")
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        for title, cls in pages:
            self.stack.addWidget(cls(state))
            self.nav.addItem(title)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.sidebar())
        row.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        for i in range(len(pages)):
            act = QAction(self)
            act.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            act.triggered.connect(lambda _=False, i=i: self.nav.setCurrentRow(i))
            self.addAction(act)

    def sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(250)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(14, 22, 14, 18)
        lay.setSpacing(6)
        title = QLabel("Phishing Detector")
        title.setObjectName("appTitle")
        sub = QLabel("Deep learning text classification")
        sub.setObjectName("appSubtitle")
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(14)
        self.nav.setFixedHeight(5 * 46)
        lay.addWidget(self.nav)
        lay.addStretch(1)

        heading = QLabel("<b>Decision threshold</b>")
        self.thr_value = QLabel()
        self.thr_value.setAlignment(Qt.AlignRight)
        lay.addWidget(self._row(heading, self.thr_value))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 95)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(lambda v: self.state.set_threshold(v / 100))
        lay.addWidget(self.slider)
        cap = QLabel("Messages at or above this probability are flagged. Lower → fewer missed phishing, "
                     "more false alarms. Applies to every page.")
        cap.setObjectName("sideCaption")
        cap.setWordWrap(True)
        lay.addWidget(cap)
        reset = QPushButton("Reset to 0.50")
        reset.clicked.connect(lambda: self.state.set_threshold(0.5))
        lay.addWidget(reset)
        lay.addSpacing(10)
        about = QPushButton("About")
        about.clicked.connect(self.about)
        lay.addWidget(about)

        self.state.thresholdChanged.connect(self.sync_threshold)
        self.sync_threshold(self.state.threshold)
        return side

    @staticmethod
    def _row(left, right):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(left)
        h.addWidget(right)
        return w

    def sync_threshold(self, value):
        self.thr_value.setText(f"<b>{value:.2f}</b>")
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(value * 100)))
        self.slider.blockSignals(False)

    def about(self):
        QMessageBox.about(self, "About",
                          "<h3>Phishing Message Detector</h3>"
                          "<p>AI course project: phishing message detection with deep learning.</p>"
                          "<p>Models: TF-IDF + Logistic Regression (baseline), TextCNN and BiLSTM + Attention "
                          "(PyTorch), and their averaged ensemble.</p>"
                          "<p>Shortcuts: Ctrl+1…5 switch pages · Ctrl+Enter analyzes a message.</p>")
