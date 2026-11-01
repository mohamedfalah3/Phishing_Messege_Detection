"""Reusable widgets: page headers, metric tiles, probability cards, verdict
banner, matplotlib canvases and a pandas table model."""
import html
import warnings

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget

from app_core import SHORT_NAMES
from theme import BLUE, BORDER, MUTED, NAVY, ORANGE

# Charts on screens that are not visible yet are drawn at zero size once.
warnings.filterwarnings("ignore", message="constrained_layout not applied")


def label(text, name=None, wrap=True):
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lab


def page_header(layout, title, caption):
    layout.addWidget(label(title, "pageTitle"))
    layout.addWidget(label(caption, "caption"))


def hbox(*widgets, stretch_last=False, spacing=10):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for item in widgets:
        if item is None:
            lay.addStretch(1)
        else:
            lay.addWidget(item)
    if stretch_last:
        lay.addStretch(1)
    return w


class Tile(QFrame):
    """Bordered box with a small label and a big value."""

    def __init__(self, title, tooltip=None):
        super().__init__()
        self.setObjectName("tile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.addWidget(label(title, "tileLabel"))
        self.value = label("—", "tileValue", wrap=False)
        lay.addWidget(self.value)
        if tooltip:
            self.setToolTip(tooltip)

    def set(self, text, color=None):
        self.value.setText(str(text))
        self.value.setStyleSheet(f"color: {color};" if color else "")


def verdict_tag(kind):
    color = ORANGE if kind == "phishing" else BLUE
    return (f'<span style="background-color:{color}; color:white; font-weight:600;">'
            f'&nbsp;{kind.upper()}&nbsp;</span>')


class ProbCard(QFrame):
    """One model's probability, verdict tag and bar."""

    def __init__(self, name):
        super().__init__()
        self.setObjectName("tile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.addWidget(label(f"<b>{SHORT_NAMES[name]}</b>", wrap=False))
        self.value = label("—", "tileValue", wrap=False)
        self.tag = QLabel()
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        for w in (self.value, self.tag, self.bar):
            lay.addWidget(w)

    def set(self, p, thr):
        kind = "phishing" if p >= thr else "safe"
        self.value.setText(f"{p:.3f}")
        self.tag.setText(verdict_tag(kind))
        self.bar.setValue(int(round(p * 1000)))


class VerdictBanner(QLabel):
    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self.setMinimumHeight(74)

    def set(self, p, thr, model_name):
        phishing = p >= thr
        color = ORANGE if phishing else BLUE
        title = "Likely PHISHING" if phishing else "Looks SAFE"
        self.setStyleSheet(f"background:{color}; color:white; border-radius:12px; padding:12px 18px;")
        self.setText(f'<div style="font-size:19pt; font-weight:700;">{title}</div>'
                     f'<div style="font-size:10.5pt;">{html.escape(model_name)}: P(phishing) = {p:.3f}'
                     f' &nbsp;·&nbsp; threshold {thr:.2f}</div>')


class Canvas(FigureCanvasQTAgg):
    """Matplotlib figure styled to match the app."""

    def __init__(self, height=3.2, width=5.0):
        self.fig = Figure(figsize=(width, height), dpi=100, facecolor="white", layout="constrained")
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(int(height * 100))

    def axes(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_color(MUTED)
        return ax


def attention_html(pairs):
    """Each token's background shows its BiLSTM attention weight."""
    if not pairs:
        return "<i>No attention available.</i>"
    top = max(w for _, w in pairs) or 1.0
    spans = []
    for tok, w in pairs:
        r = w / top
        alpha = 0.10 + 0.85 * r if r > 0.05 else 0.0
        bg = f"background-color: rgba(235,104,52,{alpha:.2f});" if alpha else ""
        bold = "font-weight:700;" if r > 0.5 else ""
        spans.append(f'<span style="{bg}{bold}">&nbsp;{html.escape(tok)}&nbsp;</span>')
    return f'<p style="line-height:190%; font-size:10.5pt;">{" ".join(spans)}</p>'


def signals_html(found):
    if not found:
        return "<p>None of the common signals were found.</p>"
    rows = []
    for name, hits in found.items():
        chips = " ".join(f'<span style="background-color:#fbe1d6;">&nbsp;{html.escape(h)}&nbsp;</span>'
                         for h in hits[:8])
        rows.append(f"<p><b>{html.escape(name)}</b>&nbsp;&nbsp;{chips}</p>")
    return "".join(rows)


class DataFrameModel(QAbstractTableModel):
    """Read-only table model for a pandas DataFrame. UserRole returns the raw
    value so a QSortFilterProxyModel sorts numbers numerically."""

    def __init__(self, df=None, formats=None, colors=None):
        super().__init__()
        self.df = df if df is not None else pd.DataFrame()
        self.formats = formats or {}
        self.colors = colors or {}

    def set_frame(self, df):
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col = self.df.columns[index.column()]
        value = self.df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            fmt = self.formats.get(col)
            if fmt:
                return fmt(value)
            text = str(value)
            return text if len(text) < 200 else text[:200] + "…"
        if role == Qt.UserRole:
            return value.item() if hasattr(value, "item") else value
        if role == Qt.ToolTipRole and isinstance(value, str) and len(value) > 60:
            return value[:1000]
        if role == Qt.ForegroundRole and col in self.colors:
            color = self.colors[col](value)
            return QColor(color) if color else None
        if role == Qt.TextAlignmentRole and not isinstance(value, str):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return str(self.df.columns[section])
        return None


def verdict_color(value):
    return {"phishing": ORANGE, "safe": BLUE}.get(value)


def pct(v):
    return f"{v:.2%}"


def prob(v):
    return f"{v:.3f}"


__all__ = ["Canvas", "DataFrameModel", "NAVY", "ProbCard", "Tile", "VerdictBanner", "attention_html", "hbox",
           "label", "page_header", "pct", "prob", "signals_html", "verdict_color", "verdict_tag"]
