"""Phishing Message Detector - desktop app (PySide6 / Qt).

Run:  python desktop/main.py        (or double-click desktop/run_desktop.bat)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src")]

import torch  # noqa: E402, F401  (import before Qt: avoids DLL conflicts on Windows)
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen  # noqa: E402

from theme import NAVY, ORANGE, QSS  # noqa: E402

CACHE = HERE / ".cache" / "test_predictions.csv"


def make_icon(size=256):
    """Envelope with an orange warning badge, drawn in code (no image files)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 256
    p.setBrush(QColor(NAVY))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(8 * s, 8 * s, 240 * s, 240 * s), 48 * s, 48 * s)
    p.setPen(QPen(QColor("white"), 14 * s, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(QRectF(42 * s, 72 * s, 150 * s, 108 * s), 12 * s, 12 * s)
    path = QPainterPath()
    path.moveTo(46 * s, 80 * s)
    path.lineTo(117 * s, 132 * s)
    path.lineTo(188 * s, 80 * s)
    p.drawPath(path)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(ORANGE))
    p.drawEllipse(QRectF(150 * s, 136 * s, 82 * s, 82 * s))
    p.setPen(QColor("white"))
    f = QFont("Segoe UI", int(52 * s))
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(150 * s, 136 * s, 82 * s, 82 * s), Qt.AlignCenter, "!")
    p.end()
    return pm


def make_splash():
    pm = QPixmap(560, 300)
    pm.fill(QColor(NAVY))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.drawPixmap(40, 70, make_icon(120))
    p.setPen(QColor("white"))
    f = QFont("Segoe UI", 20)
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(185, 78, 360, 80), Qt.AlignLeft | Qt.TextWordWrap, "Phishing Message\nDetector")
    p.setFont(QFont("Segoe UI", 10))
    p.setPen(QColor("#aeb8c8"))
    p.drawText(QRectF(187, 160, 360, 40), Qt.AlignLeft, "Deep learning text classification")
    p.end()
    return pm


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Phishing Message Detector")
    app.setStyleSheet(QSS)
    app.setWindowIcon(QIcon(make_icon()))

    splash = QSplashScreen(make_splash())
    splash.show()
    splash.showMessage("Loading trained models…", Qt.AlignBottom | Qt.AlignHCenter, QColor("white"))
    app.processEvents()

    import app_core
    models = app_core.load_models()
    if not models:
        splash.close()
        QMessageBox.critical(None, "No trained models",
                             "No trained models were found in the 'models' folder.\n\n"
                             "Train them first from the project folder:\n    python run_all.py")
        return 1

    from state import AppState, run_in_background
    from window import MainWindow

    state = AppState(models)
    window = MainWindow(state)
    window.show()
    splash.finish(window)

    window.statusBar().showMessage("Scoring the test set for the performance pages…")
    run_in_background(lambda: app_core.test_predictions(models, CACHE),
                      lambda df: (state.set_test_df(df), window.statusBar().showMessage(
                          f"Ready · {len(models)} models loaded · test set: {len(df):,} messages", 0)),
                      lambda msg: window.statusBar().showMessage(f"Could not score the test set: {msg}"))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
