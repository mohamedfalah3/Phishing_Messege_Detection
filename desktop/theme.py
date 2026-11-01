"""Colours and Qt stylesheet for the desktop app."""
NAVY = "#0f1b2d"
NAVY_2 = "#23344f"
ORANGE = "#eb6834"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
PANEL = "#eef2f7"
BORDER = "#d9dee7"
MUTED = "#4a5568"
MODEL_COLORS = {"baseline": BLUE, "cnn": ORANGE, "bilstm": AQUA, "ensemble": NAVY}

QSS = f"""
QWidget {{ font-family: 'Segoe UI'; font-size: 10pt; color: {NAVY}; }}
QMainWindow, #content, #page {{ background: #ffffff; }}
#sidebar {{ background: {NAVY}; }}
#sidebar QLabel {{ color: #ffffff; }}
#appTitle {{ font-size: 13pt; font-weight: 700; }}
#sidebar QLabel#appSubtitle, #sidebar QLabel#sideCaption {{ color: #aeb8c8; font-size: 9pt; }}
#nav {{ background: transparent; border: none; color: #d6deea; font-size: 10.5pt; outline: none; }}
#nav::item {{ padding: 9px 12px; border-radius: 8px; margin: 2px 6px; }}
#nav::item:selected {{ background: {NAVY_2}; color: #ffffff; }}
#nav::item:hover:!selected {{ background: #1a2a42; }}
#sidebar QSlider::groove:horizontal {{ height: 6px; background: #31435f; border-radius: 3px; }}
#sidebar QSlider::sub-page:horizontal {{ background: {ORANGE}; border-radius: 3px; }}
#sidebar QSlider::handle:horizontal {{ background: #ffffff; width: 16px; margin: -6px 0; border-radius: 8px; }}
#sidebar QPushButton {{ background: {NAVY_2}; color: #ffffff; border: none; }}
#sidebar QPushButton:hover {{ background: #31435f; }}
QLabel#pageTitle {{ font-size: 20pt; font-weight: 700; }}
QLabel#section {{ font-size: 13pt; font-weight: 700; padding-top: 6px; }}
QLabel#caption {{ color: {MUTED}; }}
QFrame#card {{ background: {PANEL}; border-radius: 10px; }}
QFrame#tile {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#tileLabel {{ color: {MUTED}; font-size: 9pt; }}
QLabel#tileValue {{ font-size: 18pt; font-weight: 700; }}
QPushButton {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ background: {PANEL}; }}
QPushButton:disabled {{ color: #a0a8b5; }}
QPushButton#primary {{ background: {NAVY}; color: #ffffff; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: {NAVY_2}; }}
QPushButton:checked {{ background: {NAVY}; color: #ffffff; border-color: {NAVY}; }}
QPlainTextEdit, QTextBrowser, QLineEdit, QComboBox, QSpinBox {{
    background: #ffffff; border: 1px solid {BORDER}; border-radius: 8px; padding: 5px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QTableView {{ border: 1px solid {BORDER}; border-radius: 8px; gridline-color: #edf0f5;
    selection-background-color: #dbe7f8; selection-color: {NAVY}; alternate-background-color: #f8f9fb; }}
QHeaderView::section {{ background: {PANEL}; border: none; padding: 6px; font-weight: 600; }}
QProgressBar {{ border: none; background: #e2e6ed; border-radius: 4px; }}
QProgressBar::chunk {{ border-radius: 4px; background: {NAVY}; }}
QScrollArea {{ border: none; background: #ffffff; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ padding: 7px 14px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px;
    background: {PANEL}; }}
QTabBar::tab:selected {{ background: {NAVY}; color: #ffffff; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QSplitter::handle {{ background: {BORDER}; }}
"""
