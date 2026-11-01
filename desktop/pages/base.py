from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget


class Page(QScrollArea):
    """Scrollable page with a vertical layout in self.lay."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName("page")
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(36, 28, 36, 28)
        self.lay.setSpacing(12)
        self.setWidget(body)

    def finish(self):
        self.lay.addStretch(1)
