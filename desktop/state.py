"""Application state shared by every page, plus a helper to run slow work
(model scoring) on a background thread so the window never freezes."""
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

import app_core


class AppState(QObject):
    thresholdChanged = Signal(float)
    testReady = Signal()

    def __init__(self, models):
        super().__init__()
        self.models = models
        self.names = app_core.model_names(models)
        self.threshold = 0.5
        self.test_df = None
        self.metrics = app_core.metrics()

    def set_threshold(self, value):
        value = round(float(value), 2)
        if value != self.threshold:
            self.threshold = value
            self.thresholdChanged.emit(value)

    def set_test_df(self, df):
        self.test_df = df
        self.testReady.emit()


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn, signals):
        super().__init__()
        self.fn, self.signals = fn, signals

    @Slot()
    def run(self):
        try:
            result = self.fn()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.done.emit(result)


_alive = set()


def run_in_background(fn, on_done, on_error=None):
    """Run fn() on the thread pool; on_done(result) is called on the UI thread."""
    signals = _Signals()
    _alive.add(signals)
    signals.done.connect(on_done)
    if on_error:
        signals.failed.connect(on_error)
    signals.done.connect(lambda _=None: _alive.discard(signals))
    signals.failed.connect(lambda _=None: _alive.discard(signals))
    QThreadPool.globalInstance().start(_Task(fn, signals))
