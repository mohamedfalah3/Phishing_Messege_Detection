"""Streamlit wrappers (caching) around the shared helpers in src/app_core.py,
so the web UI gives exactly the same predictions as the scripts."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import app_core  # noqa: E402
from app_core import (COLORS, EXAMPLES, LONG_NAMES, SHORT_NAMES, confusion,  # noqa: E402, F401
                      cost_threshold, signals, threshold_sweep)
from config import COST_FN, COST_FP, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402, F401

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    return app_core.load_models()


def model_names():
    return app_core.model_names(get_models())


def predict(texts) -> pd.DataFrame:
    return app_core.predict(get_models(), texts)


def attention(text):
    return app_core.attention(get_models(), text)


@st.cache_data(show_spinner="Scoring the test set with every model (first run only)...")
def test_predictions() -> pd.DataFrame:
    return app_core.test_predictions(get_models(), CACHE_DIR / "test_predictions.csv")


@st.cache_data
def metrics():
    return app_core.metrics()


def figure(name):
    path = FIGURES_DIR / name
    return str(path) if path.exists() else None
