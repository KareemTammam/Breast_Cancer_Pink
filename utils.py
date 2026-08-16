"""Cached loaders for model artifacts and data."""

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "model"
DATA_DIR = Path(__file__).parent / "data"


@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_DIR / "trained_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_scaler():
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_feature_names():
    with open(MODEL_DIR / "feature_names.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_meta():
    with open(MODEL_DIR / "meta.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_comparison():
    return pd.read_json(MODEL_DIR / "model_comparison.json")


@st.cache_data(show_spinner=False)
def load_feature_importance():
    return pd.read_json(MODEL_DIR / "feature_importance.json")


@st.cache_data(show_spinner=False)
def load_confusion_matrices():
    with open(MODEL_DIR / "confusion_matrices.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_target_correlation():
    s = pd.read_json(MODEL_DIR / "target_correlation.json", typ="series")
    return s


@st.cache_data(show_spinner=False)
def load_feature_stats():
    with open(MODEL_DIR / "feature_stats.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_sample_data():
    return pd.read_csv(MODEL_DIR / "sample_data.csv")


def prettify_feature(name: str) -> str:
    """Turn 'concave points_worst' into 'Concave Points (Worst)'."""
    suffix_map = {"_mean": " (Mean)", "_se": " (SE)", "_worst": " (Worst)"}
    for suffix, label in suffix_map.items():
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            return base.replace("_", " ").title() + label
    return name.replace("_", " ").title()


def build_raw_input_row(form_values: dict, feature_names: list) -> pd.DataFrame:
    """Build a single-row DataFrame matching the exact column order the
    scaler was fit on."""
    row = {col: form_values[col] for col in feature_names}
    return pd.DataFrame([row], columns=feature_names)
