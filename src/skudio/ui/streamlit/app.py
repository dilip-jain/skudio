"""skudio.ui.streamlit - Streamlit (Single-page, top-to-bottom flow) UI adapter for skudio."""

from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
import streamlit as st

import skudio
from skudio.data.reader import load_csv
from skudio.server.token import verify_token

_TOKEN_ENV = "SKUDIO_TOKEN"


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title=skudio.NAME, layout="wide")
    if not _token_ok():
        _lock_screen()
        return

    st.title(skudio.NAME)
    st.caption(skudio.TAGLINE)

    df = _dataset_section()
    if df is None:
        return

    target, features = _target_and_features(df)
    if target is None or not features:
        return

    task = _infer_task(df, target)
    task = st.radio(
        "Task",
        options=["classification", "regression"],
        index=0 if task == "classification" else 1,
        horizontal=True,
        help="Auto-detected from the target column. Override if needed.",
    )


def _token_ok() -> bool:
    """Check if the token is valid."""
    expected = os.environ.get(_TOKEN_ENV)
    if not expected:
        return True # dev-mode (--no-token or direct streamlit run)

    candidate = st.query_params.get("token", "")
    ok = verify_token(expected, candidate)
    if ok:
        st.session_state["_token_verified"] = True
    return ok or bool(st.session_state.get("_token_verified", False))


def _lock_screen() -> None:
    """Display a lock screen if the token is invalid."""
    st.title(skudio.NAME)
    st.error("Invalid token. Please provide a valid token in the URL query parameters.")
    st.stop()


# ---------------------------------------------------------------------------
# Dataset


def _dataset_section() -> Any:
    st.header("1. Dataset")
    upload = st.file_uploader("Upload a CSV", type=["csv"])
    if upload is None:
        st.info("Upload a CSV to begin.")
        return None

    df = _load_uploaded(upload)
    return df


def _load_uploaded(upload: Any) -> Any:
    key = f"csv::{upload.name}::{upload.size}"
    cached = st.session_state.get(key)
    if cached is not None:
        return cached

    df = load_csv(io.BytesIO(upload.getvalue()))
    st.session_state[key] = df
    st.session_state["_last_upload"] = upload
    return df


# ---------------------------------------------------------------------------
# Explore (EDA)


def _target_and_features(df: Any) -> tuple[str | None, list[str]]:
    st.header("2. Target and features")
    all_cols: list[str] = [str(c) for c in df.columns]
    target = st.selectbox("Target column", options=["", *all_cols], index=0)
    if not target:
        st.info("Pick the target column.")
        return None, []

    default_features = [c for c in all_cols if c != target]
    features = st.multiselect("Feature columns", options=default_features, default=default_features)
    if not features:
        st.info("Pick at least one feature column.")
    return target, features


def _infer_task(df: Any, target: str) -> str:
    s = df[target]
    if pd.api.types.is_bool_dtype(s):
        return "classification"
    if pd.api.types.is_numeric_dtype(s):
        n = len(s)
        n_unique = int(s.nunique(dropna=True))
        if n and (n_unique > 20 or n_unique / n > 0.05):
            return "regression"
    return "classification"


# streamlit run app.py
if __name__ == "__main__":
    main()
