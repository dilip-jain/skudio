"""skudio.ui.streamlit - Streamlit (Single-page, top-to-bottom flow) UI adapter for skudio."""

from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
import streamlit as st

import skudio
from skudio.components.metadata import ComponentMetadata, ParamDescriptor
from skudio.components.registry import default_registry
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

    _split_columns(df, features)
    st.divider()

    _preprocessing_section()
    st.divider()



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


def _split_columns(df: Any, features: list[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)

    with st.expander("Column roles", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            numeric = st.multiselect("Numeric", options=features, default=numeric)
        with col2:
            leftover = [c for c in features if c not in numeric]
            categorical = st.multiselect("Categorical", options=features, default=leftover)
    return numeric, categorical


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


# ---------------------------------------------------------------------------
# Train & Report


def _preproc_default_numeric() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("sklearn.impute.SimpleImputer", {"strategy": "median"}),
        ("sklearn.preprocessing.StandardScaler", {}),
    ]


def _preproc_default_categorical() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("sklearn.impute.SimpleImputer", {"strategy": "most_frequent"}),
        ("sklearn.preprocessing.OneHotEncoder", {"handle_unknown": "ignore"}),
    ]


def _preprocessing_section() -> tuple[
    list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]
]:
    st.header("3. Preprocessing")
    st.caption("Edit the numeric and categorical branches. Each step compiles to a Pipeline.")

    reg = default_registry()
    pre_metas = [m for m in reg.list() if m.category == "preprocessing"]
    pre_by_qn = {m.qualname: m for m in pre_metas}
    labels = {m.qualname: m.display_name for m in pre_metas}

    if "_preproc_numeric" not in st.session_state:
        st.session_state["_preproc_numeric"] = _preproc_default_numeric()
    if "_preproc_categorical" not in st.session_state:
        st.session_state["_preproc_categorical"] = _preproc_default_categorical()

    col_num, col_cat = st.columns(2)
    with col_num:
        st.subheader("Numeric")
        _branch_editor("numeric", pre_by_qn, labels)
    with col_cat:
        st.subheader("Categorical")
        _branch_editor("categorical", pre_by_qn, labels)

    return (
        list(st.session_state["_preproc_numeric"]),
        list(st.session_state["_preproc_categorical"]),
    )


def _branch_editor(
    branch: str, pre_by_qn: dict[str, ComponentMetadata], labels: dict[str, str],
) -> None:
    key = f"_preproc_{branch}"
    steps: list[tuple[str, dict[str, Any]]] = st.session_state[key]

    for i, (qn, _prm) in enumerate(list(steps)):
        display = labels.get(qn, qn.rsplit(".", 1)[-1])
        with st.expander(f"{i + 1}. {display}", expanded=False):
            meta = pre_by_qn.get(qn)
            if meta is not None:
                new_params = _param_form_prefixed(meta, key_prefix=f"{branch}::{i}::")
                steps[i] = (qn, new_params)
            if st.button("Remove", key=f"{branch}_rm_{i}"):
                steps.pop(i)
                st.session_state[key] = steps
                st.rerun()


    def _fmt(q: str) -> str:
        return "(none)" if q == "(none)" else labels.get(q, q)

    add_options = list(pre_by_qn.keys())
    add_choice = st.selectbox(
        "Add step",
        options=["(none)", *add_options],
        format_func=_fmt,
        key=f"{branch}_add_sel",
    )

    if st.button("Add", key=f"{branch}_add_btn") and add_choice != "(none)":
        steps.append((add_choice, {}))
        st.session_state[key] = steps
        st.rerun()


def _param_form_prefixed(meta: ComponentMetadata, *, key_prefix: str) -> dict[str, Any]:
    """Parameter form variant that isolates widget keys with a caller-provided prefix."""
    basic = [p for p in meta.params if p.tier == "basic"]
    advanced = [p for p in meta.params if p.tier == "advanced"]
    out: dict[str, Any] = {}

    for p in basic:
        val = _widget_for(f"{key_prefix}{meta.qualname}", p)
        if val is None and p.default is None:
            continue
        out[p.name] = val

    if advanced:
        with st.expander("Advanced"):
            for p in advanced:
                val = _widget_for(f"{key_prefix}{meta.qualname}", p)
                if val is None and p.default is None:
                    continue
                out[p.name] = val
    return out


def _widget_for(key_prefix: str, p: ParamDescriptor) -> Any:
    key = f"{key_prefix}::{p.name}"
    label = p.name
    if p.widget == "select" and p.choices is not None:
        choices = p.choices
        default_index = choices.index(p.default) if p.default in choices else 0
        return st.selectbox(label, options=choices, index=default_index, key=key)
    if p.widget == "toggle":
        return st.checkbox(label, value=bool(p.default), key=key)
    if p.widget == "int":
        return _int_input(label, p, key)
    if p.widget in ("number", "slider"):
        return _number_input(label, p, key)

    text = st.text_input(label, value="" if p.default is None else str(p.default), key=key)
    return text or None


def _int_input(label: str, p: ParamDescriptor, key: str) -> int | None:
    lo = int(p.min) if p.min is not None else None
    hi = int(p.max) if p.max is not None else None
    if p.default is None:
        if not st.checkbox(
            f"Override `{p.name}` (default: None)", value=False, key=f"{key}::override"
        ):
            return None

        start = lo if lo is not None else 1
    else:
        start = int(p.default)

    val = st.number_input(
        label,
        value=start,
        step=int(p.step) if p.step else 1,
        min_value=lo,
        max_value=hi,
        key=key,
    )
    return int(val)


def _number_input(label: str, p: ParamDescriptor, key: str) -> float | None:
    lo = float(p.min) if p.min is not None else None
    hi = float(p.max) if p.max is not None else None
    if p.default is None:
        if not st.checkbox(
            f"Override `{p.name}` (default: None)", value=False, key=f"{key}::override"
        ):
            return None

        start = lo if lo is not None else 0.0
    else:
        start = float(p.default)

    val = st.number_input(
        label,
        value=start,
        step=float(p.step) if p.step else 0.1,
        min_value=lo,
        max_value=hi,
        format="%.6g",
        key=key,
    )
    return float(val)


# streamlit run app.py
if __name__ == "__main__":
    main()
