"""Test: Graph-builder for Streamlit UI"""

from __future__ import annotations

import ast

from skudio.core.codegen import emit_module
from skudio.core.validator import validate
from skudio.ui.streamlit.graph_builder import build_graph


def test_graph_with_numeric_and_categorical_validates_and_emits() -> None:
    """Test: Graph with numerical and categorical."""
    g = build_graph(
        estimator_qualname="sklearn.ensemble.RandomForestClassifier",
        estimator_params={"n_estimators": 50, "random_state": 0},
        numeric_columns=["age", "fare"],
        categorical_columns=["sex"],
    )
    report = validate(g)
    assert report.ok, report.errors()
    code = emit_module(g)
    ast.parse(code) # emitted code parses as valid Python
    assert "RandomForestClassifier(n_estimators=50, random_state=0)" in code
    assert "ColumnTransformer(" in code


def test_graph_without_preprocessing_still_valid() -> None:
    """Test: Graph without preprocessing."""
    g = build_graph(
        estimator_qualname="sklearn.linear_model.LogisticRegression",
        estimator_params={"C": 1.0},
        numeric_columns=[],
        categorical_columns=[],
    )
    report = validate(g)
    assert report.ok
    code = emit_module(g)
    assert "LogisticRegression(C=1.0)" in code
    assert "ColumnTransformer" not in code


def test_only_numeric_columns_omits_categorical_branch() -> None:
    """Test: Numeric columns must omit categorical branch."""
    g = build_graph(
        estimator_qualname="sklearn.ensemble.RandomForestClassifier",
        estimator_params={},
        numeric_columns=["a", "b", "c"],
        categorical_columns=[],
    )
    report = validate(g)
    assert report.ok
    code = emit_module(g)
    assert "['a', 'b', 'c']" in code
    assert "OneHotEncoder" not in code
