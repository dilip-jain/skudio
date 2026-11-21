"""Test: Codegen."""

from __future__ import annotations

import ast

from skudio.core.codegen import emit_module
from skudio.core.codegen.emit import emit_script
from skudio.core.ir import (
    ComponentRef,
    Edge,
    Node,
    Graph,
    LiteralValue,
    NodeRef,
    PortSpec
)


def _linear_graph() -> Graph:
    scaler = Node(
        id="scaler",
        kind="transformer",
        component=ComponentRef(provider="sklearn", qualname="sklearn.preprocessing.StandardScaler"),
        inputs=[PortSpec(name="in", type="dataframe")],
        outputs=[PortSpec(name="out", type="dataframe")],
    )

    lr = Node(
        id="lr",
        kind="estimator",
        component=ComponentRef(
            provider="sklearn", qualname="sklearn.linear_model.LogisticRegression"
        ),
        params={"C": LiteralValue(value=1.0), "max_iter": LiteralValue(value=200)},
        inputs=[PortSpec(name="in", type="dataframe")],
    )

    return Graph(
        id="g",
        name="linear",
        nodes=[
            Node(id="data", kind="dataset", outputs=[PortSpec(name="out", type="dataframe")]),
            scaler,
            lr,
        ],
        edges=[
            Edge(src=NodeRef(node_id="data", port="out"), dst=NodeRef(node_id="scaler", port="in")),
            Edge(src=NodeRef(node_id="scaler", port="out"), dst=NodeRef(node_id="lr", port="in")),
        ],
    )


def _ct_graph() -> Graph:
    numeric = Node(
        id="num",
        kind="transformer",
        component=ComponentRef(provider="sklearn", qualname="sklearn.preprocessing.StandardScaler"),
        params={"columns": LiteralValue(value=["age", "fare"])},
    )

    categorical = Node(
        id="cat",
        kind="transformer",
        component=ComponentRef(provider="sklearn", qualname="sklearn.preprocessing.OneHotEncoder"),
        params={
            "columns": LiteralValue(value=["sex"]),
            "handle_unknown": LiteralValue(value="ignore"),
        },
    )

    ct = Node(
        id="preproc",
        kind="column_transformer",
        children=Graph(id="ct_children", nodes=[numeric, categorical], edges=[]),
        inputs=[PortSpec(name="in", type="dataframe")],
        outputs=[PortSpec(name="out", type="ndarray")],
    )

    model = Node(
        id="rf",
        kind="estimator",
        component=ComponentRef(
            provider="sklearn",
            qualname="sklearn.ensemble.RandomForestClassifier"
        ),
        params={"n_estimators": LiteralValue(value=50)},
        inputs=[PortSpec(name="in", type="ndarray")],
    )

    return Graph(id="g", name="titanic",
        nodes=[
            Node(id="data", kind="dataset", outputs=[PortSpec(name="out", type="dataframe")]),
            ct,
            model,
        ],
        edges=[
            Edge(
                src=NodeRef(node_id="data", port="out"), dst=NodeRef(node_id="preproc", port="in")
            ),
            Edge(src=NodeRef(node_id="preproc", port="out"), dst=NodeRef(node_id="rf", port="in")),
        ],
    )

def test_linear_emit_parses_and_contains_expected_calls() -> None:
    """Test: Emit linear graph."""
    code = emit_module(_linear_graph())
    parsed = ast.parse(code)
    assert isinstance(parsed, ast.Module)
    assert "def build_pipeline():" in code
    assert "from sklearn.linear_model import LogisticRegression" in code
    assert "from sklearn.preprocessing import StandardScaler" in code
    assert "from sklearn.pipeline import Pipeline" in code
    assert "LogisticRegression(C=1.0, max_iter=200)" in code
    assert "StandardScaler()" in code

def test_ct_emit_parses_and_uses_column_transformer() -> None:
    """Test: Emit ColumnTransformer."""
    code = emit_module(_ct_graph())
    ast.parse(code)  # must be valid Python
    assert "from sklearn.compose import ColumnTransformer" in code
    assert "from sklearn.ensemble import RandomForestClassifier" in code
    assert "ColumnTransformer(" in code
    assert "['age', 'fare']" in code
    assert "['sex']" in code
    assert "handle_unknown='ignore'" in code

def test_graph_random_state_propagates_into_emitted_calls() -> None:
    """Test: Graph random_state must propagate into emitted calls."""
    g = _linear_graph()
    g.random_state = 123
    code = emit_module(g)
    ast.parse(code)

    # LogisticRegression has a random_state slot; StandardScaler does not.
    assert "LogisticRegression(C=1.0, max_iter=200, random_state=123)" in code
    assert "StandardScaler()" in code

def test_explicit_param_beats_graph_random_state_in_codegen() -> None:
    """Test: Explicit Parameters must take precedence over graph random state."""
    scaler = Node(
        id="scaler",
        kind="transformer",
        component=ComponentRef(provider="sklearn", qualname="sklearn.preprocessing.StandardScaler"),
    )

    lr = Node(
        id="lr",
        kind="estimator",
        component=ComponentRef(
            provider="sklearn", qualname="sklearn.linear_model.LogisticRegression"
        ),
        params={"random_state": LiteralValue(value=7)},
    )

    g = Graph(
        id="g",
        name="linear",
        random_state=123,
        nodes=[
            Node(id="data", kind="dataset", outputs=[PortSpec(name="out", type="dataframe")]),
            scaler,
            lr,
        ],
        edges=[
            Edge(src=NodeRef(node_id="data", port="out"), dst=NodeRef(node_id="scaler", port="in")),
            Edge(src=NodeRef(node_id="scaler", port="out"), dst=NodeRef(node_id="lr", port="in")),
        ],
    )
    code = emit_module(g)

    assert "random_state=7" in code
    assert "random_state=123" not in code

def test_script_uses_pandas_alias_and_classification_metric_by_default() -> None:
    """Test: Pandas."""
    code = emit_script(_linear_graph(), dataset_path="/tmp/data.csv", target="y")
    ast.parse(code)

    assert "import pandas as pd" in code
    assert "pd.read_csv('/tmp/data.csv')" in code
    assert "from sklearn.metrics import accuracy_score" in code
    assert "accuracy_score" in code
    assert "r2_score" not in code

def test_script_switches_metric_for_regression_task() -> None:
    """Test: Task-specific metrics."""
    g = _linear_graph()
    g.random_state = 5
    code = emit_script(g, dataset_path="d.csv", target="y", task="regression")
    ast.parse(code)

    assert "from sklearn.metrics import r2_score" in code
    assert "r2_score" in code
    assert "accuracy_score" not in code
    # graph.random_state also propagates into the train_test_split call.
    assert "random_state=5" in code
