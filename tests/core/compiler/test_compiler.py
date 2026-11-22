"""Test: Compiler."""

from __future__ import annotations

import pytest

from skudio.core.compiler import compile_graph
from skudio.core.errors import CompilerError
from skudio.core.ir import (
    ComponentRef,
    Edge,
    Graph,
    LiteralValue,
    Node,
    NodeRef,
    PortSpec,
)

pytest.importorskip("sklearn")


def _lr_node(node_id: str = "lr", *, params: dict[str, object] | None = None) -> Node:
    return Node(
        id=node_id,
        kind="estimator",
        component=ComponentRef(
            provider="sklearn",
            qualname="sklearn.linear_model.LogisticRegression",
        ),
        params={k: LiteralValue(value=v) for k, v in (params or {}).items()},
        inputs=[PortSpec(name="in", type="dataframe")],
    )


def _dataset_node() -> Node:
    return Node(id="data", kind="dataset", outputs=[PortSpec(name="out", type="dataframe")])


def _edge(src_id: str, dst_id: str) -> Edge:
    return Edge(
        src=NodeRef(node_id=src_id, port="out"),
        dst=NodeRef(node_id=dst_id, port="in"),
    )


def test_single_estimator_returns_bare_estimator() -> None:
    """Test: Estimator-only Graph."""
    from sklearn.linear_model import LogisticRegression  # pylint: disable=import-outside-toplevel

    graph = Graph(
        id="g", name="lr", nodes=[_dataset_node(), _lr_node()], edges=[_edge("data", "lr")]
    )
    est = compile_graph(graph)
    assert isinstance(est, LogisticRegression)


def test_linear_pipeline_step_names_match_ids() -> None:
    """Test: Linear Pipeline step names match ids."""
    from sklearn.pipeline import Pipeline  # pylint: disable=import-outside-toplevel

    scaler = Node(
        id="scaler",
        kind="transformer",
        component=ComponentRef(provider="sklearn", qualname="sklearn.preprocessing.StandardScaler"),
    )
    graph = Graph(
        id="g",
        name="linear",
        nodes=[_dataset_node(), scaler, _lr_node()],
        edges=[_edge("data", "scaler"), _edge("scaler", "lr")],
    )
    p = compile_graph(graph)
    assert isinstance(p, Pipeline)
    assert [name for name, _ in p.steps] == ["scaler", "lr"]


def test_random_state_propagates_from_graph_root() -> None:
    """Test: Graph random_state must propagate."""
    graph = Graph(
        id="g",
        name="lr",
        random_state=42,
        nodes=[_dataset_node(), _lr_node()],
        edges=[_edge("data", "lr")],
    )
    est = compile_graph(graph)
    assert est.random_state == 42


def test_explicit_param_beats_graph_random_state() -> None:
    """Test: Set random_state explicit."""
    graph = Graph(
        id="g",
        name="lr",
        random_state=42,
        nodes=[_dataset_node(), _lr_node(params={"random_state": 7})],
        edges=[_edge("data", "lr")],
    )
    est = compile_graph(graph)
    assert est.random_state == 7


def test_component_without_random_state_slot_is_untouched() -> None:
    """Test: Components with no random_state parameter must not have one."""
    # GaussianNB doesn't have a random_state kwarg; injection must not attempt it.
    graph = Graph(
        id="g",
        name="nb",
        random_state=42,
        nodes=[
            _dataset_node(),
            Node(
                id="nb",
                kind="estimator",
                component=ComponentRef(
                    provider="sklearn",
                    qualname="sklearn.naive_bayes.GaussianNB",
                ),
            ),
        ],
        edges=[_edge("data", "nb")],
    )
    est = compile_graph(graph)
    assert not hasattr(est, "random_state") or est.random_state is None


def test_empty_graph_raises() -> None:
    """Test: Empty graph."""
    graph = Graph(id="g", name="empty", nodes=[_dataset_node()], edges=[])
    with pytest.raises(CompilerError):
        compile_graph(graph)


def test_column_transformer_preserves_column_lists() -> None:
    """Test: ColumnTransformer must preserve column lists."""
    from sklearn.compose import ColumnTransformer  # pylint: disable=import-outside-toplevel

    scaler = Node(
        id="num",
        kind="transformer",
        component=ComponentRef(provider="sklearn", qualname="sklearn.preprocessing.StandardScaler"),
        params={"columns": LiteralValue(value=["a", "b"])},
    )

    ohe = Node(
        id="cat",
        kind="transformer",
        component=ComponentRef(provider="sklearn", qualname="sklearn.preprocessing.OneHotEncoder"),
        params={
            "columns": LiteralValue(value=["x"]),
            "handle_unknown": LiteralValue(value="ignore"),
        },
    )

    ct = Node(
        id="preproc",
        kind="column_transformer",
        children=Graph(id="ct_kids", nodes=[scaler, ohe], edges=[]),
        inputs=[PortSpec(name="in", type="dataframe")],
        outputs=[PortSpec(name="out", type="ndarray")],
    )

    graph = Graph(
        id="g",
        name="ct",
        nodes=[_dataset_node(), ct, _lr_node()],
        edges=[_edge("data", "preproc"), _edge("preproc", "lr")],
    )

    p = compile_graph(graph)
    ct_step = p.steps[0][1]
    assert isinstance(ct_step, ColumnTransformer)

    cols_by_name = {name: cols for name, _, cols in ct_step.transformers}
    assert cols_by_name["num"] == ["a", "b"]
    assert cols_by_name["cat"] == ["x"]
