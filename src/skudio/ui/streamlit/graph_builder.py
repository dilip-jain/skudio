"""Assemble an IR Graph from a compact UI selection dict."""

from __future__ import annotations

from typing import Any

from skudio.components.registry import Registry, default_registry
from skudio.core.errors import RegistryError
from skudio.core.ir import (
    ComponentRef,
    Edge,
    Graph,
    LiteralValue,
    Node,
    NodeRef,
    PortSpec,
)


def _component_ref(qualname: str, registry: Registry | None = None) -> ComponentRef:
    reg = registry or default_registry()
    try:
        meta = reg.get(qualname)
        version = meta.version
    except RegistryError:
        version = ""
    return ComponentRef(provider="sklearn", qualname=qualname, version=version)


def build_graph(
    *,
    estimator_qualname: str,
    estimator_params: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_steps: list[tuple[str, dict[str, Any]]] | None = None,
    categorical_steps: list[tuple[str, dict[str, Any]]] | None = None,
    graph_id: str = "user_graph",
    name: str = "pipeline",
) -> Graph:
    """Build the standard MVP graph: dataset -> ColumnTransformer -> estimator."""
    numeric_steps = numeric_steps or _default_numeric_steps()
    categorical_steps = categorical_steps or _default_categorical_steps()

    dataset = Node(
        id="data",
        kind="dataset",
        outputs=[PortSpec(name="out", type="dataframe")],
    )
    nodes: list[Node] = [dataset]
    edges: list[Edge] = []

    has_preproc = bool(numeric_columns or categorical_columns)
    if has_preproc:
        ct = _build_column_transformer(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            numeric_steps=numeric_steps,
            categorical_steps=categorical_steps,
        )
        nodes.append(ct)
        edges.append(
            Edge(
                src=NodeRef(node_id="data", port="out"),
                dst=NodeRef(node_id="preproc", port="in"),
            )
        )

    model = Node(
        id="model",
        kind="estimator",
        component=_component_ref(estimator_qualname),
        params={k: LiteralValue(value=v) for k, v in estimator_params.items()},
        inputs=[PortSpec(name="in", type="dataframe")],
    )

    nodes.append(model)
    if has_preproc:
        edges.append(
            Edge(
                src=NodeRef(node_id="preproc", port="out"),
                dst=NodeRef(node_id="model", port="in"),
            )
        )
    else:
        edges.append(
            Edge(
                src=NodeRef(node_id="data", port="out"),
                dst=NodeRef(node_id="model", port="in"),
            )
        )

    return Graph(id=graph_id, name=name, nodes=nodes, edges=edges)


def _build_column_transformer(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_steps: list[tuple[str, dict[str, Any]]],
    categorical_steps: list[tuple[str, dict[str, Any]]],
) -> Node:
    children_nodes: list[Node] = []
    if numeric_columns:
        children_nodes.append(
            _branch(
                branch_id="num",
                columns=numeric_columns,
                steps=numeric_steps,
            )
        )
    if categorical_columns:
        children_nodes.append(
            _branch(
                branch_id="cat",
                columns=categorical_columns,
                steps=categorical_steps,
            )
        )

    return Node(
        id="preproc",
        kind="column_transformer",
        children=Graph(id="preproc_children", nodes=children_nodes, edges=[]),
        inputs=[PortSpec(name="in", type="dataframe")],
        outputs=[PortSpec(name="out", type="ndarray")],
    )


def _branch(
    *,
    branch_id: str,
    columns: list[str],
    steps: list[tuple[str, dict[str, Any]]],
) -> Node:
    """Build a small `pipeline` composite for one ColumnTransformer branch."""
    if len(steps) == 1:
        qualname, params = steps[0]
        return Node(
            id=branch_id,
            kind="transformer",
            component=_component_ref(qualname),
            params={
                **{k: LiteralValue(value=v) for k, v in params.items()},
                "columns": LiteralValue(value=columns),
            },
        )

    inner_nodes: list[Node] = []
    for i, (qualname, params) in enumerate(steps):
        sid = f"{branch_id}_step_{i}"
        inner_nodes.append(
            Node(
                id=sid,
                kind="transformer",
                component=_component_ref(qualname),
                params={k: LiteralValue(value=v) for k, v in params.items()},
            )
        )

    return Node(
        id=branch_id,
        kind="pipeline",
        children=Graph(id=f"{branch_id}_children", nodes=inner_nodes, edges=[]),
        params={"columns": LiteralValue(value=columns)},
    )


def _default_numeric_steps() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("sklearn.impute.SimpleImputer", {"strategy": "mean"}),
        ("sklearn.preprocessing.StandardScaler", {}),
    ]


def _default_categorical_steps() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("sklearn.impute.SimpleImputer", {"strategy": "most_frequent"}),
        ("sklearn.preprocessing.OneHotEncoder", {"handle_unknown": "ignore"}),
    ]
