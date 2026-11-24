"""Test: Validator."""

from skudio.core.ir import (
    ComponentRef,
    Edge,
    Graph,
    LiteralValue,
    Node,
    NodeRef,
    PortSpec,
    Reference,
    SearchSpace,
    ChoiceSpace,
)

from skudio.core.validator import validate


def _linear(node_ids: list[str], *, port_type: str = "dataframe") -> Graph:
    nodes = []
    for i, nid in enumerate(node_ids):
        kind = "dataset"
        if i > 0:
            kind = "estimator"
        if i == len(node_ids) - 1:
            kind = "transformer"

        component = None
        if i > 0:
            component = ComponentRef(provider="sklearn", qualname="sklearn.dummy.Dummy")

        nodes.append(
            Node(
                id=nid,
                kind=kind,
                component=component,
                inputs=[] if i == 0 else [
                    PortSpec(name="in", type=port_type)  # type: ignore[arg-type]
                ],
                outputs=[
                    PortSpec(name="out", type=port_type)  # type: ignore[arg-type]
                ] if i < len(node_ids) - 1 else [],
            )
        )

    edges = [
        Edge(
            src=NodeRef(node_id=node_ids[i], port="out"),
            dst=NodeRef(node_id=node_ids[i + 1], port="in"),
        )
        for i in range(len(node_ids) - 1)
    ]
    return Graph(id="g", name="t", nodes=nodes, edges=edges)

def test_ok_graph_has_no_errors() -> None:
    """Test: Graph Errors."""
    r = validate(_linear(["a", "b", "c"]))
    assert r.ok, r.issues

def test_duplicate_ids_flagged() -> None:
    """Test: Duplicate ids flagged."""
    g = _linear(["a", "a"])
    r = validate(g)
    assert any(i.code == "graph.duplicate_node_id" for i in r.issues)


def test_incompatible_ports_flagged() -> None:
    """Test: Incompatible ports flagged."""
    a = Node(id="a", kind="dataset", outputs=[PortSpec(name="out", type="predictions")])
    b = Node(
        id="b",
        kind="estimator",
        component=ComponentRef(provider="sklearn", qualname="sklearn.dummy.Dummy"),
        inputs=[PortSpec(name="in", type="dataframe")],
    )
    g = Graph(
        id="g",
        name="t",
        nodes=[a, b],
        edges=[Edge(src=NodeRef(node_id="a", port="out"), dst=NodeRef(node_id="b", port="in"))],
    )
    r = validate(g)
    assert any(i.code == "ports.incompatible" for i in r.issues)

def test_search_space_outside_scope_is_error() -> None:
    """Test: Search space outside scope is error."""
    n = Node(
        id="n",
        kind="estimator",
        component=ComponentRef(provider="sklearn", qualname="sklearn.dummy.Dummy"),
        params={"C": SearchSpace(space=ChoiceSpace(choices=[0.1, 1.0]))},
    )
    r = validate(Graph(id="g", name="t", nodes=[n]))
    assert any(i.code == "search.out_of_scope" for i in r.issues)

def test_bad_reference_flagged() -> None:
    """Test: Bad reference flagged."""
    n = Node(
        id="n",
        kind="estimator",
        component=ComponentRef(provider="sklearn", qualname="sklearn.dummy.Dummy"),
        params={"col": Reference(ref="node:nope.out")},
    )

    r = validate(Graph(id="g", name="t", nodes=[n]))
    assert any(i.code == "refs.unknown_node" for i in r.issues)


def test_literal_params() -> None:
    """Test: use of literal parameters."""
    n = Node(
        id="m",
        kind="estimator",
        component=ComponentRef(provider="sklearn", qualname="sklearn.dummy.Dummy"),
        params={"C": LiteralValue(value=1.0)}
    )

    r = validate(Graph(id="g", name="t", nodes=[n]))
    assert r.ok
