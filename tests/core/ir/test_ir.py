"""Test: IR"""

from skudio.core.hashing import ir_hash
from skudio.core.ir import (
    ColumnSchema,
    ComponentRef,
    DatasetRef,
    DatasetSchema,
    Edge,
    Graph,
    LiteralValue,
    Node,
    NodeRef,
    PortSpec,
    canonical_json,
)


def _iris_graph() -> Graph:
    """Test: Graph."""
    dataset = Node(
        id="data",
        kind="dataset",
        outputs=[PortSpec(name="out", type="dataframe")],
    )
    model = Node(
        id="model",
        kind="estimator",
        component=ComponentRef(
            provider="sklearn",
            qualname="sklearn.ensemble.RandomForestClassifier",
        ),
        params={"n_estimators": LiteralValue(value=50)},
        inputs=[PortSpec(name="in", type="dataframe")]
    )
    return Graph(
        id="g",
        name="iris",
        nodes=[dataset, model],
        edges=[
            Edge(src=NodeRef(node_id="data", port="out"), dst=NodeRef(node_id="model", port="in"))
        ],
        dataset_ref=DatasetRef(
            id="iris",
            kind="csv",
            path="datasets/iris/data.csv",
            schema_=DatasetSchema(
                columns=[
                    ColumnSchema(name="sepal_length", dtype="float64", role="numeric"),
                    ColumnSchema(name="species", dtype="object", role="categorical"),
                ]
            ),
        ),
    )

def test_graph_roundtrips_through_json() -> None:
    """Test: Graph roundtrips through JSON."""
    g = _iris_graph()
    encoded = g.model_dump_json()
    reloaded = Graph.model_validate_json(encoded)
    assert reloaded.model_dump() == g.model_dump()

def test_canonical_json_is_stable_across_key_orders() -> None:
    """Test: Canonical JSON must be stable across key orders."""
    g1 = _iris_graph()
    g2 = Graph.model_validate_json(g1.model_dump_json())
    assert canonical_json(g1) == canonical_json(g2)

def test_ir_hash_changes_when_component_version_changes() -> None:
    """Test: Hash must change on component version change."""
    g1 = _iris_graph()
    g2 = _iris_graph()
    assert g2.nodes[1].component is not None
    g2.nodes[1].component.version = "1.5"
    assert ir_hash(g1) != ir_hash(g2)
