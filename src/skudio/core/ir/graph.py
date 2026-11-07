"""Graph Root"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from skudio.core.ir.edges import Edge
from skudio.core.ir.nodes import Node
from skudio.core.ir.refs import DatasetRef

IR_VERSION = "0.1"


class Graph(BaseModel):
    """A pipeline IR graph."""
    model_config = ConfigDict(extra="forbid")

    ir_version: Literal["0.1"] = IR_VERSION
    id: str
    name: str = ""
    nodes: list[Node] = []
    edges: list[Edge] = []
    dataset_ref: DatasetRef | None = None
    random_state: int | None = None
    metadata: dict[str, Any] = {}


    def get_adjacency_list(self) -> dict[str, set[str]]:
        """Build adjacency list for the graph."""
        adj: dict[str, set[str]] = {n.id: set() for n in self.nodes}
        for edge in self.edges:
            if edge.src.node_id in adj and edge.dst.node_id in adj:
                adj[edge.src.node_id].add(edge.dst.node_id)
        return adj

# Node reference Graph via forward ref
Node.model_rebuild()
Graph.model_rebuild()
