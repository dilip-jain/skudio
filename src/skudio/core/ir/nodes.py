"""IR nodes: kind literals, ui hints, and the Node model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, model_validator

from skudio.core.ir.params import ParamValue
from skudio.core.ir.ports import PortSpec
from skudio.core.ir.refs import ComponentRef

if TYPE_CHECKING:
    from skudio.core.ir.graph import Graph

NodeKind = Literal[
    "dataset",
    "split",
    "transformer",
    "estimator",
    "column_transformer",
    "pipeline",
    "feature_union",
    "search",
]

NODE_KINDS: tuple[NodeKind, ...] = get_args(NodeKind)
_COMPOSITE_KINDS: frozenset[str] = frozenset(
    {"column_transformer", "pipeline", "feature_union", "search"}
)


class UIHints(BaseModel):
    """Layout hints for the UI. Never read by the core."""
    model_config = ConfigDict(extra="allow")

    x: float | None = None
    y: float | None = None
    color: str | None = None


class Node(BaseModel):
    """Node in the IR graph."""
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: NodeKind
    label: str = ""
    component: ComponentRef | None = None
    params: dict[str, ParamValue] = {}
    inputs: list[PortSpec] = []
    outputs: list[PortSpec] = []
    children: Graph | None = None
    ui: UIHints = UIHints()

    @model_validator(mode="after")
    def _children_only_on_composites(self) -> Node:
        if self.children is not None and self.kind not in _COMPOSITE_KINDS:
            raise ValueError(f"Kind={self.kind} may not have children")
        return self

    def is_composite(self) -> bool:
        """Return True if this node kind can hold a `children` subgraph."""
        return self.kind in _COMPOSITE_KINDS

# Imported lazily by graph after Graph is defined
def _model_rebuild_hook() -> Any:
    Node.model_rebuild()
    return Node
