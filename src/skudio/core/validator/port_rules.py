"""Port-type compatibility rule."""

from __future__ import annotations

from skudio.core.ir.graph import Graph
from skudio.core.ir.nodes import Node
from skudio.core.ir.ports import PortType
from skudio.core.validator.report import ValidationIssue

# Symmetric compatibility. dataframe/ndarray/series are numeric-tabular and
# auto-adapt; fitted objects, predictions, metrics each stay in their lane.
_TABULAR: frozenset[str] = frozenset({"dataframe", "ndarray", "series"})


def _compatible(a: PortType, b: PortType) -> bool:
    if a == b:
        return True
    return a in _TABULAR and b in _TABULAR


def check_port_compatibility(graph: Graph) -> list[ValidationIssue]:
    """Flag edges whose src and dst port types are not mutually compatible."""
    issues: list[ValidationIssue] = []
    _check(graph, "", issues)
    return issues


def _check(graph: Graph, path: str, issues: list[ValidationIssue]) -> None:
    id_to_node: dict[str, Node] = {n.id: n for n in graph.nodes}
    for i, edge in enumerate(graph.edges):

        src = id_to_node.get(edge.src.node_id)
        dst = id_to_node.get(edge.dst.node_id)
        if src is None or dst is None:
            # graph-rules will flag this
            continue

        src_port = next((p for p in src.outputs if p.name == edge.src.port), None)
        dst_port = next((p for p in dst.inputs if p.name == edge.dst.port), None)
        if src_port is None or dst_port is None:
            continue


        if not _compatible(src_port.type, dst_port.type):
            issues.append(ValidationIssue(
                severity="error",
                code="ports.incompatible",
                path=f"{path}edges[{i}]",
                message=(
                    f"{src.id}.{src_port.name}:{src_port.type} -> "
                    f"{dst.id}.{dst_port.name}:{dst_port.type} is not compatible"
                ),
            ))

    for node in graph.nodes:
        if node.children is not None:
            _check(node.children, path + f"nodes[{node.id}].children.", issues)
