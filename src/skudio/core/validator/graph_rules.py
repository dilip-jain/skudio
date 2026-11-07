"""Graph-shape rules: unique ids, edge references, acyclicity."""

from __future__ import annotations

from skudio.core.ir.graph import Graph
from skudio.core.ir.nodes import Node
from skudio.core.validator.report import ValidationIssue


def check_graph_shape(graph: Graph) -> list[ValidationIssue]:
    """Check unique ids, edge endpoints, and acyclicity across nested composites."""
    issues: list[ValidationIssue] = []
    issues.extend(_check_unique_ids(graph, path=""))
    issues.extend(_check_edges_reference_existing(graph, path=""))
    issues.extend(_check_acyclic(graph, path=""))
    for node in graph.nodes:
        if node.children is not None:
            issues.extend(check_graph_shape(node.children))
    return issues


def _check_unique_ids(graph: Graph, path: str) -> list[ValidationIssue]:
    seen: dict[str, int] = {}
    issues: list[ValidationIssue] = []
    for i, node in enumerate(graph.nodes):
        if node.id in seen:
            issues.append(ValidationIssue(
                severity="error",
                code="graph.duplicate_node_id",
                path=f"{path}nodes[{i}].id",
                message=f"duplicate node id: {node.id}",
            ))
        seen[node.id] = i
    return issues


def _check_edges_reference_existing(graph: Graph, path: str) -> list[ValidationIssue]:
    id_to_node: dict[str, Node] = {n.id: n for n in graph.nodes}

    issues: list[ValidationIssue] = []
    for i, edge in enumerate(graph.edges):
        for side, ref in (("src", edge.src), ("dst", edge.dst)):

            node = id_to_node.get(ref.node_id)
            if node is None:
                issues.append(ValidationIssue(
                    severity="error",
                    code="graph.unknown_node_in_edge",
                    path=f"{path}edges[{i}].{side}",
                    message=f"edge references missing node {ref.node_id!r}",
                ))
                continue

            ports = node.outputs if side == "src" else node.inputs
            if ref.port and not any(p.name == ref.port for p in ports):
                issues.append(ValidationIssue(
                    severity="error",
                    code="graph.unknown_port_in_edge",
                    path=f"{path}edges[{i}].{side}",
                    message=f"node {ref.node_id!r} has no {side} port {ref.port!r}",
                ))

    return issues


def _check_acyclic(graph: Graph, path: str) -> list[ValidationIssue]:
    adj: dict[str, set[str]] = graph.get_adjacency_list()

    color: dict[str, int] = dict.fromkeys(adj, 0)  # 0=white 1=grey 2=black
    issues: list[ValidationIssue] = []

    def visit(u: str) -> bool:
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1:
                return True
            if color[v] == 0 and visit(v):
                return True
        color[u] = 2
        return False

    for n in adj:
        if color[n] == 0 and visit(n):
            issues.append(ValidationIssue(
                severity="error",
                code="graph.cycle",
                path=f"{path}edges",
                message="graph contains a cycle",
            ))
            break

    return issues


