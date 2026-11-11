"""Search-space scope rules: `SearchSpace` values need a search-wrapping context."""

from __future__ import annotations

from skudio.core.ir.graph import Graph
from skudio.core.ir.params import SearchSpace
from skudio.core.validator.report import ValidationIssue


def check_search_space(graph: Graph) -> list[ValidationIssue]:
    """Flag `SearchSpace` values that appear outside a `search`-node scope."""
    return _walk(graph, in_search_scope=False, path="")


def _walk(graph: Graph, *, in_search_scope: bool, path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for node in graph.nodes:
        scope = in_search_scope or node.kind == "search"
        
        for name, pv in node.params.items():
            if isinstance(pv, SearchSpace) and not scope:
                issues.append(ValidationIssue(
                    severity="error",
                    code="search.out_of_scope",
                    path=f"{path}nodes[{node.id}].params.{name}",
                    message="SearchSpace values used outside a `search` scope."
                ))

        if node.children is not None:
            issues.extend(_walk(
                node.children,
                in_search_scope=scope,
                path=path + f"nodes[{node.id}].children.",
            ))

    return issues
