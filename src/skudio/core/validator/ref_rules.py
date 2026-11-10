"""Reference-value rules: dataset columns and upstream ports must exist."""

from __future__ import annotations

from skudio.core.ir.graph import Graph
from skudio.core.ir.params import Reference
from skudio.core.validator.report import ValidationIssue


def check_references(graph: Graph) -> list[ValidationIssue]:
    """Flag `Reference` params whose target dataset column or node is missing."""
    issues: list[ValidationIssue] = []
    dataset_columns = _dataset_columns(graph)

    node_ids = {n.id for n in graph.nodes}
    for node in graph.nodes:
        for name, pv in node.params.items():
            if not isinstance(pv, Reference):
                continue

            issue = _check_ref(
                pv,
                node_id=node.id,
                param_name=name,
                dataset_columns=dataset_columns,
                node_ids=node_ids,
            )

            if issue:
                issues.append(issue)

        if node.children is not None:
            issues.extend(check_references(node.children))
    return issues


def _dataset_columns(graph: Graph) -> set[str]:
    if graph.dataset_ref is None:
        return set()
    return {c.name for c in graph.dataset_ref.schema_.columns}


def _check_ref(
    ref: Reference,
    *,
    node_id: str,
    param_name: str,
    dataset_columns: set[str],
    node_ids: set[str],
) -> ValidationIssue | None:
    r = ref.ref
    if r.startswith("dataset.column:"):
        col = r.split(":", 1)[1]
        if col not in dataset_columns:
            return ValidationIssue(
                severity="error",
                code="refs.unknown_column",
                path=f"nodes[{node_id}].params.{param_name}",
                message=f"unknown dataset column: {col!r}",
            )

        return None

    elif r.startswith("node:"):
        rest = r.split(":", 1)[1]
        target_node = rest.split(".", 1)[0]
        if target_node not in node_ids:
            return ValidationIssue(
                severity="error",
                code="refs.unknown_node",
                path=f"nodes[{node_id}].params.{param_name}",
                message=f"unknown node reference: {target_node!r}",
            )

        return None

    return ValidationIssue(
        severity="error",
        code="refs.unparseable",
        path=f"nodes[{node_id}].params.{param_name}",
        message=f"reference does not parse: {r!r}",
    )
