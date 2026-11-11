"""IR validator: rule-based, structured error output."""

from __future__ import annotations

from skudio.core.ir.graph import Graph
from skudio.core.validator.graph_rules import check_graph_shape
from skudio.core.validator.port_rules import check_port_compatibility
from skudio.core.validator.ref_rules import check_references
from skudio.core.validator.search_rules import check_search_space
from skudio.core.validator.report import Severity, ValidationIssue, ValidationReport


def validate(graph: Graph) -> ValidationReport:
    """Run every rule and aggregate issues."""
    issues: list[ValidationIssue] = []
    issues.extend(check_graph_shape(graph))
    issues.extend(check_port_compatibility(graph))
    issues.extend(check_references(graph))
    issues.extend(check_search_space(graph))
    return ValidationReport(issues=issues)


__all__ = ["Severity", "ValidationIssue", "ValidationReport", "validate"]
