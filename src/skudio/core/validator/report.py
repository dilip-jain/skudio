"""Validation issue and report models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Severity = Literal["error", "warning", "info"]


class ValidationIssue(BaseModel):
    """Issue."""
    model_config = ConfigDict(extra="forbid")

    severity: Severity
    code: str
    path: str
    message: str


class ValidationReport(BaseModel):
    """Aggregated report."""
    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssue] = []

    @property
    def ok(self) -> bool:
        """True when no `error`-severity issues."""
        return not any(i.severity == "error" for i in self.issues)

    def errors(self) -> list[ValidationIssue]:
        """Return only the `error`-severity issues."""
        return [i for i in self.issues if i.severity == "error"]
