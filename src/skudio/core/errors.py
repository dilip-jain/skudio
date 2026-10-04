"""Exceptions raised by skudio core."""

from __future__ import annotations

from typing import Any


class SkudioError(Exception):
    """Base for every domain error raised by skudio."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Create the error with a message and optional IR path and context."""
        super().__init__(message)
        self.message = message
        self.path = path
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        """Render the message, prefixed with the IR path when known."""
        if self.path:
            return f"[{self.path}] {self.message}"
        return self.message


class IRError(SkudioError):
    """IR construction, load, or migration failed."""


class CompilerError(SkudioError):
    """The compiler could not produce an estimator"""


class CodegenError(SkudioError):
    """Python codegen failed."""


class RegistryError(SkudioError):
    """Unknown or duplicate component registration."""
