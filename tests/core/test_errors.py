"""Test: Errors."""

from __future__ import annotations

from skudio.core.errors import (
    SkudioError,
    CodegenError,
    CompilerError,
    IRError,
    RegistryError,
    ValidationError,
)


def test_subclasses_inherit_from_base() -> None:
    """Test: Error classes must inherit from base SkudioError."""
    for cls in (IRError, CompilerError, CodegenError, RegistryError, ValidationError):
        assert issubclass(cls, SkudioError)


def test_str_without_path() -> None:
    """Test: Error str without path."""
    assert str(IRError("test")) == "test"


def test_str_with_path() -> None:
    """Test: Error str with path."""
    assert str(IRError("test", path="graph.nodes[3]")) == "[graph.nodes[3]] test"


def test_context_defaults_to_empty_dict() -> None:
    """Test: Error context must default to empty dictionary."""
    assert ValidationError("test").context == {}


def test_context_carries_provided_data() -> None:
    """Test: Error context must carry the provided data."""
    err = RegistryError("test", context={"qualname": "sklearn.svm.SVC"})
    assert err.context == {"qualname": "sklearn.svm.SVC"}
