"""Test: Component Registry."""

from __future__ import annotations

import pytest

from skudio.components.registry import Registry, default_registry, reset_default_registry
from skudio.core.errors import RegistryError


def setup_function(_: object) -> None:
    """Registry Reset."""
    reset_default_registry()


def test_default_registry_lists_expected_estimators() -> None:
    """Test: Default registry must contain expected estimators."""
    reg = default_registry()
    qualnames = {m.qualname for m in reg.list()}
    assert "sklearn.linear_model.LogisticRegression" in qualnames
    assert "sklearn.ensemble.RandomForestClassifier" in qualnames
    assert "sklearn.compose.ColumnTransformer" in qualnames
    assert "sklearn.model_selection.GridSearchCV" in qualnames


def test_duplicate_registration_raises() -> None:
    """Test: Duplicate registrations are prohibited."""
    reg = Registry()
    meta = default_registry().list()[0]
    reg.register(meta)
    with pytest.raises(RegistryError):
        reg.register(meta)


def test_unknown_lookup_raises() -> None:
    """Test: Unknown registrations are prohibited."""
    reg = Registry()
    with pytest.raises(RegistryError):
        reg.get("nope.does.not.Exist")


def tet_resolve_class_imports_real_sklearn_class() -> None:
    """Test: Resolve sklearn imports."""
    reg = default_registry()
    cls = reg.resolve_class("sklearn.linear_model.LogisticRegression")
    assert cls.__name__ == "LogisticRegression"


def test_resolve_class_rejects_unregistered_qualname() -> None:
    """Test: Reject unregistered class."""
    reg = default_registry()
    with pytest.raises(RegistryError):
        reg.resolve_class("os.system")
    with pytest.raises(RegistryError):
        reg.resolve_class("subprocess.Popen")


def test_sklearn_provider_stamps_version() -> None:
    """Test: Sklearn provider version."""
    reg = default_registry()
    meta = reg.get("sklearn.ensemble.RandomForestClassifier")
    assert meta.version, "component version should be stamped at registration"


def test_expanded_transformer_manifest_includes_new_entries() -> None:
    """Test: Expanded transformer manifest includes new entries."""
    reg = default_registry()
    qualnames = {m.qualname for m in reg.list()}
    for expected in [
        "sklearn.preprocessing.PolynomialFeatures",
        "sklearn.preprocessing.KBinsDiscretizer",
        "sklearn.preprocessing.PowerTransformer",
        "sklearn.preprocessing.QuantileTransformer",
        "sklearn.impute.KNNImputer",
        "sklearn.feature_selection.SelectKBest",
        "sklearn.feature_selection.VarianceThreshold",
    ]:
        assert expected in qualnames, f"missing manifest entry: {expected}"
        assert reg.resolve_class(expected).__name__ == expected.rsplit(".", 1)[-1]
