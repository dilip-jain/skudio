"""Test: EDA."""

from __future__ import annotations

import pytest
import pandas as pd
from matplotlib.figure import Figure

from skudio.visualization import eda

# `Figure.axes` is a list[Axes]; astroid infers ndarray and flags subscripting.
# pylint: disable=unsubscriptable-object


def _titanic_like() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [22, 38, 26, 35, None, 54, 2, 27, 14, 4],
            "fare": [7.25, 71.28, 7.92, 53.10, 8.05, 51.86, 21.08, 11.13, 30.07, 16.70],
            "sex": ["m", "f", "f", "f", "m", "m", "m", "f", "f", "f"],
            "survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        }
    )

def test_histogram_numeric_returns_figure_with_hist_patches() -> None:
    """Test: Histogram numeric returns figure with hist patches."""
    fig = eda.histogram(_titanic_like(), "age", bins=5)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Distribution of age"
    assert ax.get_xlabel() == "age"
    assert ax.get_ylabel() == "count"
    assert len(ax.patches) == 5

def test_histogram_falls_back_to_value_counts_for_categorical() -> None:
    """Test: Histogram fallback to value_counts for categorical."""
    fig = eda.histogram(_titanic_like(), "sex")
    ax = fig.axes[0]
    assert ax.get_title() == "Value counts for sex"
    assert len(ax.patches) == 2  # "m" and "f"

def test_histogram_missing_column_raises() -> None:
    """Test: Missing column histogram raise error."""
    with pytest.raises(ValueError, match="not in dataframe"):
        eda.histogram(_titanic_like(), "does_not_exist")

def test_class_balance_returns_bar_per_target_value() -> None:
    """Test: Class balance returns bar per target value."""
    fig = eda.class_balance(_titanic_like(), "survived")
    ax = fig.axes[0]
    assert ax.get_title() == "Class balance for survived"
    assert ax.get_xlabel() == "survived"
    assert len(ax.patches) == 2  # 0 and 1

def test_missing_bar_reports_null_counts() -> None:
    """Test: Missing values bar report."""
    fig = eda.missing_bar(_titanic_like())
    ax = fig.axes[0]
    assert ax.get_title() == "Missing values per column"

    # Bars are Rectangle patches; `get_height` returns the count for each.
    heights = sorted(p.get_height() for p in ax.patches)  # type: ignore[attr-defined]
    assert heights[-1] == 1
    assert heights[0] == 0

def test_correlation_heatmap_contains_image_and_labels() -> None:
    """Test: Correlation heatmap."""
    fig = eda.correlation_heatmap(_titanic_like())
    ax = fig.axes[0]
    assert ax.get_title() == "Correlation matrix (numeric columns)"
    assert len(ax.images) == 1

    numeric_cols = ["age", "fare", "survived"]
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    assert xlabels == numeric_cols

def test_high_correlation_pairs_returns_sorted_pairs() -> None:
    """Test: High correlation pairs returns sorted pairs."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 3, 1, 4, 2],
        }
    )
    pairs = eda.high_correlation_pairs(df, threshold=0.99)
    assert pairs == [("a", "b", 1.0)]

def test_high_correlation_pairs_threshold_filters_weak_correlations() -> None:
    """Test: High correlation pairs threshold filters weak correlations."""
    assert not eda.high_correlation_pairs(_titanic_like(), threshold=0.999)

def test_target_vs_feature_numeric_vs_numeric_produces_scatter() -> None:
    """Test: Target (numeric) vs Feature (numeric)."""
    fig = eda.target_vs_feature(_titanic_like(), target="fare", feature="age")
    ax = fig.axes[0]
    assert ax.get_title() == "fare vs age"
    assert len(ax.collections) == 1

def test_target_vs_feature_numeric_target_categorical_feature_produces_boxplot() -> None:
    """Test: Target (numeric) vs Feature (categorical)."""
    fig = eda.target_vs_feature(_titanic_like(), target="fare", feature="sex")
    ax = fig.axes[0]
    assert ax.get_title() == "fare distribution by sex"
    # boxplot draws lines rather than patches; presence of any line is enough.
    assert len(ax.lines) > 0

def test_target_vs_feature_categorical_vs_categorical_produces_stacked_bar() -> None:
    """Test: Target (categorical) vs Feature (categorical)."""
    df = pd.DataFrame(
        {
            "target": ["a", "b", "a", "b", "a"],
            "feature": ["x", "x", "y", "y", "y"],
        }
    )

    fig = eda.target_vs_feature(df, target="target", feature="feature")
    ax = fig.axes[0]
    assert "composition by feature" in ax.get_title()

    # Legend exists with target classes.
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert labels == {"a", "b"}

def test_target_vs_feature_missing_column_raises() -> None:
    """Test: Target vs Feature missing column."""
    with pytest.raises(ValueError, match="not in dataframe"):
        eda.target_vs_feature(_titanic_like(), target="fare", feature="nope")
