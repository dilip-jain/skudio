"""Visualization."""

from __future__ import annotations

from skudio.visualization.eda import (
    class_balance,
    correlation_heatmap,
    high_correlation_pairs,
    histogram,
    missing_bar,
    target_vs_feature,
)
from skudio.visualization.importance import permutation_importance_figure
from skudio.visualization.result import (
    confusion_matrix_figure,
    pr_curves_figure,
    prediction_vs_actual_figure,
    residuals_figure,
    roc_curves_figure,
)

__all__ = [
    "class_balance",
    "confusion_matrix_figure",
    "correlation_heatmap",
    "high_correlation_pairs",
    "histogram",
    "missing_bar",
    "target_vs_feature",
    "permutation_importance_figure",
    "pr_curves_figure",
    "prediction_vs_actual_figure",
    "residuals_figure",
    "roc_curves_figure",
]
