"""Post-training result plots.

Every builder takes precomputed arrays {`y_true`, `y_pred`, optional `y_score`, optional `classes`}
so the same figures render from a persisited `TrainingResult` without touching the fitted estimator.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

# `Figure.subplots()` returns `Axes | ndarray`; pylint's astroid picks the ndarray
# branch and flags every method call. All usage here is single-Axes.
# pylint: disable=no-member


def confusion_matrix_figure(
    y_true: Any,
    y_pred: Any,
    *,
    labels: list[Any] | None = None,
    normalize: Literal["true", "pred", "all"] | None = None,
) -> Figure:
    """Return sklearn's `ConfusionMatrixDisplay` rendered on a fresh figure."""
    fig = Figure(figsize=(5, 4.5))
    ax = cast(Axes, fig.subplots())

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize=normalize, ax=ax, colorbar=True
    )

    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def roc_curves_figure(y_true: Any, y_score: Any, classes: list[Any] | None = None) -> Figure:
    """Return ROC curve(s) - one line per class for multiclass (one-vs-rest)."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    fig = Figure(figsize=(5, 4.5))
    ax = cast(Axes, fig.subplots())

    if y_score_arr.ndim == 1 or y_score_arr.shape[1] == 1:
        y_score_1d = y_score_arr.ravel()
        RocCurveDisplay.from_predictions(y_true_arr, y_score_1d, ax=ax)
    else:
        cls_labels = classes or list(range(y_score_arr.shape[1]))
        for i, cls in enumerate(cls_labels):
            RocCurveDisplay.from_predictions(
                (y_true_arr == cls).astype(int), y_score_arr[:, i], ax=ax, name=str(cls)
            )

    ax.set_title("ROC curve(s)")
    fig.tight_layout()
    return fig


def pr_curves_figure(y_true: Any, y_score: Any, classes: list[Any] | None = None) -> Figure:
    """Return precision-recall curve(s)."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    fig = Figure(figsize=(5, 4.5))
    ax = cast(Axes, fig.subplots())

    if y_score_arr.ndim == 1 or y_score_arr.shape[1] == 1:
        PrecisionRecallDisplay.from_predictions(y_true_arr, y_score_arr.ravel(), ax=ax)
    else:
        cls_labels = classes or list(range(y_score_arr.shape[1]))
        for i, cls in enumerate(cls_labels):
            PrecisionRecallDisplay.from_predictions(
                (y_true_arr == cls).astype(int), y_score_arr[:, i], ax=ax, name=str(cls)
            )

    ax.set_title("Precision-recall curve(s)")
    fig.tight_layout()
    return fig


def prediction_vs_actual_figure(y_true: Any, y_pred: Any) -> Figure:
    """Regression: scatter of predicted vs actual with y=x reference line."""
    fig = Figure(figsize=(5, 4.5))
    ax = cast(Axes, fig.subplots())

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    lo = float(min(y_true_arr.min(), y_pred_arr.min()))
    hi = float(max(y_true_arr.max(), y_pred_arr.max()))

    ax.scatter(y_true_arr, y_pred_arr, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs actual")
    fig.tight_layout()
    return fig


def residuals_figure(y_true: Any, y_pred: Any) -> Figure:
    """Regression: scatter of residuals vs predicted with zero-line."""
    fig = Figure(figsize=(5, 4.5))
    ax = cast(Axes, fig.subplots())

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    ax.scatter(y_pred_arr, y_true_arr - y_pred_arr, alpha=0.5)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("predicted")
    ax.set_ylabel("residual (actual - predicted)")
    ax.set_title("Residuals vs predicted")
    fig.tight_layout()
    return fig
