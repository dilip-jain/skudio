"""Test: Results."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from skudio.visualization import results

# `Figure.axes` is a list[Axes]; astroid infers ndarray and flags subscripting.
# pylint: disable=unsubscriptable-object


def test_confusion_matrix_figure_binary() -> None:
    """Test: Confusion Matrix - Binary."""
    fig = results.confusion_matrix_figure([0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1])
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Confusion matrix"
    # ConfusionMatrixDisplay draws a matrix image on the axes.
    assert len(ax.images) >= 1


def test_confusion_matrix_figure_multiclass_with_labels() -> None:
    """Test: Confusion Matrix - Multiclass."""
    fig = results.confusion_matrix_figure(
        ["a", "b", "c", "a", "b"], ["a", "b", "b", "a", "c"], labels=["a", "b", "c"]
    )
    assert isinstance(fig, Figure)


def test_roc_curves_figure_binary_scores() -> None:
    """Test: ROC Curves - Binary."""
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15]
    fig = results.roc_curves_figure(y_true, y_score)
    assert fig.axes[0].get_title() == "ROC curve(s)"


def test_roc_curves_figure_multiclass_matrix() -> None:
    """Test: ROC Curves - Multiclass."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_score = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    fig = results.roc_curves_figure(y_true, y_score, classes=[0, 1, 2])

    # Legend should have one entry per class.
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert all(str(c) in " ".join(labels) for c in ["0", "1", "2"])


def test_pr_curves_figure_binary() -> None:
    """Test: PR Curves - Binary."""
    fig = results.pr_curves_figure([0, 1, 0, 1, 1, 0], [0.1, 0.9, 0.2, 0.85, 0.75, 0.3])
    assert fig.axes[0].get_title() == "Precision-recall curve(s)"


def test_prediction_vs_actual_figure_has_scatter_and_reference_line() -> None:
    """Test: Prediction vs Actual - Scatter."""
    fig = results.prediction_vs_actual_figure([1.0, 2.0, 3.0, 4.0], [1.1, 1.8, 3.5, 3.9])
    ax = fig.axes[0]
    assert ax.get_title() == "Predicted vs actual"

    # One scatter collection + one reference line.
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1


def test_residuals_figure_has_zero_line() -> None:
    """Test: Residuals."""
    fig = results.residuals_figure([1.0, 2.0, 3.0], [1.5, 1.9, 2.7])
    ax = fig.axes[0]
    assert ax.get_title() == "Residuals vs predicted"

    # zero-line drawn via axhline shows as a line on the axes.
    assert any(line.get_linestyle() == "-" for line in ax.lines)
