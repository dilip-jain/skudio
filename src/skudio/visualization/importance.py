"""Feature-importance figures using sklearn's built-in inspection routines."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# `Figure.subplots()` returns `Axes | ndarray`; pylint's astroid picks the ndarray
# branch and flags every method call. All usage here is single-Axes.
# pylint: disable=no-member


def permutation_importance_figure(
    estimator: Any,
    x_test: Any,
    y_test: Any,
    *,
    n_repeats: int = 5,
    random_state: int = 0,
    max_features: int = 20,
) -> Figure:
    """Return sklearn's `permutation_importance` as a horizontal bar chart.

    Bounded at `max_features` bars so the chart stays legible on wide
    schemas. The importance means and stds are reused for error bars.
    """
    from matplotlib.figure import Figure  # pylint: disable=import-outside-toplevel
    from sklearn.inspection import permutation_importance  # pylint: disable=import-outside-toplevel

    result = permutation_importance(
        estimator, x_test, y_test, n_repeats=n_repeats, random_state=random_state)

    means = np.asarray(result.importances_mean)  # type: ignore[union-attr]
    stds = np.asarray(result.importances_std)  # type: ignore[union-attr]
    feature_names = _feature_names(estimator, x_test, means.shape[0])

    # Sort by |mean| desc, take the top `max_features`.
    order = np.argsort(np.abs(means))[::-1][:max_features]
    names = [feature_names[i] for i in order][::-1]
    heights = means[order][::-1]
    errors = stds[order][::-1]

    fig = Figure(figsize=(6, max(3.0, 0.3 * len(names) + 1.5)))
    ax = cast(Axes, fig.subplots())
    ax.barh(names, heights, xerr=errors, color="#4c78a8", edgecolor="white")
    ax.axvline(0, color="k", linewidth=1)
    ax.set_xlabel("Mean decrease in score")
    ax.set_title(f"Permutation importance (top {len(names)})")
    fig.tight_layout()
    return fig


def _feature_names(estimator: Any, x_test: Any, n_features: int) -> list[str]:
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]

    cols = getattr(x_test, "columns", None)
    if cols is not None:
        return [str(c) for c in cols]

    return [f"f{i}" for i in range(n_features)]
