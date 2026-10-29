"""Exploratory-data-analysis chart builders."""

from __future__ import annotations

from typing import cast

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# `Figure.subplots()` returns `Axes | ndarray`; pylint's astroid picks the ndarray
# branch and flags every method call. All usage here is single-Axes.
# pylint: disable=no-member


def histogram(df: pd.DataFrame, column: str, *, bins: int = 30) -> Figure:
    """Return a histogram of one numeric column.
    Falls back to a bar-of-value-counts for non-numeric columns.
    """
    _require_column(df, column)
    fig = Figure(figsize=(6, 3.5))
    ax = cast(Axes, fig.subplots())
    series = df[column]

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        ax.hist(series.dropna(), bins=bins, edgecolor="white")
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.set_title(f"Distribution of {column}")
    else:
        counts = series.astype(str).value_counts(dropna=False)
        ax.bar(counts.index.tolist(), counts.to_numpy())
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.set_title(f"Value counts for {column}")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def class_balance(df: pd.DataFrame, target: str) -> Figure:
    """Return a bar chart of the target-column value counts."""
    _require_column(df, target)
    fig = Figure(figsize=(6, 3.5))
    ax = cast(Axes, fig.subplots())
    counts = df[target].astype(str).value_counts(dropna=False)
    ax.bar(counts.index.tolist(), counts.to_numpy())
    ax.set_xlabel(target)
    ax.set_ylabel("count")
    ax.set_title(f"Class balance for {target}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"column {column!r} not in dataframe")
