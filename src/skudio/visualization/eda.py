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


def missing_bar(df: pd.DataFrame) -> Figure:
    """Return a bar chart of null counts per column, descending."""
    fig = Figure(figsize=(6, 3.5))
    ax = cast(Axes, fig.subplots())
    missing = df.isna().sum().sort_values(ascending=False)
    ax.bar(missing.index.astype(str).tolist(), missing.to_numpy())
    ax.set_ylabel("missing")
    ax.set_title("Missing values per column")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame) -> Figure:
    """Return a heatmap of pairwise Pearson correlations of numeric columns.
    Non-numeric columns are silently dropped. If fewer than two numeric
    columns remain the returned Figure has an empty axes.
    """
    fig = Figure(figsize=(6, 5))
    ax = cast(Axes, fig.subplots())
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)
    labels = list(corr.columns.astype(str))

    im = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Correlation matrix (numeric columns)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return fig


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"column {column!r} not in dataframe")
