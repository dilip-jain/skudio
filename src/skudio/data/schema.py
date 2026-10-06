"""Dataset schema inference from a pandas DataFrame."""

from __future__ import annotations

import pandas as pd

from skudio.core.ir.refs import ColumnSchema, DatasetSchema, Role


def infer_schema(df: pd.DataFrame) -> DatasetSchema:
    """Infer a `DatasetSchema` from a pandas DataFrame."""
    columns = [
        ColumnSchema(name=str(c), dtype=str(df[c].dtype), role=_infer_role(df[c]))
        for c in df.columns
    ]
    return DatasetSchema(columns=columns)


def _infer_role(series: pd.Series) -> Role:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        return "text" if unique_ratio > 0.5 else "categorical"
    return "unknown"
