"""Compact dataset inspection for the UI."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from skudio.core.ir.refs import DatasetSchema
from skudio.data.schema import infer_schema


class ColumnInspection(BaseModel):
    """Per-column stats surfaced in the Streamlit inspect table."""
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    role: str
    non_null: int
    null: int
    unique: int
    sample: list[Any] = []


class DatasetInspection(BaseModel):
    """Compact snapshot of a dataset used by the UI."""
    model_config = ConfigDict(extra="forbid")

    n_rows: int
    n_cols: int
    schema_: DatasetSchema
    columns: list[ColumnInspection]


def inspect_dataframe(df: pd.DataFrame, *, sample_size: int = 5) -> DatasetInspection:
    """Return a per-column inspection with dtype, role, counts, and a sample."""
    schema = infer_schema(df)
    role_by_name = {c.name: c.role for c in schema.columns}

    columns = []
    for name in df.columns:
        series = df[name]
        columns.append(
            ColumnInspection(
                name=str(name),
                dtype=str(series.dtype),
                role=role_by_name.get(str(name), "unknown"),
                non_null=int(series.notna().sum()),
                null=int(series.isna().sum()),
                unique=int(series.nunique(dropna=True)),
                sample=[_to_python(v) for v in series.dropna().head(sample_size).tolist()],
            )
        )

    return DatasetInspection(
        n_rows=len(df), n_cols=len(df.columns), schema_=schema, columns=columns
    )


def _to_python(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return str(v)
    return v
