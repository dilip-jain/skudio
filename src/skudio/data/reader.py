"""CSV -> pandas DataFrame + inferred schema + content hash."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import IO

import pandas as pd

from skudio.core.ir.refs import DatasetSchema
from skudio.data.schema import infer_schema


def load_csv(path: str | Path | IO[bytes]) -> pd.DataFrame:
    """Read a CSV at `path` into a pandas DataFrame."""
    return pd.read_csv(path)


def load_csv_with_schema(path: str | Path | IO[bytes]) -> tuple[pd.DataFrame, DatasetSchema]:
    """Read a CSV and return `(df, inferred_schema)`."""
    df = load_csv(path)
    return df, infer_schema(df)


def dataset_hash(path: str | Path) -> str:
    """Return a SHA-256 of the file contents, streamed in 1 MB chunks."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
