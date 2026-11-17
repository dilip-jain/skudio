"""Test: Data reader and inspect."""

from __future__ import annotations

from pathlib import Path

import pytest

from skudio.data.inspect import inspect_dataframe
from skudio.data.reader import dataset_hash, load_csv_with_schema


pytest.importorskip("pandas")


def _write_csv(path: Path) -> Path:
    path.write_text(
    "age,fare,sex,alive\n25,7.25,male,0\n38,71.28,female,1\n26,7.92,female,1\n35,53.10,female,1\n",
    encoding="utf-8",)
    return path

def test_schema_classifies_columns(tmp_path: Path) -> None:
    """Test: Dataset schema classifies columns."""
    csv = _write_csv(tmp_path / "t.csv")
    df, schema = load_csv_with_schema(csv)
    assert len(df) == 4

    roles = {c.name: c.role for c in schema.columns}
    assert roles["age"] == "numeric"
    assert roles["fare"] == "numeric"
    assert roles["sex"] == "categorical"
    assert roles["alive"] in {"numeric", "categorical"}

def test_dataset_hash_is_stable_across_runs(tmp_path: Path) -> None:
    """Test: Dataset hash is stable across runs."""
    csv = _write_csv(tmp_path / "t.csv")
    a = dataset_hash(csv)
    b = dataset_hash(csv)
    assert a == b
    assert len(a) == 64

def test_dataset_hash_changes_on_content_change(tmp_path: Path) -> None:
    """Test: Dataset hash must change on content change."""
    csv = _write_csv(tmp_path / "t.csv")
    a = dataset_hash(csv)

    csv.write_text(csv.read_text(encoding="utf-8") + "40,10.0,male,0\n", encoding="utf-8")
    b = dataset_hash(csv)
    assert a != b

def test_inspect_dataframe_populates_counts(tmp_path: Path) -> None:
    """Test: Inspect dataframe populates counts."""
    csv = _write_csv(tmp_path / "t.csv")
    df, _ = load_csv_with_schema(csv)

    info = inspect_dataframe(df)
    assert info.n_rows == 4
    assert info.n_cols == 4

    names = {c.name for c in info.columns}
    assert names == {"age", "fare", "sex", "alive"}

    sex = next(c for c in info.columns if c.name == "sex")
    assert sex.non_null == 4
    assert sex.null == 0
    assert sex.unique == 2
