"""Test: Predict."""

from __future__ import annotations

from pathlib import Path

import pytest

from skudio.core.ir import (
    ComponentRef,
    Edge,
    Node,
    Graph,
    LiteralValue,
    NodeRef,
    PortSpec
)
from skudio.execution.train import train_supervised
from skudio.execution.predict import PredictionError, batch_predict
from skudio.experiments.project import Project


def _iris_frame():
    pytest.importorskip("pandas")
    sk_datasets = pytest.importorskip("sklearn.datasets")
    frame = sk_datasets.load_iris(as_frame=True).frame
    frame.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in frame.columns]
    return frame

def _iris_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "iris.csv"
    _iris_frame().to_csv(csv, index=False)
    return csv

def _rf_graph() -> Graph:
    return Graph(
        id="g",
        name="iris",
        nodes=[
            Node(id="data", kind="dataset", outputs=[PortSpec(name="out", type="dataframe")]),
            Node(
                id="model",
                kind="estimator",
                component=ComponentRef(
                    provider="sklearn",
                    qualname="sklearn.ensemble.RandomForestClassifier",
                ),
                params={
                    "n_estimators": LiteralValue(value=50),
                    "random_state": LiteralValue(value=0),
                },
                inputs=[PortSpec(name="in", type="dataframe")],
            ),
        ],
        edges=[
            Edge(src=NodeRef(node_id="data", port="out"), dst=NodeRef(node_id="model", port="in"))
        ],
    )

def _train_and_project(tmp_path: Path) -> tuple[Project, str]:
    pytest.importorskip("sklearn")
    csv = _iris_csv(tmp_path)
    project_dir = tmp_path / "proj"
    result = train_supervised(
        _rf_graph(),
        dataset_path=str(csv),
        target="target",
        save_to_project_dir=str(project_dir),
    )

    assert result.status == "success"
    assert result.run_id
    return Project.open(project_dir), result.run_id

def test_batch_predict_matches_direct_sklearn_predict(tmp_path: Path) -> None:
    """Test: batch predict matches direct sklearn predict."""
    import joblib # pylint: disable=import-outside-toplevel

    project, run_id = _train_and_project(tmp_path)
    x = _iris_frame().drop(columns=["target"]).head(20)

    predictions, warnings = batch_predict(project, run_id, x)
    assert not warnings
    assert len(predictions) == 20
    assert "prediction" in predictions.columns

    # proba_0 / proba_1 / proba_2 should exist for the 3-class iris classifier.
    assert {"proba_0", "proba_1", "proba_2"}.issubset(predictions.columns)

    # Sanity: our predictions match sklearn's directly.
    model = joblib.load(project.run_dir(run_id) / "pipeline.joblib")
    assert (predictions["prediction"].to_numpy() == model.predict(x)).all()

def test_batch_predict_drops_extra_columns_with_warning(tmp_path: Path) -> None:
    """Test: batch predict drops extra columns with warning."""
    project, run_id = _train_and_project(tmp_path)
    x = _iris_frame().drop(columns=["target"]).head(5).copy()
    x["ignored_column"] = "junk"

    predictions, warnings = batch_predict(project, run_id, x)
    assert len(predictions) == 5
    assert len(warnings) == 1 and "ignored_column" in warnings[0]

    # The extra column is preserved in the output frame; only the aligned frame drops it.
    assert "ignored_column" in predictions.columns
    assert "prediction" in predictions.columns

def test_batch_predict_missing_columns_raises(tmp_path: Path) -> None:
    """Test: batch predict raise on missing columns."""
    project, run_id = _train_and_project(tmp_path)
    x = _iris_frame().drop(columns=["target", "sepal_length"]).head(5)

    with pytest.raises(PredictionError) as exc:
        batch_predict(project, run_id, x)
    assert "sepal_length" in str(exc.value.context.get("missing", ""))

def test_batch_predict_unknown_run_raises(tmp_path: Path) -> None:
    """Test: batch predict raise unknown run."""
    project, _ = _train_and_project(tmp_path)
    x = _iris_frame().drop(columns=["target"]).head(2)

    with pytest.raises(PredictionError):
        batch_predict(project, "does-not-exist", x)
