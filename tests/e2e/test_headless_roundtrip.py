"""Test: End-to-end."""

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
from skudio.execution.runner import run_training
from skudio.execution.train import train_supervised


def _iris_csv(tmp_path: Path) -> Path:
    pytest.importorskip("pandas")
    sk_datasets = pytest.importorskip("sklearn.datasets")
    iris = sk_datasets.load_iris(as_frame=True).frame
    iris.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in iris.columns]
    csv = tmp_path / "iris.csv"
    iris.to_csv(csv, index=False)
    return csv


def _diabetes_csv(tmp_path: Path) -> Path:
    pytest.importorskip("pandas")
    sk_datasets = pytest.importorskip("sklearn.datasets")
    frame = sk_datasets.load_diabetes(as_frame=True).frame
    csv = tmp_path / "diabetes.csv"
    frame.to_csv(csv, index=False)
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


def _rf_regressor_graph() -> Graph:
    return Graph(
        id="g",
        name="diabetes",
        nodes=[
            Node(id="data", kind="dataset", outputs=[PortSpec(name="out", type="dataframe")]),
            Node(
                id="model",
                kind="estimator",
                component=ComponentRef(
                    provider="sklearn",
                    qualname="sklearn.ensemble.RandomForestRegressor",
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


def test_train_classifier_in_process(tmp_path: Path) -> None:
    """Test: Train classifier in process."""
    pytest.importorskip("sklearn")
    csv = _iris_csv(tmp_path)
    result = train_supervised(_rf_graph(), dataset_path=str(csv), target="target")
    assert result.status == "success", result.error
    assert result.metrics.accuracy is not None
    assert result.metrics.accuracy > 0.9


def test_train_regressor_in_process(tmp_path: Path) -> None:
    """Test: Train regressor in process."""
    pytest.importorskip("sklearn")
    csv = _diabetes_csv(tmp_path)
    result = train_supervised(
        _rf_regressor_graph(),
        dataset_path=str(csv),
        target="target",
        task="regression"
    )

    assert result.status == "success", result.error
    assert result.metrics.r2 is not None
    assert result.metrics.mae is not None
    assert result.metrics.rmse is not None
    assert result.confusion_matrix is None  # regression path skips CM



def test_train_via_subprocess_runner(tmp_path: Path) -> None:
    """Test: Train via subprocess."""
    pytest.importorskip("sklearn")
    csv = _iris_csv(tmp_path)
    result = run_training(_rf_graph(), dataset_path=str(csv), target="target")
    assert result.status == "success", result.error
    assert result.metrics.accuracy is not None
    assert result.metrics.accuracy > 0.9


def test_run_persistence_writes_all_four_artifacts(tmp_path: Path) -> None:
    """Test: Run persistence - write all artifacts."""
    pytest.importorskip("sklearn")
    csv = _iris_csv(tmp_path)
    project = tmp_path / "proj"

    r1 = train_supervised(
        _rf_graph(),
        dataset_path=str(csv),
        target="target",
        save_to_project_dir=str(project),
    )
    r2 = train_supervised(
        _rf_graph(),
        dataset_path=str(csv),
        target="target",
        save_to_project_dir=str(project),
    )

    assert r1.status == "success"
    assert r2.status == "success"
    assert r1.run_id
    assert r2.run_id
    assert r1.run_id != r2.run_id

    runs = sorted((project / "experiments").iterdir())
    assert len(runs) == 2
    for d in runs:
        assert (d / "ir.json").exists()
        assert (d / "result.json").exists()
        assert (d / "generated.py").exists()
        assert (d / "pipeline.joblib").exists()


def test_run_persistence_through_subprocess_runner(tmp_path: Path) -> None:
    """Test: Run persistence through subprocess runner."""
    pytest.importorskip("sklearn")
    csv = _iris_csv(tmp_path)
    project = tmp_path / "proj_runner"

    result = run_training(
        _rf_graph(),
        dataset_path=str(csv),
        target="target",
        save_to_project_dir=str(project),
    )

    assert result.status == "success"
    assert result.run_id
    d = project / "experiments" / result.run_id
    assert (d / "pipeline.joblib").exists()
    assert (d / "ir.json").exists()
