"""Test: Persistence."""

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
from skudio.execution.result import Metrics, TrainingResult
from skudio.experiments.persist import list_runs, load_run, save_run
from skudio.experiments.project import Project


def _tiny_graph() -> Graph:
    return Graph(
        id="g",
        name="lr",
        nodes=[
            Node(id="data", kind="dataset", outputs=[PortSpec(name="out", type="dataframe")]),
            Node(
                id="lr",
                kind="estimator",
                component=ComponentRef(
                    provider="sklearn", qualname="sklearn.linear_model.LogisticRegression"
                ),
                params={"C": LiteralValue(value=1.0)},
            ),
        ],
        edges=[Edge(src=NodeRef(node_id="data", port="out"), dst=NodeRef(node_id="lr", port="in"))],
    )


def test_project_create_open_roundtrip(tmp_path: Path) -> None:
    """Test: project creation."""
    p = Project.create(tmp_path / "proj", name="demo")
    assert p.meta().name == "demo"
    reopened = Project.open(tmp_path / "proj")
    assert reopened.meta().name == "demo"


def test_project_rejects_paths_outside_root(tmp_path: Path) -> None:
    """Test: project rejects path outside root."""
    p = Project.create(tmp_path / "proj", name="demo")
    # '..' twice: from proj/experiments/, that lands outside proj/.
    with pytest.raises(ValueError):
        p.run_dir("../../escape")


def test_save_run_writes_expected_files(tmp_path: Path) -> None:
    """Test: Save run."""
    p = Project.create(tmp_path / "proj", name="demo")
    graph = _tiny_graph()
    result = TrainingResult(status="success", metrics=Metrics(accuracy=0.99))
    rid = save_run(p, graph=graph, result=result)
    d = p.run_dir(rid)
    assert (d / "ir.json").exists()
    assert (d / "result.json").exists()
    # Codegen may fail without sklearn; either way, one of these paths exists.
    assert (d / "generated.py").exists() or (d / "generated.py.error").exists()


def test_load_run_returns_saved_pair(tmp_path: Path) -> None:
    """Test: load run."""
    p = Project.create(tmp_path / "proj", name="demo")
    graph = _tiny_graph()
    result = TrainingResult(status="success", metrics=Metrics(accuracy=0.42))
    rid = save_run(p, graph=graph, result=result)
    got_graph, got_result = load_run(p, rid)
    assert got_result.metrics.accuracy == 0.42
    assert got_graph.id == graph.id


def test_list_runs_reflects_directory(tmp_path: Path) -> None:
    """Test: list runs."""
    p = Project.create(tmp_path / "proj", name="demo")
    assert list_runs(p) == []
    save_run(
        p,
        graph=_tiny_graph(),
        result=TrainingResult(status="success", metrics=Metrics(accuracy=0.5)),
    )
    save_run(
        p,
        graph=_tiny_graph(),
        result=TrainingResult(status="success", metrics=Metrics(accuracy=0.7)),
    )
    assert len(list_runs(p)) == 2
