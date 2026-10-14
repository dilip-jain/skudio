"""Train model: Executed inside the runner subprocess."""

from __future__ import annotations

import time
from typing import Any
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from skudio.components import Task
from skudio.core.compiler import compile_graph
from skudio.core.hashing import ir_hash
from skudio.core.ir.graph import Graph
from skudio.data.reader import load_csv
from skudio.execution.result import TrainingResult
from skudio.experiments.persist import new_run_id, save_run
from skudio.experiments.project import Project


def train_supervised(
    graph: Graph,
    dataset_path: str,
    target: str,
    task: Task = "classification",
    random_state: int = 0,
    save_to_project_dir: str | None = None,
) -> TrainingResult:
    """Load a CSV, fit a compiled pipeline, return metrics for the given task.

    If ``save_to_project_dir`` is given, persist the run (IR, result,
    generated code, fitted pipeline) into that project directory and
    stamp the run id onto the returned result.
    """
    # graph.random_state (IR-level) > caller's default.
    effective_rs = graph.random_state if graph.random_state is not None else random_state

    started = time.perf_counter()
    fitted_model: Any | None = None
    try:
        df = load_csv(dataset_path)
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} not in dataset")

        y = df[target]
        X = df.drop(columns=[target])  # noqa: N806  # pylint: disable=invalid-name

        model = compile_graph(graph)

        X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806  # pylint: disable=invalid-name
            X, y, random_state=effective_rs)

        model.fit(X_train, y_train)
        fitted_model = model
        y_pred = model.predict(X_test)

        if task == "classification":
            accuracy = float(accuracy_score(y_test, y_pred))
        else:
            accuracy = float(r2_score(y_test, y_pred))

        result = TrainingResult(
            status="success",
            metrics=accuracy,
            ir_hash=ir_hash(graph),
            duration_seconds=time.perf_counter() - started,
        )

    except Exception as exc:  # pylint: disable=broad-except
        result = TrainingResult(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            ir_hash=ir_hash(graph),
            duration_seconds=time.perf_counter() - started,
        )

    if save_to_project_dir:
        result = _persist(save_to_project_dir, graph, result, fitted_model)
    return result


def _persist(
    project_dir: str, graph: Graph, result: TrainingResult, fitted_model: Any | None
) -> TrainingResult:
    root = Path(project_dir).expanduser()
    project = (
        Project.open(root)
        if (root / "project.json").exists()
        else Project.create(root, name=root.name or "default")
    )
    rid = new_run_id()
    stamped = result.model_copy(update={"run_id": rid})
    save_run(project, graph=graph, result=stamped, fitted_model=fitted_model, run_id=rid)
    return stamped
