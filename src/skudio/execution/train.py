"""Train model: Executed inside the runner subprocess."""

from __future__ import annotations

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from skudio.components import Task
from skudio.core.compiler import compile_graph
from skudio.core.hashing import ir_hash
from skudio.core.ir.graph import Graph
from skudio.data.reader import load_csv
from skudio.execution.result import TrainingResult


def train_supervised(
    graph: Graph,
    dataset_path: str,
    target: str,
    task: Task = "classification",
    random_state: int = 0
) -> TrainingResult:
    """Load a CSV, fit a compiled pipeline, return metrics for the given task.

    If ``save_to_project_dir`` is given, persist the run (IR, result,
    generated code, fitted pipeline) into that project directory and
    stamp the run id onto the returned result.
    """
    # graph.random_state (IR-level) > caller's default.
    effective_rs = graph.random_state if graph.random_state is not None else random_state

    started = time.perf_counter()
    try:
        df = load_csv(dataset_path)
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} not in dataset")

        y = df[target]
        X = df.drop(columns=[target])  # noqa: N806  # pylint: disable=invalid-name

        model = compile_graph(graph)

        X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806 # pylint: disable=invalid-name
            X, y, random_state=effective_rs)

        model.fit(X_train, y_train)
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
    return result
