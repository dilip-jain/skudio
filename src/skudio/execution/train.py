"""Train model: Executed inside the runner subprocess."""

from __future__ import annotations

import time
from typing import Any
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from skudio.components import Task
from skudio.core.compiler import compile_graph
from skudio.core.hashing import ir_hash
from skudio.core.ir.graph import Graph
from skudio.data.reader import dataset_hash, load_csv
from skudio.execution.result import Metrics, TrainingResult
from skudio.experiments.persist import new_run_id, save_run
from skudio.experiments.project import Project


def train_supervised(
    graph: Graph,
    dataset_path: str,
    target: str,
    task: Task = "classification",
    cv: int | None = None,
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

        cv_scores: list[float] | None = None
        if cv and cv > 1:
            scoring = None if task == "classification" else "r2"
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            cv_scores = [float(s) for s in scores]

        stratify = y if task == "classification" and _is_stratifiable(y) else None
        X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806  # pylint: disable=invalid-name
            X, y, random_state=effective_rs, stratify=stratify,)

        model.fit(X_train, y_train)
        fitted_model = model
        y_pred = model.predict(X_test)

        if task == "classification":
            metrics, cm = _classification_metrics(y_test, y_pred)
        else:
            metrics, cm = _regression_metrics(y_test, y_pred), None

        extras = _build_extras(task, model, X_test, y_test, y_pred)
        result = TrainingResult(
            status="success",
            metrics=metrics,
            confusion_matrix=cm,
            feature_importance=_maybe_feature_importance(model),
            cv_scores=cv_scores,
            ir_hash=ir_hash(graph),
            dataset_hash=dataset_hash(dataset_path),
            duration_seconds=time.perf_counter() - started,
            extras=extras,
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


def _classification_metrics(y_true: Any, y_pred: Any) -> tuple[Metrics, list[list[int]]]:
    # `zero_division=0` is accepted at runtime; the stubs are stricter than sklearn.
    metrics = Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    )

    return metrics, confusion_matrix(y_true, y_pred).tolist()


def _regression_metrics(y_true: Any, y_pred: Any) -> Metrics:
    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def _is_stratifiable(y: Any) -> bool:
    try:
        counts = y.value_counts()
        return bool((counts >= 2).all() and len(counts) > 1)
    except AttributeError:
        return False


def _build_extras(task: str, model: Any, x_test: Any, y_test: Any, y_pred: Any) -> dict[str, Any]:
    # Persist test-set predictions for UI
    extras: dict[str, Any] = {
        "task": task,
        "y_test": [_to_native(v) for v in y_test.tolist()],
        "y_pred": [_to_native(v) for v in list(y_pred)],

        # `x_test` as a records dict lets permutation importance and other
        # ad-hoc inspections rehydrate without re-splitting the dataset.
        "x_test_records": x_test.to_dict(orient="records"),
        "x_test_columns": [str(c) for c in x_test.columns],
    }

    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_test)
            extras["y_proba"] = [[float(v) for v in row] for row in proba]

            classes = getattr(model, "classes_", None)
            if classes is not None:
                extras["classes"] = [_to_native(c) for c in list(classes)]
        finally:
            proba = None
    return extras


def _to_native(v: Any) -> Any:
    # Convert numpy scalars to plain Python so the extras dict is JSON-friendly.
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


def _maybe_feature_importance(model: Any) -> dict[str, float] | None:
    est = _final_estimator(model)
    if est is None:
        return None

    importances = getattr(est, "feature_importances_", None)
    if importances is None:
        return None

    names = _feature_names(model)
    if names is None or len(names) != len(importances):
        names = [f"f{i}" for i in range(len(importances))]

    return {str(n): float(v) for n, v in zip(names, importances, strict=False)}


def _final_estimator(model: Any) -> Any:
    inner = getattr(model, "best_estimator_", None) or model
    steps = getattr(inner, "steps", None)
    if steps:
        return steps[-1][1]
    return inner


def _feature_names(model: Any) -> list[str] | None:
    for candidate in (model, getattr(model, "best_estimator_", None)):
        if candidate is None:
            continue
        names_fn = getattr(candidate, "get_feature_names_out", None)
        if names_fn is None:
            continue
        try:
            return [str(n) for n in names_fn()]
        except Exception:  # pylint: disable=broad-except
            return None
    return None
