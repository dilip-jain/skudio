"""Persist a training run: IR, generated code, TrainingResult, fitted model."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

from skudio.core.codegen import emit_module
from skudio.core.ir.canonical import canonical_json
from skudio.core.ir.graph import Graph
from skudio.execution.result import TrainingResult
from skudio.experiments.project import Project


def new_run_id() -> str:
    """Return a fresh run id: 'YYYYMMDDThhmmssZ-<hex4>'."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(4)}"


def save_run(
    project: Project,
    *,
    graph: Graph,
    result: TrainingResult,
    fitted_model: Any | None = None,
    run_id: str | None = None,
) -> str:
    """Write ir.json, result.json, generated.py (and pipeline.joblib if given)."""
    rid = run_id or new_run_id()

    d = project.run_dir(rid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "ir.json").write_text(canonical_json(graph), encoding="utf-8")
    (d / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    try:
        (d / "generated.py").write_text(emit_module(graph), encoding="utf-8")
    except Exception as exc:  # pylint: disable=broad-except  # codegen error must never block persistence
        (d / "generated.py.error").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")

    if fitted_model is not None:
        import joblib  # pylint: disable=import-outside-toplevel
        joblib.dump(fitted_model, d / "pipeline.joblib")
    return rid


def load_run(project: Project, run_id: str) -> tuple[Graph, TrainingResult]:
    """Return the '(graph, result)' pair persisted under 'run_id'."""
    d = project.run_dir(run_id)
    graph = Graph.model_validate_json((d / "ir.json").read_text(encoding="utf-8"))
    result = TrainingResult.model_validate_json((d / "result.json").read_text(encoding="utf-8"))
    return graph, result


def list_runs(project: Project) -> list[str]:
    """Return sorted run ids for every run directory under the project."""
    if not project.experiments_dir.exists():
        return []
    return sorted(p.name for p in project.experiments_dir.iterdir() if p.is_dir())
