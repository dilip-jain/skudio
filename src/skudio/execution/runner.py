"""Local subprocess runner that isolates training from the server process."""

from __future__ import annotations

from typing import Any
import multiprocessing as mp

from skudio.components import Task
from skudio.core.ir.canonical import canonical_json
from skudio.core.ir.graph import Graph
from skudio.execution.result import TrainingResult


def run_training(
    graph: Graph,
    *,
    dataset_path: str,
    target: str,
    task: Task = "classification",
    random_state: int = 0,
    timeout_seconds: float | None = None
) -> TrainingResult:
    """Run training in a spawned child process; return its TrainingResult."""
    ctx = mp.get_context("spawn")

    # mp.Queue is generic at type-check time
    # pylint: disable=unsubscriptable-object
    queue: mp.Queue[Any] = ctx.Queue()

    payload = canonical_json(graph)
    child = ctx.Process(
        target=_child_main,
        args=(queue, payload, dataset_path, target, task, random_state),
        daemon=False,
    )

    child.start()
    try:
        child.join(timeout=timeout_seconds)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
            return TrainingResult(status="failed", error="training exceeded timeout")

        if queue.empty():
            return TrainingResult(
                status="failed",
                error=f"child exited with code {child.exitcode} and no result",
            )

        payload = queue.get_nowait()
        return TrainingResult.model_validate_json(payload)

    finally:
        if child.is_alive():
            child.terminate()


def _child_main(
    queue: mp.Queue[Any],  # pylint: disable=unsubscriptable-object  # mp.Queue is generic at type-check time
    ir_json: str,
    dataset_path: str,
    target: str,
    task: Task,
    random_state: int
) -> None:
    # Deferred: train pulls in sklearn; keep it out of the parent process import path.
    # pylint: disable=import-outside-toplevel
    from skudio.execution.train import train_supervised

    try:
        graph = Graph.model_validate_json(ir_json)
        result = train_supervised(
            graph,
            dataset_path=dataset_path,
            target=target,
            task=task,
            random_state=random_state
        )

    except Exception as exc:  # pylint: disable=broad-except
        result = TrainingResult(status="failed", error=f"{type(exc).__name__}: {exc}")
    queue.put(result.model_dump_json())
