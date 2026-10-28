"""Local subprocess runner that isolates training from the server process."""

from __future__ import annotations

import time
from typing import Any
from queue import Empty
import multiprocessing as mp
from multiprocessing.process import BaseProcess

from skudio.components import Task
from skudio.core.ir.canonical import canonical_json
from skudio.core.ir.graph import Graph
from skudio.execution.result import TrainingResult

_POLL_SECONDS = 0.1
_JOIN_GRACE_SECONDS = 5


def run_training(
    graph: Graph,
    *,
    dataset_path: str,
    target: str,
    task: Task = "classification",
    cv: int | None = None,
    random_state: int = 0,
    timeout_seconds: float | None = None,
    save_to_project_dir: str | None = None,
) -> TrainingResult:
    """Run training in a spawned child process; return its TrainingResult."""
    ctx = mp.get_context("spawn")

    # mp.Queue is generic at type-check time
    # pylint: disable=unsubscriptable-object
    queue: mp.Queue[Any] = ctx.Queue()

    payload = canonical_json(graph)
    child = ctx.Process(
        target=_child_main,
        args=(queue, payload, dataset_path, target, task, cv, random_state, save_to_project_dir),
        daemon=False,
    )

    child.start()
    try:
        payload = _drain_queue(queue, child, timeout_seconds)
        if payload is None:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
                return TrainingResult(status="failed", error="training exceeded timeout")

            child.join(timeout=_JOIN_GRACE_SECONDS)
            return TrainingResult(
                status="failed",
                error=f"child exited with code {child.exitcode} and no result",
            )

        child.join(timeout=_JOIN_GRACE_SECONDS)
        return TrainingResult.model_validate_json(payload)

    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=_JOIN_GRACE_SECONDS)


def _drain_queue(
    queue: mp.Queue[Any],  # pylint: disable=unsubscriptable-object
    child: BaseProcess,
    timeout_seconds: float | None
) -> Any:
    """Reading before/while the child is alive avoids mp.Queue deadlock."""
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while True:
        try:
            return queue.get(timeout=_POLL_SECONDS)
        except Empty:
            pass

        if not child.is_alive():
            return None

        if deadline is not None and time.monotonic() >= deadline:
            return None


def _child_main(
    queue: mp.Queue[Any],  # pylint: disable=unsubscriptable-object
    ir_json: str,
    dataset_path: str,
    target: str,
    task: Task,
    cv: int | None,
    random_state: int,
    save_to_project_dir: str | None,
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
            cv=cv,
            random_state=random_state,
            save_to_project_dir=save_to_project_dir,
        )

    except Exception as exc:  # pylint: disable=broad-except
        result = TrainingResult(status="failed", error=f"{type(exc).__name__}: {exc}")
    queue.put(result.model_dump_json())
