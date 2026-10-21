"""Training Result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Status = Literal["success", "failed"]


class Metrics(BaseModel):
    """Task-appropriate scalar metrics from a training run."""
    model_config = ConfigDict(extra="allow")

    # classification
    accuracy: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    roc_auc: float | None = None

    # regression
    r2: float | None = None
    mae: float | None = None
    rmse: float | None = None


class TrainingResult(BaseModel):
    """Everything a run produces, minus large arrays (which live on disk)."""
    model_config = ConfigDict(extra="forbid")

    status: Status
    metrics: Metrics = Metrics()
    confusion_matrix: list[list[int]] | None = None
    error: str | None = None
    ir_hash: str = ""
    dataset_hash: str = ""
    duration_seconds: float = 0.0
    logs: list[str] = []
    run_id: str | None = None
