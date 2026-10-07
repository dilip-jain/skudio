"""Training Result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Status = Literal["success", "failed"]


class TrainingResult(BaseModel):
    """Everything a run produces, minus large arrays (which live on disk)."""
    model_config = ConfigDict(extra="forbid")

    status: Status
    metrics: float | None = None
    error: str | None = None
    ir_hash: str = ""
    duration_seconds: float = 0.0
    logs: list[str] = []
