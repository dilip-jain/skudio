"""Typed ports on IR nodes."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, model_validator

PortType = Literal[
    "dataframe",
    "ndarray",
    "series",
    "fitted_transformer",
    "fitter_estimator",
    "predictions",
    "metrics",
]

PORT_TYPES: tuple[PortType, ...] = get_args(PortType)


class PortSpec(BaseModel):
    """A named typed port on a node."""
    model_config = ConfigDict(extra="forbid")

    name: str
    type: PortType
    columns: list[str] | None = None

    @model_validator(mode="after")
    def _columns_only_on_dataframe(self) -> PortSpec:
        if self.columns is not None and self.type != "dataframe":
            raise ValueError("`columns` is only valid when type == 'dataframe'")
        return self
