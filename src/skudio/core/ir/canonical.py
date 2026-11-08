"""Canonical JSON serialization for IR graphs."""

from __future__ import annotations

import json
from typing import Any

from skudio.core.ir.graph import Graph


def canonical_dict(graph: Graph, *, strip_ui: bool=False) -> dict[str, Any]:
    """Return the graph as a canonical dict (sorted keys, defaults included)"""
    data = graph.model_dump(mode="json")
    if strip_ui:
        _strip_ui(data)
    return data

def canonical_json(graph: Graph, *, strip_ui: bool=False) -> str:
    """Return canonical, byte-stable JSON for a graph."""
    return json.dumps(
        canonical_dict(graph, strip_ui=strip_ui),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )

def _strip_ui(value: Any) -> None:
    """Strip UI metadata"""
    if isinstance(value, dict):
        value.pop("ui", None)
        for v in value.values():
            _strip_ui(v)

    elif isinstance(value, list):
        for item in value:
            _strip_ui(item)
