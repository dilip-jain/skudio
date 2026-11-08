"""Canonical, UI-independent hash of an IR graph."""

from __future__ import annotations

import hashlib

from skudio.core.ir.canonical import canonical_json
from skudio.core.ir.graph import Graph


def ir_hash(graph: Graph) -> str:
    """SHA-256 of the graph's canonical JSON with 'ui' fields stripped."""
    payload = canonical_json(graph, strip_ui=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
