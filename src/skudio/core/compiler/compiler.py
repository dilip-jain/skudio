""" Compile an IR graph into sklearn estimator objects.

Supported graph shape (v0.1):
    dataset -> [column_transformer] -> estimator
optionally wrapped by a single top-level `search` composite whose children carries the pipeline.

Composite children semantics:
- pipeline: children.nodes are steps in order; the last is the terminal step.
- column_transformer: each child is a branch whose `params["columns"]` is the column selector.
- search: single child is the graph to wrap; 
node.params carry cv/scoring/n_iter/refit + a `param_grid`/`param_distributions` as a literal dict.
"""

from __future__ import annotations

from typing import Any

from skudio.components.registry import Registry, default_registry
from skudio.core.errors import CompilerError, RegistryError
from skudio.core.ir.graph import Graph
from skudio.core.ir.nodes import Node
from skudio.core.ir.params import LiteralValue, ParamValue, Reference, SearchSpace


def compile_graph(graph: Graph, *, registry: Registry | None = None) -> Any:
    """Compile the graph and return the terminal sklearn estimator"""
    reg = registry or default_registry()
    return _compile_pipeline(graph, reg, graph)

def _compile_pipeline(graph: Graph, reg: Registry, root: Graph) -> Any:
    """Compile the graph pipeline."""
    # pylint: disable=import-outside-toplevel
    from sklearn.pipeline import Pipeline

    steps: list[tuple[str, Any]] = []
    search_node: Node | None = None
    for node in graph.nodes:
        if node.kind == "dataset":
            continue
        if node.kind == "search":
            if search_node is not None:
                raise CompilerError("Multiple search nodes in the same graph are not supported.")
            search_node = node
            continue
        steps.append((node.id, _compile_node(node, reg, root)))

    if search_node is not None:
        if steps:
            raise CompilerError("Search nodes cannot be combined with other in the same graph.")

        inner = _compile_search_child(search_node, reg, root)
        return _wrap_search(search_node, inner, reg, root)

    if not steps:
        raise CompilerError("Empty Pipeline: no steps found in the graph.")
    if len(steps) == 1:
        return steps[0][1]
    return Pipeline(steps=steps)


def _compile_node(node: Node, reg: Registry, root: Graph) -> Any:
    """Compile Node."""
    if node.kind == "column_transformer":
        return _compile_column_transformer(node, reg, root)
    if node.kind == "feature_union":
        return _compile_feature_union(node, reg, root)
    if node.kind == "pipeline":
        if node.children is None:
            raise CompilerError(f"Pipeline node {node.id} has no children.")
        return _compile_pipeline(node.children, reg, root)
    if node.kind in ("transformer", "estimator"):
        return _instantiate(node, reg, root)
    if node.kind == "search":
        raise CompilerError(f"Unexpected nested search node {node.id}")
    raise CompilerError(f"Unsupported node kind for codegen: {node.kind}")


def _compile_column_transformer(node: Node, reg: Registry, root: Graph) -> Any:
    """Compile ColumnTransformer."""
    # pylint: disable=import-outside-toplevel
    from sklearn.compose import ColumnTransformer

    if node.children is None:
        raise CompilerError(f"ColumnTransformer node {node.id} has no children.")

    transformers: list[tuple[str, Any, list[str]]] = []
    for child in node.children.nodes:
        columns = _literal_str_list(
            child.params.get("columns"), path=f"{node.id}.{child.id}.params.columns"
        )

        est = _compile_node(_strip_columns(child), reg, root)
        transformers.append((child.id, est, columns))

    remainder = _literal_scalar(node.params.get("remainder")) or "drop"
    return ColumnTransformer(transformers=transformers, remainder=remainder)


def _compile_feature_union(node: Node, reg: Registry, root: Graph) -> Any:
    """Compile FeatureUnion."""
    # pylint: disable=import-outside-toplevel
    from sklearn.pipeline import FeatureUnion

    if node.children is None:
        raise CompilerError(f"FeatureUnion node {node.id} has no children.")

    transformers: list[tuple[str, Any]] = [
        (child.id, _compile_node(child, reg, root))
    for child in node.children.nodes]
    return FeatureUnion(transformer_list=transformers)


def _wrap_search(node: Node, inner: Any, reg: Registry, root: Graph) -> Any:
    """Wrap Search Node."""
    if node.component is None:
        raise CompilerError(f"Search node {node.id} has no component assigned.")

    cls = reg.resolve_class(node.component.qualname)
    kwargs = _kwargs_from_params(node.params, exclude={"param_grid", "param_distributions"})
    _inject_random_state(kwargs, node, reg, root)

    grid = _literal_scalar(node.params.get("param_grid"))
    dists = _literal_scalar(node.params.get("param_distributions"))

    key = "param_grid" if grid is not None else "param_distributions"
    return cls(inner, **{key: grid if grid is not None else dists}, **kwargs)


def _compile_search_child(node: Node, reg: Registry, root: Graph) -> Any:
    """Compile Search Child."""
    if node.children is None:
        raise CompilerError(f"Search node {node.id} has no children.")
    return _compile_pipeline(node.children, reg, root)


def _inject_random_state(kwargs: dict[str, Any], node: Node, reg: Registry, root: Graph) -> None:
    """Inject Random State Keyword."""
    if root.random_state is None or "random_state" in kwargs or node.component is None:
        return

    try:
        meta = reg.get(node.component.qualname)
    except RegistryError:
        return

    if any(p.name == "random_state" for p in meta.params):
        kwargs["random_state"] = root.random_state


def _kwargs_from_params(
    params: dict[str, ParamValue], *, exclude: set[str] | None = None
) -> dict[str, Any]:
    """Collect kwargs from params """
    exclude = exclude or set()

    out: dict[str, Any] = {}
    for k, pv in params.items():
        if k in exclude or k == "columns":
            continue

        if isinstance(pv, LiteralValue):
            out[k] = pv.value

        elif isinstance(pv, Reference):
            raise CompilerError(
                "References in constructor kwargs are not supported yet",
                path=f"params.{k}",
                context={"ref": pv.ref},
            )

        elif isinstance(pv, SearchSpace):
            raise CompilerError(
                "SearchSpace param must be lifted into a search node's param_grid",
                path=f"params.{k}",
            )
    return out


def _literal_str_list(pv: ParamValue | None, *, path: str) -> list[str]:
    """Param to List[str]"""
    if not isinstance(pv, LiteralValue):
        raise CompilerError("Expected LiteralValue", path=path)

    value = pv.value
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise CompilerError("Expected list[str]", path=path)
    return list(value)


def _literal_scalar(pv: ParamValue | None) -> Any:
    """Param to Scalar"""
    if pv is None:
        return None
    if isinstance(pv, LiteralValue):
        return pv.value
    return None


def _strip_columns(node: Node) -> Node:
    """Return a copy of `node` with the `columns` selector removed from params."""
    if "columns" not in node.params:
        return node
    stripped = node.model_copy(
        update={"params": {k: v for k, v in node.params.items() if k != "columns"}}
    )
    return stripped


def _instantiate(node: Node, reg: Registry, root: Graph) -> Any:
    if node.component is None:
        raise CompilerError(f"{node.kind} node {node.id} has no component assigned.")

    cls = reg.resolve_class(node.component.qualname)
    kwargs = _kwargs_from_params(node.params)
    _inject_random_state(kwargs, node, reg, root)
    return cls(**kwargs)
