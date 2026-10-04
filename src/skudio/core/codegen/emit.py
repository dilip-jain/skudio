"""Emit runnable Python for an IR Graph.

Two entry points:
- `emit_module` - returns a `.py` with a `build_pipeline()` fn that returns a sklearn estimator.
- `emit_script` - returns a standalone script that load, fit, and print task-appropriate metrics.

Uses `ast.unparse` throughout; no string templating of code fragments.
"""

from __future__ import annotations

import ast
from typing import Any

from skudio.components import Task
from skudio.components.registry import Registry, default_registry
from skudio.core.errors import CodegenError, RegistryError
from skudio.core.ir.graph import Graph
from skudio.core.ir.nodes import Node
from skudio.core.ir.params import LiteralValue, ParamValue


def emit_module(graph: Graph, *, registry: Registry | None = None) -> str:
    """Emit a Python module that builds a sklearn pipeline from an IR Graph."""
    reg = registry or default_registry()
    imports = _collect_imports(graph)
    body = _emit_graph(graph, reg=reg, root=graph)
    module = _build_module(
        imports=imports,
        factory_return=body,
        docstring=_header_docstring(graph),
    )

    return ast.unparse(module) + "\n"

def emit_script(
    graph: Graph,
    dataset_path: str,
    target: str,
    *,
    task: Task = "classification",
    registry: Registry | None = None,
) -> str:
    """Emit a standalone Python script that loads a CSV, fits the pipeline, and prints metrics."""
    reg = registry or default_registry()
    module = _build_script(graph, dataset_path=dataset_path, target=target, task=task, reg=reg)
    return ast.unparse(module) + "\n"

def _collect_imports(graph: Graph) -> dict[str, set[str]]:
    """ Collect all imports needed for the graph."""
    modules: dict[str, set[str]] = {}
    for node in _iter_nodes(graph):
        if node.component is not None:
            module_name, _, cls_name = node.component.qualname.rpartition(".")
            modules.setdefault(module_name, set()).add(cls_name)
        if node.kind == "column_transformer":
            modules.setdefault("sklearn.compose", set()).add("ColumnTransformer")
        if node.kind == "pipeline":
            modules.setdefault("sklearn.pipeline", set()).add("Pipeline")
        if node.kind == "feature_union":
            modules.setdefault("sklearn.pipeline", set()).add("FeatureUnion")

    # If the top graph itself becomes a Pipeline of steps, add it
    if _needs_pipeline_wrapper(graph):
        modules.setdefault("sklearn.pipeline", set()).add("Pipeline")
    return modules

def _needs_pipeline_wrapper(graph: Graph) -> bool:
    """Determine if the top-level graph needs to be wrapped in a Pipeline."""
    steps = [n for n in _iter_nodes(graph) if n.kind not in ("dataset", "search")]
    return len(steps) > 1

def _iter_nodes(graph: Graph) -> list[Node]:
    """Iterate over all nodes in the graph, including nested graphs."""
    nodes = []
    for node in graph.nodes:
        nodes.append(node)
        if node.children is not None:
            nodes.extend(_iter_nodes(node.children))
    return nodes

def _emit_graph(graph: Graph, *, reg: Registry, root: Graph) -> ast.expr:
    """Emit an AST for the given graph."""
    steps: list[tuple[str, ast.expr]] = []
    search_node: Node | None = None
    for node in _iter_nodes(graph):
        if node.kind == "search":
            if search_node is not None:
                raise CodegenError("Multiple search nodes in the same graph are not supported.")
            search_node = node
            continue

        if node.kind == "dataset":
            continue  # datasets are handled separately

        steps.append((node.id, _emit_node(node, reg=reg, root=root)))

    if search_node is not None:
        if steps:
            raise CodegenError("Search nodes cannot be combined with other in the same graph.")
        return _emit_search(search_node, reg=reg, root=root)

    if not steps:
        raise CodegenError("Empty Pipeline: no steps found in the graph.")
    if len(steps) == 1:
        return steps[0][1]  # single step, no need for a Pipeline wrapper
    return _emit_pipeline(steps)

def _emit_pipeline(steps: list[tuple[str, ast.expr]]) -> ast.expr:
    """Emit an AST for a Pipeline of steps."""
    step_tuples: list[ast.expr] = [
        ast.Tuple(elts=[ast.Constant(value=n), e], ctx=ast.Load()) for n, e in steps
    ]
    return ast.Call(
        func=ast.Name(id="Pipeline", ctx=ast.Load()),
        args=[],
        keywords=[ast.keyword(arg="steps", value=ast.List(elts=step_tuples, ctx=ast.Load()))],
    )

def _emit_node(node: Node, *, reg: Registry, root: Graph) -> ast.expr:
    """Emit an AST for a single node."""
    if node.kind == "column_transformer":
        return _emit_column_transformer(node, reg=reg, root=root)
    if node.kind == "feature_union":
        return _emit_feature_union(node, reg=reg, root=root)
    if node.kind == "pipeline":
        if node.children is None:
            raise CodegenError(f"Pipeline node {node.id} has no children.")
        return _emit_graph(node.children, reg=reg, root=root)
    if node.kind in ("estimator", "transformer"):
        return _emit_component(node, reg=reg, root=root)
    raise CodegenError(f"Unsupported node kind for codegen: {node.kind}")

def _emit_column_transformer(node: Node, *, reg: Registry, root: Graph) -> ast.expr:
    """Emit an AST for a ColumnTransformer node."""
    if node.children is None:
        raise CodegenError(f"ColumnTransformer node {node.id} has no children.")

    branches = []
    for child in node.children.nodes:
        cols = _literal_str_list(child.params.get("columns"))
        stripped = child.model_copy(
            update={"params": {k: v for k, v in child.params.items() if k != "columns"}}
        )

        est = _emit_node(stripped, reg=reg, root=root)
        branches.append(ast.Tuple(
            elts=[ast.Constant(value=child.id), est, _list_of_str(cols)], ctx=ast.Load()
        ))

    remainder = _literal_or(node.params.get("remainder"), default="drop")
    return ast.Call(
        func=ast.Name(id="ColumnTransformer", ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg="transformers", value=ast.List(elts=branches, ctx=ast.Load())),
            ast.keyword(arg="remainder", value=ast.Constant(value=remainder)),
        ],
    )

def _emit_feature_union(node: Node, *, reg: Registry, root: Graph) -> ast.expr:
    """Emit an AST for a FeatureUnion node."""
    if node.children is None:
        raise CodegenError(f"FeatureUnion node {node.id} has no children.")

    branches: list[ast.expr] = [ast.Tuple(
        elts=[ast.Constant(value=child.id), _emit_node(child, reg=reg, root=root)], ctx=ast.Load())
    for child in node.children.nodes]

    return ast.Call(
        func=ast.Name(id="FeatureUnion", ctx=ast.Load()),
        args=[],
        keywords=[ast.keyword(
            arg="transformer_list",
            value=ast.List(elts=branches, ctx=ast.Load()))
        ],
    )

def _emit_search(node: Node, *, reg: Registry, root: Graph) -> ast.expr:
    """Emit an AST for a Search node."""
    if node.component is None:
        raise CodegenError(f"Search node {node.id} has no component assigned.")
    if node.children is None:
        raise CodegenError(f"Search node {node.id} has no children.")

    inner = _emit_graph(node.children, reg=reg, root=root)
    cls_name = node.component.qualname.rpartition(".")[2]
    kws: list[ast.keyword] = [
        ast.keyword(arg=k, value=_to_ast(_literal_or(v)))
    for k, v in node.params.items()]

    _inject_random_state(kws, node, reg=reg, root=root)
    return ast.Call(
        func=ast.Name(id=cls_name, ctx=ast.Load()),
        args=[inner],
        keywords=kws,
    )

def _emit_component(node: Node, *, reg: Registry, root: Graph) -> ast.expr:
    """Emit an AST for a component node (estimator or transformer)."""
    if node.component is None:
        raise CodegenError(f"Node {node.id} has no component assigned.")

    cls_name = node.component.qualname.rpartition(".")[2]
    kws: list[ast.keyword] = [
        ast.keyword(arg=k, value=_to_ast(_literal_or(v)))
    for k, v in node.params.items() if k != "columns"]

    _inject_random_state(kws, node, reg=reg, root=root)
    return ast.Call(
        func=ast.Name(id=cls_name, ctx=ast.Load()),
        args=[],
        keywords=kws,
    )

def _inject_random_state(kws: list[ast.keyword], node: Node, *, reg: Registry, root: Graph) -> None:
    """Inject a random_state parameter if the component supports it and it's not already set."""
    if root.random_state is None or node.component is None:
        return
    if any(k.arg == "random_state" for k in kws):
        return  # already set

    try:
        meta = reg.get(node.component.qualname)
    except RegistryError:
        return  # component not found in registry, skip

    if any(p.name == "random_state" for p in meta.params):
        kws.append(ast.keyword(arg="random_state", value=ast.Constant(value=root.random_state)))

def _literal_or(value: ParamValue | None, *, default: Any = None) -> Any:
    """Return the literal value of a ParamValue, or a default if None."""
    if value is None:
        return default
    if isinstance(value, LiteralValue):
        return value.value
    raise CodegenError(f"Expected a LiteralValue, got {type(value)} for value: {value}")

def _literal_str_list(value: ParamValue | None) -> list[str]:
    """Return a list of strings from a ParamValue, or an empty list if None."""
    value = _literal_or(value, default=[])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CodegenError(f"Expected a list of strings, got {value}")
    return list(value)

def _list_of_str(values: list[str]) -> ast.List:
    """Return an AST List of strings."""
    return ast.List(elts=[ast.Constant(value=v) for v in values], ctx=ast.Load())

def _to_ast(value: Any) -> ast.expr:
    """Convert a Python value to an AST expression."""
    if value is None or isinstance(value, (int, float, str, bool)):
        return ast.Constant(value=value)
    elif isinstance(value, list):
        return ast.List(elts=[_to_ast(v) for v in value], ctx=ast.Load())
    elif isinstance(value, tuple):
        return ast.Tuple(elts=[_to_ast(v) for v in value], ctx=ast.Load())
    elif isinstance(value, dict):
        return ast.Dict(
            keys=[_to_ast(k) for k in value.keys()],
            values=[_to_ast(v) for v in value.values()]
        )
    raise CodegenError(f"Unsupported value type for AST conversion: {type(value)}")

def _build_module(
    *,
    imports: dict[str, set[str]],
    factory_return: ast.expr,
    docstring: ast.Expr | None,
) -> ast.Module:
    """Build an AST for a Python module with a build_pipeline() factory."""
    body: list[ast.stmt] = []
    if docstring is not None:
        body.append(docstring)

    for module in sorted(imports):
        names = sorted(imports[module])
        body.append(ast.ImportFrom(
            module=module,
            names=[ast.alias(name=n) for n in names],
            level=0
        ))

    body.append(_factory_function(factory_return))
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    return mod

def _factory_function(return_expr: ast.expr) -> ast.FunctionDef:
    """Build an AST for the build_pipeline() factory function."""
    return ast.FunctionDef(
        name="build_pipeline",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=[ast.Return(value=return_expr)],
        decorator_list=[],
        returns=None,
    )

def _header_docstring(graph: Graph) -> ast.Expr:
    """Build a docstring for the generated module."""
    # pylint: disable=import-outside-toplevel
    from skudio import NAME, __version__
    from skudio.core.hashing import ir_hash

    text = f"""Generated by {NAME} {__version__} (run: {ir_hash(graph)}) for graph: {graph.id}
This module contains a factory function `build_pipeline()` that returns a sklearn estimator."""
    return ast.Expr(value=ast.Constant(value=text))

def _build_script(
    graph: Graph,
    *,
    dataset_path: str,
    target: str,
    task: Task,
    reg: Registry
) -> ast.Module:
    """Build an AST for a standalone script that load, fit, and print metrics."""
    imports: dict[str, set[str]] = _collect_imports(graph)
    imports.setdefault("sklearn.model_selection", set()).add("train_test_split")
    if task == "classification":
        imports.setdefault("sklearn.metrics", set()).add("accuracy_score")
    elif task == "regression":
        imports.setdefault("sklearn.metrics", set()).add("r2_score")
    else:
        raise CodegenError(f"Unsupported task for script generation: {task}")

    module = _build_module(
        imports=imports,
        factory_return=_emit_graph(graph, reg=reg, root=graph),
        docstring=_header_docstring(graph)
    )

    module.body.insert(
        1 if module.body and isinstance(module.body[0], ast.Expr) else 0,
        ast.Import(names=[ast.alias(name="pandas", asname="pd")]),
    )

    module.body.extend(
        _script_main(
            dataset_path=dataset_path,
            target=target,
            task=task,
            random_state=graph.random_state if graph.random_state else 0,
        )
    )

    ast.fix_missing_locations(module)
    return module

def _script_main(
    *,
    dataset_path: str,
    target: str,
    task: Task,
    random_state: int
) -> list[ast.stmt]:
    if task == "classification":
        metric_line = "print('accuracy:', accuracy_score(y_test, model.predict(X_test)))"
    else:
        metric_line = "print('r2:', r2_score(y_test, model.predict(X_test)))"

    src = f"""
if __name__ == '__main__':
    df = pd.read_csv({dataset_path!r})
    y = df[{target!r}]
    X = df.drop(columns=[{target!r}])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state={random_state})
    model = build_pipeline()
    model.fit(X_train, y_train)
    {metric_line}
"""
    return list(ast.parse(src).body)
