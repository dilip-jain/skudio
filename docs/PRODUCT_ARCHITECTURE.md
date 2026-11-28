# Product and Architecture Decisions

This document records the current product position and the constraints that should guide roadmap work. It is not a claim that every future choice is settled; decisions with evidence gates are marked conditional.

## Positioning and User Promise

**Positioning:** an independent visual development environment for scikit-learn.

**Promise:** build, understand, tune, and evaluate real scikit-learn pipelines visually, then export them as ordinary Python code.

The audience is Python users at different skill levels: learners who benefit from seeing pipeline structure, intermediate users who want less workflow boilerplate, and experienced users who want rapid prototyping. The product should teach the underlying API rather than conceal it. Do not claim that a visual workflow is inherently better, faster, or more accurate than notebooks; validate each claim with users and workflow evidence.

## Core Architecture

```text
UI adapter(s)
      |
      v
Application services
      |
      v
Versioned Pipeline IR -> validator -> normalized execution plan
                                 |                  |
                                 v                  v
                           sklearn compiler     Python AST codegen
                                 |
                                 v
                         isolated local runner
                                 |
                                 v
             structured result + local project artifacts
                                 |
                                 v
                         visualization adapters
```

### IR is the source of truth

- UI state is an editing buffer, not durable workflow semantics. Saved project state is versioned IR plus dataset/run references.
- Nodes, edges, component references, typed parameter values, nested composites, data/column references, search configuration, and non-UI metadata belong in the framework-independent core.
- Layout hints are optional and non-semantic. Changing x/y/color must not change hashes, compilation, codegen, or execution.
- Define edge semantics before using a graph canvas. For the MVP, support a small explicit subset and reject unsupported topology. Ordered `Pipeline` children may remain ordered by definition, but that ordering must be stated and validated. `ColumnTransformer`/`FeatureUnion` branches are explicit composites, not a visual illusion of general arbitrary DAG support.
- Compiler, codegen, validation, and runner consume one normalized validated plan. This is the guardrail against current node-list ordering becoming an accidental second source of truth.
- Every persisted format change has a version and migration fixture. Hashes must be based on documented canonical semantic content.

### sklearn compatibility and introspection

- Instantiate only components present in an approved provider registry. A qualname in an IR document is data, not permission to import arbitrary Python.
- Introspect registered sklearn estimators using `get_params(deep=False)` and constructor signatures to discover parameter names/defaults. Keep curated metadata authoritative for UX: descriptions, docs, categories, widgets, constraints, task suitability, deprecation notes, and basic/advanced tiers.
- Infer controls conservatively. Unknown types, conditional parameters, callable values, and version-sensitive options should not be guessed into unsafe or misleading widgets. Preserve sklearn defaults by omitting unchanged arguments where possible.
- Show supported sklearn versions and test actual instantiation/compile/codegen behavior across them. Constructor signatures and defaults can change; introspection is not a substitute for compatibility tests.
- Keep sklearn objects as the execution model. Do not invent a second estimator abstraction that cannot map transparently to real sklearn.

### Code generation and interoperability

- Generate Python from AST nodes, not interpolated code strings. Source should be formatted, deterministic, readable, and contain normal sklearn imports and constructors.
- `build_pipeline()` is the primary portable artifact. A training script is a separate convenience and must state its split, metric, seed, and data assumptions.
- Test code by parsing/importing it and comparing estimator structure/parameters to compiler output. Do not execute arbitrary user-entered source as part of codegen.
- UI to IR to Python is an MVP requirement. Python to IR is not. A future importer accepts only a documented AST allowlist and never uses `eval` or `exec`.
- Exported Python should not require Skudio at runtime unless the user explicitly chooses a Skudio helper API.

## UI and Local Server Strategy

### Streamlit first

Streamlit is the MVP adapter because it reduces time to test the product workflow. Keep domain decisions in core/application services, and keep Streamlit out of core imports. Build an end-to-end form/stepper first; do not require an infinite canvas to validate the value proposition.

### React only on evidence

React + TypeScript + Vite and a graph editor are a conditional replacement, not an architectural destination that should consume MVP time by default. Move only after user tests identify concrete Streamlit limits in repeated tasks, graph composition, or editing ergonomics. Reuse the same API/services and IR; do not duplicate training/compiler rules in TypeScript.

If adopted, React is built into static assets and included in the Python distribution. Node/npm are build-time contributor/release tools, never end-user runtime prerequisites. Verify wheel contents and launch from an installed wheel, not only a development checkout.

### Local execution and security boundary

- Bind to loopback by default. Do not expose a network service or enable remote access as a casual CLI option.
- Keep datasets, runs, generated code, and model artifacts local. Explain that joblib/pickle artifacts must only be loaded from trusted sources.
- Apply request/upload size limits, safe project path resolution, origin/CSRF protections for browser APIs, and non-leaking token/error logging.
- The default workflow executes registered components. Python snippets, expressions, third-party providers, and user-installed plugins are code execution and need a separate trust prompt and policy.
- Local single-user security is not a sandbox. Any hosted or multi-user product requires a new threat model and explicit isolation work; it is not a deployment toggle on this design.
- Training should not block the UI server. The existing subprocess runner is a suitable MVP isolation base; add persistent job management, cancellation, or queues only when actual workflow needs justify them.

## Data, Evaluation, and Experiments

- Keep dataset inspection/loading separate from UI. pandas DataFrames are the initial tabular interchange format; NumPy arrays are expected outputs from sklearn transformers, not a competing product data model.
- Start with CSV and bounded in-memory datasets. Parquet is a separate optional capability; SQL requires credential, connection, and query safety decisions. Do not recreate all pandas/NumPy visually.
- Treat transformations as typed, serializable operations with equivalent Python export. Provide a custom-code escape hatch only as an explicit advanced trust boundary, not an implicit text field.
- Keep train/test and CV protocols explicit. Prevent leakage by putting learned preprocessing inside the sklearn pipeline and distinguish held-out metrics from cross-validation results.
- Results are structured data, not UI plot objects: metrics, class labels, predictions/probabilities, confusion matrix, curves, importance, duration, logs, artifact references, and errors. Keep large arrays in files with references rather than unbounded JSON.
- The current local project/JSON/run-directory model is appropriate until a demonstrated scale/collaboration need requires another store. Each run should reference immutable dataset identity, IR, config, environment versions, metrics, artifacts, and generated source.
- Run comparison must compare like with like or clearly flag differences in dataset, task, split, scoring, and component versions.

## Provider and Plugin Policy

The core defines a provider contract; sklearn is the default provider. Optional pandas/NumPy transformation providers and third-party estimator providers can follow. Separate providers should not contaminate core imports or make optional packages mandatory.

An extension point is not a sandbox. Discovering/loading an installed Python entry point executes trusted package code. Keep plugin loading explicit/opt-in, document the trust implications, and test provider contracts. Add external packages only when user demand, licensing, install size, version compatibility, and codegen behavior are understood.

## Differentiation and Alternatives

| Alternative | Strong at | Skudio's intended distinction |
| --- | --- | --- |
| Notebooks | Flexible code, exploration, reproducible narratives when authored well | A structured visual workflow with explicit state, validation, repeatable runs, and Python export; notebooks remain a first-class escape hatch |
| Orange and visual workflow tools | Broad visual composition, education, and accessible experimentation | Center ordinary sklearn estimators/pipelines and code interoperability rather than a separate workflow runtime; validate this distinction with users rather than asserting superiority |
| MLJAR and AutoML products | Automated search, reporting, and model selection | Skudio is user-directed pipeline construction and understanding, not automated model selection or a claim to find the best model |
| Other visual ML platforms | Rapid construction and guided workflows | Differentiate through transparent IR, visible sklearn semantics, portable generated source, local-first operation, and disciplined compatibility |

Before external comparison is used in marketing, verify current products and feature claims. The table is a positioning hypothesis, not a competitive analysis or endorsement.

## Open-Source and scikit-learn Relationship

Default to an independent, community-governed open-source project. Use public sklearn APIs; do not modify sklearn core or imply that Skudio is part of, approved by, or maintained by the scikit-learn project. A future collaboration or ecosystem recognition should follow demonstrated utility, maintenance maturity, compatibility discipline, and respectful engagement with maintainers; it is not a dependency for success.

Use the BSD license selected by the project and include required notices for dependencies/assets. Licensing permissiveness does not grant trademark permission. Before publishing branding or marketing:

- Review the current scikit-learn trademark and branding terms directly; terms may change.
- Use “scikit-learn” descriptively to explain compatibility, not in a way that suggests an official product, partnership, or endorsement.
- Avoid scikit-learn logos/visual identity unless expressly permitted.
- Consider qualified legal advice if the project name, logo, domain, or product presentation could imply affiliation. Keep an independent product name and visible non-affiliation statement.

This is product guidance, not legal advice. Record the result of a real trademark review before 1.0.

## UX Quality Principles

- Make the normal path obvious: inspect data, select target/features, configure preprocessing/model, validate, train, inspect results, compare, show Python.
- Display the actual component name, key parameters, and generated source. Explain defaults and evaluation settings where they affect interpretation.
- Treat validation errors as part of the editing workflow: locate the node/parameter, explain why it is invalid, and preserve the user's work.
- Favor compact, task-focused controls over an infinite canvas until branching and nested composition are common user tasks.
- Preserve accessibility: keyboard-operable controls, labels, focus order, color-independent status, readable code, and responsive layouts.
- Measure task completion and user comprehension, not only number of components or graphs created.
