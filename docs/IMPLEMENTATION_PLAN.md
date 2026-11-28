# Agent-Ready Implementation Plan

This document divides the product roadmap into proposed commit-sized tasks. Implement in order unless a maintainer explicitly changes dependencies. A task is complete only when its acceptance checks pass and its documentation is updated. Do not fold unrelated cleanup into a task.

## Instructions for Implementation Agents

- Read [ROADMAP.md](ROADMAP.md) and [PRODUCT_ARCHITECTURE.md](PRODUCT_ARCHITECTURE.md) before changing architecture.
- Work on one numbered commit task at a time. Inspect current behavior and tests first; preserve existing public APIs unless the task explicitly changes them.
- Add or update focused tests in the existing test modules. Prefer end-to-end tests for contracts spanning IR, compilation, codegen, persistence, and UI adapters.
- Do not add new runtime dependencies without documenting the need, optional-extra placement, Python support, and packaging impact.
- Never import or instantiate a component solely because a user-supplied qualname names it. Component construction goes through an approved provider registry.
- Do not execute generated Python as a validation strategy. Parse it, import it in an isolated test process where appropriate, and compare the resulting sklearn object with the compiled object.
- Do not change IR/project semantics without a migration or an explicit compatibility decision and regression fixture.
- Do not add React until the evidence gate in the roadmap has been met. Do not add Node as a runtime requirement.
- Report files changed, tests run, compatibility impact, unresolved risks, and whether the acceptance criteria passed. Do not claim a test passed if it was not run.

## 0.3 Alpha: Correctness and Reproducibility

### 0.3.1 - Reconcile release metadata and install contract

**Proposed subject:** `chore: reconcile package version and install documentation`

**Scope:** Align `src/skudio/__init__.py`, `pyproject.toml` metadata behavior, README status/version, optional UI extra, and CLI help/install examples. Establish one authoritative version source and a clean-wheel smoke test. Keep Streamlit optional unless the UI entry point is deliberately made a mandatory dependency.

**Acceptance:** A built wheel installs into a clean Python 3.10+ environment; `skudio --help` works; UI launch gives an actionable message when the UI extra is missing; `--no-browser` does not open a browser; displayed and packaged versions agree. CI tests the supported Python versions actually declared, not merely classifiers.

### 0.3.2 - Persist uploaded datasets as project assets

**Proposed subject:** `feat: persist datasets and bind them to graph references`

**Scope:** Extend the existing project dataset layout and `DatasetRef` use so a dataset can be copied/imported into the project, assigned a stable ID, hashed, and referenced by a graph/run. Keep path resolution confined to project roots where the project owns the asset. Do not make a user upload silently disappear into a temporary path after training.

**Acceptance:** Importing the same bytes has deterministic content identity; a run records dataset ID/hash and can identify its source; persistence round-trips; traversal paths are rejected; existing external CSV training remains supported during migration.

### 0.3.3 - Add persisted-format migration framework

**Proposed subject:** `feat: version and migrate persisted IR documents`

**Scope:** Introduce explicit read/upgrade/write handling for IR documents and project metadata. Add fixtures for current IR `0.1`; choose the next format version only when a semantic change needs it. Unknown future versions fail with a helpful, non-destructive error. Never rewrite a user's project merely by opening it.

**Acceptance:** Current fixtures load unchanged; migration is deterministic and idempotent; canonical hashing behavior is documented; malformed and unsupported versions produce actionable errors; old fixtures remain in tests.

### 0.3.4 - Define and validate graph execution semantics

**Proposed subject:** `feat: validate executable graph topology and port contracts`

**Scope:** Define the supported MVP topology precisely. Validate unique node IDs, edge endpoint and port existence, port cardinality, duplicate edges, component/node-kind compatibility, required inputs/outputs, reachability, acyclicity, and legal terminal nodes. Define how ordered children represent Pipeline steps and how ColumnTransformer/FeatureUnion branches are represented. Reject unsupported shapes rather than guessing. Keep UI layout hints non-semantic.

**Acceptance:** Reversed node list with identical edges has identical meaning; ambiguous/disconnected/cyclic graphs fail with stable issue codes and paths; nested composite checks include full paths; tests cover each supported and rejected topology.

### 0.3.5 - Compile from validated execution order

**Proposed subject:** `fix: compile pipelines from validated graph semantics`

**Scope:** Add or reuse one normalized execution-plan/topological-order helper. Update `core/compiler` so it consumes validated semantics rather than top-level `nodes` order. Preserve sklearn `Pipeline`, `ColumnTransformer`, `FeatureUnion`, and nested Pipeline constructions in the supported subset. Search semantics may remain limited until 0.5.

**Acceptance:** Permuting serialized node order does not alter compiled step order; invalid graphs fail before constructor calls; nested pipeline and transformer parameter behavior matches direct sklearn construction; no arbitrary imports are enabled.

### 0.3.6 - Make generated code match compiled objects

**Proposed subject:** `fix: generate Python from the shared execution plan`

**Scope:** Update AST codegen to consume the same validated plan as compilation. Preserve readable sklearn imports, stable node-derived step names, literals, nested composites, and random-state behavior. Avoid separate graph interpretation in codegen. Add import/constructor parity tests for representative graphs.

**Acceptance:** Generated source parses; a test module can import it and call `build_pipeline`; the returned estimator's deep parameters and nested structure match `compile_graph`; list-order permutation has no effect; unsupported IR values raise a path-aware `CodegenError`.

### 0.3.7 - Specify evaluation and randomness in run configuration

**Proposed subject:** `feat: make training evaluation settings explicit`

**Scope:** Replace scattered defaults with a typed run/evaluation configuration: split strategy, test size, random seed, task, scoring, and optional CV. Keep backward-compatible defaults for current calls. Validate target, empty feature sets, row counts, class counts, and invalid CV before fitting. Do not report CV scores as if they were held-out test metrics.

**Acceptance:** Same graph/data/config and supported deterministic estimator reproduce split indices and metrics; classification stratification fallback is explicit; invalid settings return structured failures; API and generated script document which evaluation path they implement.

## 0.4 MVP: Complete the Local Workflow

### 0.4.1 - Add a durable dataset import and inspection flow

**Proposed subject:** `feat: add project-backed dataset import and inspection`

**Scope:** Connect UI upload to 0.3.2 project dataset storage. Surface row/column counts, dtypes/roles, missingness, unique counts, summary statistics, and a bounded sample. Keep memory and sample limits explicit; CSV is the required format. Parquet may be an optional extra only if dependency and error behavior are tested.

**Acceptance:** Reopening a project restores imported datasets; names are not trusted as filesystem paths; inspection output is deterministic for a fixture; large data inspection does not materialize unbounded samples or plots.

### 0.4.2 - Clarify target, feature, task, and column roles

**Proposed subject:** `feat: validate target and feature selection against dataset schema`

**Scope:** Make target selection, feature inclusion/exclusion, inferred numeric/categorical/datetime roles, and task choice explicit in persisted graph/run configuration. Add warnings for likely target leakage, unsupported values, high-cardinality text, and empty/constant columns; warnings must not silently rewrite data.

**Acceptance:** The selected feature list and target survive save/load and code export; task override is honored; invalid selections fail before training; warnings explain how to resolve the issue.

### 0.4.3 - Make preprocessing presets editable and inspectable

**Proposed subject:** `feat: expose typed preprocessing branches in the workflow`

**Scope:** Keep curated numeric/categorical preprocessing presets but make the represented IR visible and editable. Support ordered imputer/scaler/encoder steps and `ColumnTransformer` branch columns. Ensure custom branch composition compiles and exports identically. Do not implement arbitrary pandas transforms here.

**Acceptance:** Numeric and categorical branch tests include missing values, unknown categories, and empty branch cases; UI state serializes into IR rather than being the only source of truth; generated code reflects every selected step and parameter.

### 0.4.4 - Discover registered estimator parameters safely

**Proposed subject:** `feat: introspect parameters for registered sklearn estimators`

**Scope:** For registered estimator classes only, combine `get_params(deep=False)` and constructor signatures with curated metadata. Infer conservative controls for known scalar/list/enum/bool types; keep unknown or conditional parameters advanced and editable through an explicit typed form. Validate values before compilation and show sklearn defaults accurately.

**Acceptance:** Constructor defaults and displayed values agree across the supported sklearn matrix; deprecated/conditional parameters are not advertised as universally valid; no arbitrary installed classes appear; invalid combinations show component and parameter context.

### 0.4.5 - Curate the first useful sklearn component set

**Proposed subject:** `feat: complete MVP component metadata for common workflows`

**Scope:** Audit roughly 10-15 common estimators across classification and regression plus the transformations required by their workflows. Add descriptions, documentation URLs, category/task, parameter tiers, constraints, and compatibility notes. Avoid count-driven registry expansion.

**Acceptance:** Every listed component can be instantiated with defaults, compiled, trained on a small suitable fixture, and code-generated. Registry tests assert metadata is complete and documentation links use current official API pages.

### 0.4.6 - Make evaluation results task-appropriate and interpretable

**Proposed subject:** `feat: standardize evaluation metrics and result artifacts`

**Scope:** Consolidate classification/regression scoring around the run config. Add label-aware confusion matrix data, optional probability metrics only when supported, regression residual data, and explicit metric names/averaging. Keep result structures UI-independent. Bound or file-store large prediction arrays rather than bloating JSON.

**Acceptance:** Classification and regression fixtures validate metrics independently against sklearn; unsupported metrics are absent with an explanation, not fabricated; result JSON round-trips; plots consume structured results only.

### 0.4.7 - Record reproducible experiment manifests

**Proposed subject:** `feat: persist reproducible run manifests and compare runs`

**Scope:** Extend the existing JSON/directory persistence with timestamps, dataset identity, IR format, Skudio/Python/sklearn versions, run config, metrics, artifact references, and generated source status. Keep storage local and inspectable. Add comparison by compatible task/metric and identify missing artifacts.

**Acceptance:** A saved run can be inspected without loading joblib; run metadata and artifacts are atomic or recoverably written; comparison never mixes task-incompatible metrics; corrupt runs do not hide healthy runs.

### 0.4.8 - Ship the install-to-export vertical slice

**Proposed subject:** `test: cover the documented local MVP workflow`

**Scope:** Add an end-to-end smoke path: clean package install, CLI launch configuration, dataset import/inspection, build, validation, training, persisted result, run comparison, and generated Python. Improve UI errors and empty/loading states revealed by this path.

**Acceptance:** The documented quickstart succeeds from a clean environment without Node, Docker, a notebook, or a cloud account. The test is deterministic and uses a small fixture. Any manual-only browser check is listed explicitly in release notes.

## 0.5: Hyperparameter Search

### 0.5.1 - Define search IR and serialization rules

**Proposed subject:** `feat: define validated grid and randomized search configuration`

**Scope:** Specify `SearchSpace` lowering for GridSearchCV and RandomizedSearchCV, including choice/range/distribution semantics, scoring, CV, refit, `n_iter`, `n_jobs`, and seed. Resolve whether spaces live in node parameters or a dedicated search config without ambiguous duplicate representations.

**Acceptance:** Search configs round-trip and migrate; illegal bounds and strategy-specific fields fail validation; no UI-specific search types enter core IR.

### 0.5.2 - Resolve nested sklearn parameter paths

**Proposed subject:** `feat: map IR search targets to sklearn parameter names`

**Scope:** Define stable mapping from node IDs/composite paths to sklearn Pipeline and ColumnTransformer parameter names. Do not infer paths from labels or list positions. Detect duplicate, stale, and non-searchable targets.

**Acceptance:** Tests assert exact `get_params(deep=True)` keys for flat and nested pipelines; emitted and compiled search objects receive identical grids; invalid paths fail before fitting.

### 0.5.3 - Execute and persist bounded search runs

**Proposed subject:** `feat: execute GridSearchCV and RandomizedSearchCV runs`

**Scope:** Extend isolated runner and result schema for best params, best score, best estimator, bounded CV results, duration, failure state, and cancellation/timeout reporting. Keep computational budgets visible and prevent accidental unbounded `n_iter`/parallelism defaults.

**Acceptance:** Small deterministic search matches direct sklearn results; timeout and estimator failures are structured; best estimator is what is persisted for prediction; result payload size is bounded.

### 0.5.4 - Add tuning controls and result inspection

**Proposed subject:** `feat: expose search spaces and CV results in the UI`

**Scope:** Add explicit Grid/Randomized mode, search-space editor, scoring/CV/budget controls, estimated-fit count, run display, and best-parameter inspection. Keep ordinary training distinct from tuning and generated code includes the selected search object.

**Acceptance:** UI-generated search IR passes the same validator/compiler/codegen tests; controls show estimated fits before launch; a persisted search run reloads and compares correctly.

## 0.6: Core and Compatibility Stability

### 0.6.1 - Define the sklearn support policy and CI matrix

**Proposed subject:** `ci: test declared Python and sklearn compatibility ranges`

**Scope:** Document minimum and tested sklearn versions and Python versions. Add a bounded CI matrix using compatible dependency pins; exercise component manifests, compile, codegen, training, and persistence. Separate supported versions from classifiers that are merely descriptive.

**Acceptance:** Matrix runs in CI; known version-specific defaults and deprecations are tested; raising/dropping support requires a migration note and reviewed dependency policy.

### 0.6.2 - Establish a provider protocol

**Proposed subject:** `refactor: isolate component providers behind a stable protocol`

**Scope:** Extract a typed provider boundary for component listing, metadata, class resolution/construction, and version reporting. Keep sklearn as the first provider and preserve the public registry facade during transition. Providers are trusted Python extensions, not sandboxed plugins.

**Acceptance:** Existing sklearn behavior passes contract tests through the protocol; core compiler does not import provider implementation modules directly; duplicate and unknown component failures remain explicit.

### 0.6.3 - Verify UI-framework independence

**Proposed subject:** `test: enforce framework-independent core imports and contracts`

**Scope:** Add import-boundary tests and headless tests proving IR, validation, compiler, codegen, data contracts, result models, and persistence work without importing Streamlit. Move any product logic found inside UI state/functions to the nearest core/service owner.

**Acceptance:** Core package import works without the UI extra; tests assert the core public surface; UI adapter tests exercise behavior through public core APIs.

### 0.6.4 - Harden local project and artifact lifecycle

**Proposed subject:** `fix: make local project writes atomic and artifact paths safe`

**Scope:** Validate project/run/dataset IDs, use atomic temp-write/replace for manifests, handle interrupted writes, and document joblib trust boundaries. Add artifact cleanup/export operations only where needed by the current UI.

**Acceptance:** Traversal and symlink/path escape attempts are rejected where applicable; interrupted writes do not corrupt an existing manifest; loading a model is clearly limited to artifacts created by a trusted local run.

## 0.7: HTTP API and Alternate UI (Conditional)

Do not start this milestone until the roadmap's user-research gate is met. If Streamlit remains adequate, keep investing in it and skip this release rather than building React speculatively.

### 0.7.1 - Freeze the UI-independent application service contract

**Proposed subject:** `refactor: expose workflow operations through application services`

**Scope:** Provide adapter-neutral operations for project/dataset listing, graph validation, component metadata, run submission/status/result, prediction, and code export. The Streamlit adapter consumes these services; no FastAPI dependency enters the core.

**Acceptance:** Service tests are headless; existing Streamlit workflow remains functional; request/response models do not expose Streamlit types or mutable session state.

### 0.7.2 - Add a loopback-only HTTP adapter

**Proposed subject:** `feat: serve local workflow API on loopback`

**Scope:** Add FastAPI or Starlette as an optional UI/server extra, expose the frozen service contract, enforce loopback binding, request-size limits, origin/CSRF protections appropriate to the local browser threat model, and token handling that does not unnecessarily leak secrets into logs/history. Keep arbitrary code execution absent.

**Acceptance:** API contract tests cover auth failures, invalid graph payloads, upload limits, path traversal, and run lifecycle; non-loopback binding is rejected by default; no data is sent off machine.

### 0.7.3 - Package a React UI as build-time assets

**Proposed subject:** `build: bundle the alternate frontend into the Python wheel`

**Scope:** Add a TypeScript/Vite frontend only after the gate. Use a graph editor library only if user tests confirm graph editing is the needed interaction. Build static assets in release/build CI and include them in the wheel; Node is a development/build dependency only.

**Acceptance:** Clean Python install serves the bundled UI without Node/npm; asset paths work under the local server; reproducible build instructions and license inventory are included; Streamlit remains available during the adapter transition.

### 0.7.4 - Port one workflow slice and compare adapters

**Proposed subject:** `feat: implement dataset-to-export flow in the alternate UI`

**Scope:** Port one complete vertical slice through the shared API, starting with dataset inspection, feature/target selection, pipeline editing, validation, training, results, and generated Python. Avoid parallel duplicated backend logic.

**Acceptance:** Same fixture and IR produce matching compiled object, metrics, persisted run, and exported source in both adapters; keyboard and narrow viewport workflows are usable; adapter comparison results are documented.

## 1.0: Stable Local Release

### 1.0.1 - Freeze public IR and project compatibility policy

**Proposed subject:** `docs: publish IR and project format compatibility policy`

**Scope:** Publish supported public APIs, persisted format guarantees, migration behavior, provider compatibility, and deprecation policy. Mark internal APIs clearly. Do not promise stability for implementation details accidentally imported by users.

**Acceptance:** A new version can read all documented prior project formats; examples are validated in CI; compatibility policy names the supported upgrade path.

### 1.0.2 - Complete release and security review

**Proposed subject:** `security: audit local server and exported artifact boundaries`

**Scope:** Review bind address, token flow, browser origin/CSRF behavior, upload parsing/limits, project path validation, dependency surface, pickle/joblib caveats, and error/log redaction. Include threat model for a single-user local app and explicit exclusions for hosted multi-user use.

**Acceptance:** Security tests cover the documented threats; limitations are visible in docs; no unresolved critical/high issue remains without a maintainer-approved exception.

### 1.0.3 - Verify wheel, CLI, and browser distribution

**Proposed subject:** `ci: verify release wheel across supported environments`

**Scope:** Build sdist and wheel, install from wheel in clean environments, test CLI and optional extras, verify bundled assets if React is active, and exercise a minimal headless workflow on supported OS/Python versions.

**Acceptance:** Release artifacts contain only intended files; no source checkout or Node runtime is needed; install and launch commands match documentation.

### 1.0.4 - Publish user and contributor documentation

**Proposed subject:** `docs: publish quickstart workflow and troubleshooting guide`

**Scope:** Document installation, local launch, project/dataset layout, supported components, evaluation semantics, generated code, run comparison, compatibility, privacy, troubleshooting, and contribution/release process. State plainly that Skudio is independent and not endorsed by scikit-learn.

**Acceptance:** All examples execute in CI where practical; docs distinguish current capabilities from planned capabilities; support and issue-reporting paths are clear.

### 1.0.5 - Release candidate acceptance and 1.0 sign-off

**Proposed subject:** `test: gate the 1.0 local workflow release`

**Scope:** Run the full quality gate, clean-install user journey, supported-version matrix, migration fixtures, generated-code parity, and manual browser accessibility/usability checklist. Fix only release-blocking defects in this task.

**Acceptance:** All release gates are green and signed off; changelog lists breaking changes and known limitations; artifacts are built from a tagged source state by the maintainer. No agent creates or pushes a tag without explicit authorization.

## Post-1.0: Evidence-Led Extensions

Each item below is a separate milestone and should not block the 1.0 workflow.

### 1.x.1 - Import a constrained sklearn Python subset

**Proposed subject:** `feat: import supported sklearn pipeline source into IR`

Accept only a documented AST subset: direct registered sklearn constructors, `Pipeline`, `ColumnTransformer`, `FeatureUnion`, basic literals, and supported nesting. Never `eval`, `exec`, import, or run the submitted source. Unsupported syntax must produce source-located diagnostics. Acceptance requires round-trip tests: source to IR to normalized Python and equivalent estimator parameters.

### 1.x.2 - Publish external provider extension points

**Proposed subject:** `feat: support opt-in third-party component providers`

Stabilize package entry-point discovery only after the provider protocol has been exercised. Loading an installed provider executes trusted Python code; make that explicit, offer opt-in/allowlist behavior, and keep providers disabled by default if they are not explicitly configured. Test provider isolation at the API boundary, not by claiming sandboxing.

### 1.x.3 - Add a first external estimator provider

**Proposed subject:** `feat: add an optional external estimator provider`

Choose XGBoost, LightGBM, or CatBoost only from user demand and license/dependency review. Place its dependency in an optional extra, avoid importing it from core, and add provider contract, compile, codegen, and training tests in a compatible environment.

### 1.x.4 - Add bounded pandas/NumPy and data-source integrations

**Proposed subject:** `feat: add demand-backed data transformation providers`

Add a small, typed set of operations (select, rename, filter, simple feature derivation) only with clear semantics, reproducible code export, and leakage-aware execution ordering. Consider SQL/Parquet separately with explicit optional dependencies and safe credential handling. Keep arbitrary expressions out of the default workflow.

### 1.x.5 - Consider advanced experiment integrations

**Proposed subject:** `feat: add an optional experiment tracking integration`

Integrate Optuna or an external tracking system only when native local experiment comparison reaches a demonstrated limit. Keep native project/run manifests portable and make any external service opt-in; never require cloud connectivity for local workflows.

## Definition of Done for Every Task

- Focused tests pass, and existing tests in the changed ownership area still pass.
- User-visible behavior, IR/project compatibility, and dependency changes are documented.
- Generated code remains valid ordinary Python and equivalent to compiler output for the supported graph subset.
- Errors are actionable and do not silently drop graph edges, parameters, metrics, or artifacts.
- The task does not expand the threat model or product scope without updating the product/architecture decision record.
