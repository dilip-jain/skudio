# Skudio Product and Release Roadmap

Status: proposed planning baseline, 2026-09-28  
Audience: maintainers, contributors, and implementation agents

## Product Thesis

Skudio is an independent visual development environment for real scikit-learn workflows. Its promise is: **build, understand, tune, and evaluate scikit-learn pipelines visually, then export ordinary Python code**. It is not an AutoML system and must not replace sklearn objects with a proprietary modeling abstraction.

The product should keep a useful two-way relationship between the UI and Python: the UI makes construction and inspection easier; exported Python remains understandable, portable, and usable without Skudio. Initial scope is UI to IR to Python. Python to UI is a later, explicitly bounded feature.

## Existing Baseline

The repository already contains:

- A Pydantic IR with nodes, edges, nested composite graphs, component references, parameters, dataset schemas, and search-space value types.
- Graph validation, sklearn object compilation, AST-based Python generation, and a Streamlit UI builder.
- A curated sklearn component manifest and registry, CSV inspection, EDA, feature/target selection, preprocessing branches, classification/regression estimators, training, cross-validation, metrics, plots, prediction, run comparison, and code download.
- A CLI that launches Streamlit on loopback, plus subprocess training and JSON/joblib run persistence.

The plan is additive, not a greenfield rewrite. Known gaps to resolve deliberately:

- `Graph.edges` are validated in part but compiler and code generator principally use node-list order. The IR has not yet earned the title of source of truth.
- Composite pipeline ordering is implicit in child-list order; search value types exist but user-facing tuning is incomplete.
- Dataset loading is CSV-only; uploaded datasets are not yet a durable, first-class project asset in the displayed workflow.
- Component parameter forms rely on curated descriptors; `get_params`-based discovery and robust parameter type/widget inference are not complete.
- IR and project formats have version markers but no migration framework.
- The existing server/UI boundary is an adapter boundary, not a stable HTTP API; Streamlit remains the active UI.
- README and package version declarations currently disagree and must be reconciled before publishing a release.

Do not interpret this list as a claim that all adjacent functionality is absent. Add tests around existing behavior before replacing it.

## Release Map

Release numbers below are planning labels, not promises. They are intentionally alpha milestones until packaging, support policy, and compatibility are verified.

| Milestone | Outcome | Exit gate |
| --- | --- | --- |
| 0.3 Alpha: Correctness | Versioned IR, edge semantics, compiler/codegen agreement, durable dataset reference | Reordered node lists do not change graph meaning; invalid graphs fail before execution; generated code constructs an equivalent sklearn object |
| 0.4 MVP: Local workflow | Reliable CSV-to-trained-model workflow, supported estimator controls, explainable evaluation, durable runs | A new user can install, launch, inspect data, build, train, compare, and export with no notebook or Node installation |
| 0.5 Tuning | Useful GridSearchCV and RandomizedSearchCV workflows | Nested parameter paths, CV configuration, best estimator, and CV results survive persistence and match sklearn |
| 0.6 Core stability | Compatibility policy, provider boundary, migrations, adapter-independent core | Core imports and tests without importing Streamlit; supported sklearn matrix passes the contract suite |
| 0.7 Alternate UI (conditional) | Local HTTP API and optionally bundled React UI | Proceed only if user research demonstrates a workflow that Streamlit cannot support well; installed package serves bundled assets without Node |
| 1.0 Stable local product | Documented, reproducible, secure local workflow with compatibility and support policy | Release checklist passes on supported Python/OS/sklearn matrix; clean wheel install exercises CLI and full workflow |
| 1.x Extensions | Bounded Python import, external providers, and additional data integrations | Each extension has demonstrated demand, a versioned contract, tests, and a clear trust model |

## Release Strategy

1. Keep the Streamlit UI as the MVP learning instrument. Improve UX and core contracts without moving product logic into Streamlit session state.
2. Make one end-to-end workflow reliable before broadening component count or introducing another UI framework.
3. Version persisted IR and project data before changing their semantics. Keep a reader for each supported prior format or provide a clear, tested migration error.
4. Treat the compiler, validator, code generator, and runner as consumers of one validated execution plan. No consumer should silently invent different graph ordering or parameter meaning.
5. Add estimator support through a registered provider plus curated metadata. Introspection may reveal constructor parameters; it must not make arbitrary installed Python classes executable.
6. Keep training local and synchronous from the user's perspective, but isolated from the UI process. Add job lifecycle/polling only when the API/UX requires it.
7. Do not add a database, hosted service, distributed queue, authentication system, arbitrary expression runtime, or mandatory Node installation to the MVP.
8. Keep Optuna, SQL, Python-to-IR, non-sklearn providers, and broad pandas/NumPy visual programming behind demand and architecture gates.

## Evidence Gates

Before starting the React milestone, run moderated task tests with beginners and Python/sklearn users. Record task completion, time-to-first-valid-pipeline, validation recovery, understanding of generated code, and whether users can reproduce the exported result outside Skudio. Move to React only if recurring interaction problems are specifically caused by Streamlit's interaction or graph-editing constraints, not merely because React is familiar or fashionable.

Before calling the product stable, run a clean-install test from the built wheel, document supported platform and dependency ranges, and publish the exact sklearn compatibility matrix. Track actual issues from users rather than adding features based on speculative platform completeness.

## Non-Goals for the MVP

- AutoML, automatic model selection, or claims that Skudio finds the best model.
- Recreating all of pandas, NumPy, notebooks, or scikit-learn's API in visual form.
- Arbitrary Python execution or arbitrary Python parsing.
- Cloud hosting, multi-user collaboration, distributed execution, auth, or a server-side user-data service.
- An official scikit-learn product, endorsed project, or modification to scikit-learn core.
- A requirement that users install Node, npm, Docker, or a separate frontend application.

## How to Execute the Plan

Use [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) as the commit-level source of work and [PRODUCT_ARCHITECTURE.md](PRODUCT_ARCHITECTURE.md) for decisions and constraints. Each numbered commit is one reviewable change with its own tests. Keep commits in order within a milestone; parallelize only independent test/documentation work. The commit subjects are proposed titles, not authorization for an agent to create Git commits. Integrators create commits according to repository policy.
