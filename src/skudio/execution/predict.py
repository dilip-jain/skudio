"""Load a persisted fitted pipeline and score new data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from skudio.core.errors import SkudioError
from skudio.experiments.project import Project


class PredictionError(SkudioError):
    """Raised when a persisted run cannot be turned into predictions."""


def batch_predict(
    project: Project, run_id: str, df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Predict on `df` using the fitted pipeline saved under `run_id`.

    Returns a tuple `(predictions_df, warnings)`:

    - `predictions_df` is the input frame with a `prediction` column appended,
      and - when the fitted estimator exposes `predict_proba` - one `proba_<class>`
      column per learned class.
    - `warnings` is a list of user-visible strings (currently: extra columns
      that were dropped to match the training schema).

    Raises `PredictionError` when the run has no fitted model, or when the
    input frame is missing columns the pipeline was fit on.
    """
    # pylint: disable=import-outside-toplevel
    import joblib

    run_dir = project.run_dir(run_id)
    joblib_path = run_dir / "pipeline.joblib"
    if not joblib_path.exists():
        raise PredictionError("No fitted model on disk for this run",
            path=str(joblib_path), context={"run_id": run_id},)

    model = joblib.load(joblib_path)
    aligned, warnings = _align_columns(df, model, run_id)

    out = df.copy()
    out["prediction"] = model.predict(aligned)
    _maybe_add_probas(out, model, aligned)
    return out, warnings

def _align_columns(df: pd.DataFrame, model: Any, run_id: str) -> tuple[pd.DataFrame, list[str]]:
    trained: list[str] | None = _feature_names_in(model)
    if trained is None:
        return df, []

    missing = [c for c in trained if c not in df.columns]
    if missing:
        raise PredictionError("Input is missing columns the pipeline was fit on",
            context={"run_id": run_id, "missing": missing},)

    extra = [c for c in df.columns if c not in trained]
    aligned = df.drop(columns=extra).loc[:, trained] if extra else df.loc[:, trained]
    warnings = ([
        f"Dropped {len(extra)} column(s) not present at training time: {extra}"
    ] if extra else [])
    return aligned, warnings

def _feature_names_in(model: Any) -> list[str] | None:
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "steps") and model.steps:
        # `Pipeline` propagates from the first step; fall back defensively.
        names = getattr(model.steps[0][1], "feature_names_in_", None)
    if names is None:
        return None
    return [str(n) for n in names]

def _maybe_add_probas(out: pd.DataFrame, model: Any, aligned: pd.DataFrame) -> None:
    if not hasattr(model, "predict_proba"):
        return
    try:
        proba = model.predict_proba(aligned)
    except Exception:  # pylint: disable=broad-except
        return

    classes = getattr(model, "classes_", None)
    if classes is None or getattr(proba, "shape", (0, 0))[1] != len(classes):
        return

    for i, cls in enumerate(classes):
        out[f"proba_{cls}"] = proba[:, i]
