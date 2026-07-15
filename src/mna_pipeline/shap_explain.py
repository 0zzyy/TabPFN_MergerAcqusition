"""SHAP explanations (A3, A9).

Confirmed defect A3: both the background sample and the explained
subset were hardcoded to a single row (`SHAP_BG` / `SHAP_Subset` = 1 in
the GitHub code, `SHAP_BG_K` / `SHAP_SUBSET` = 1 in the thesis's Annex D
script), so every reported "SHAP feature importance" was actually a
single-instance attribution, not a global importance estimate, and the
background distribution used by the explainer was a single data point.

Fix: `build_background` draws a configurable representative background
(default 100 rows) via `shap.sample`; `select_explain_set` draws up to
`min(200, len(X_test))` rows (configurable) from the test set. The
primary output of `explain_models` is the mean absolute SHAP value per
feature across the whole explained subset (`mean_abs_importance`), which
is what should be reported as global feature importance; a single
instance's attribution (`single_instance_importance`, the first row of
the explained subset) is retained separately, only for narrative/example
purposes, and must never be presented as the global importance metric.

`MIN_BACKGROUND_SAMPLES` / `MIN_EXPLAIN_SAMPLES` guard against silently
regressing back to n=1: below this threshold a mean of absolute SHAP
values over so few rows is not a meaningful global-importance estimate,
so `explain_models` warns loudly (or raises under `strict=True`, per
A9) rather than proceeding silently.

A9 fix: SHAP explanation is the explicit "truly optional" example given
in the brief: a single model's explainer failing should not abort the
whole benchmark run. Per-model failures are caught narrowly and logged
as warnings by default, and escalated to hard failures only when
`strict=True` (wired to the CLI's `--strict` flag).
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Set

import numpy as np
import pandas as pd

# shap==0.51.0's own colors module calls a matplotlib Colormap API that
# matplotlib now emits a PendingDeprecationWarning for, purely as an
# import-time side effect unrelated to anything this project's code does.
# Suppressed narrowly, only around this import, so `pytest -W error` is
# not tripped by third-party import noise while still catching real
# warnings from our own code.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import shap

log = logging.getLogger(__name__)

DEFAULT_BACKGROUND_SIZE = 100
DEFAULT_EXPLAIN_SIZE_CAP = 200
MIN_BACKGROUND_SAMPLES = 30
MIN_EXPLAIN_SAMPLES = 30

# Names of models whose fitted estimator itself is a valid SHAP callable
# (tree explainers accept the model directly); every other model name is
# explained via its `predict_proba` method.
_TREE_MODEL_NAMES: Set[str] = {"RF", "GB"}


def build_background(
    X_train: pd.DataFrame, size: int = DEFAULT_BACKGROUND_SIZE, seed: int = 42
) -> pd.DataFrame:
    """Draw a representative SHAP background sample from the training data.

    Input: X_train, the (preprocessed, feature-selected) training data;
    size, target background row count (default 100); seed for
    reproducibility.
    Output: a DataFrame of min(size, len(X_train)) rows, drawn via
    `shap.sample` (not the whole training set, for tractability, and
    never a single row).
    Assumes X_train contains only training-partition rows.
    """
    n = min(size, len(X_train))
    return shap.sample(X_train, n, random_state=seed)


def select_explain_set(
    X_test: pd.DataFrame, size: int | None = None, seed: int = 42
) -> pd.DataFrame:
    """Select the subset of test rows over which SHAP values are computed.

    Input: X_test, the (preprocessed, feature-selected) test data; size,
    target row count (default min(DEFAULT_EXPLAIN_SIZE_CAP, len(X_test)));
    seed for reproducibility.
    Output: a DataFrame of min(size, len(X_test)) rows with a reset
    index.
    Assumes X_test contains only test-partition rows.
    """
    target = size if size is not None else min(DEFAULT_EXPLAIN_SIZE_CAP, len(X_test))
    n = min(target, len(X_test))
    if n >= len(X_test):
        return X_test.reset_index(drop=True)
    return X_test.sample(n=n, random_state=seed).reset_index(drop=True)


def extract_shap_matrix(result) -> np.ndarray:
    """Normalize a shap.Explainer output into a (n_samples, n_features) array.

    Input: result, whatever a `shap.Explainer(...)(X)` call returned
    (an Explanation object, a list of per-class arrays, or a raw
    ndarray of 2 or 3 dimensions).
    Output: a 2D numpy array of shape (n_samples, n_features), taking
    the positive-class slice for any 3D (samples, features, classes)
    output or 2-element class list.
    Assumes `result` corresponds to a binary classification explanation.
    """
    arr = getattr(result, "values", result)
    if isinstance(arr, list):
        arr = arr[1] if len(arr) > 1 else arr[0]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 1] if arr.shape[-1] > 1 else arr[..., 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def explain_models(
    models: Dict[str, Dict[str, object]],
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    min_background: int = MIN_BACKGROUND_SAMPLES,
    min_explain: int = MIN_EXPLAIN_SAMPLES,
    strict: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Compute SHAP global (and single-instance) feature importance per model.

    Input: models, dict of name -> {"model": fitted_estimator, ...} (as
    produced by tune_baselines/train_tabpfn; entries with model=None are
    skipped); X_background/X_explain as produced by build_background /
    select_explain_set; min_background/min_explain sample-size floors;
    strict, whether a per-model SHAP failure or an undersized sample
    raises (True) instead of warning (False).
    Output: dict keyed by model name, each value containing
    "mean_abs_importance" (pd.Series, the primary global-importance
    output), "single_instance_importance" (pd.Series, narrative-only),
    "n_background", and "n_explain".
    Raises ValueError if strict=True and sample sizes are below the
    minimum thresholds, or if strict=True and any individual model's
    SHAP computation fails. Assumes every model in `models` was fit on
    the same feature columns as X_background/X_explain.
    """
    if len(X_background) < min_background or len(X_explain) < min_explain:
        msg = (
            f"SHAP sample sizes too small: background={len(X_background)} "
            f"(minimum {min_background}), explain={len(X_explain)} "
            f"(minimum {min_explain}). Mean |SHAP| over this few rows is not "
            "a reliable global importance estimate."
        )
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    results: Dict[str, Dict[str, object]] = {}
    for name, res in models.items():
        mdl = res.get("model")
        if mdl is None:
            continue
        try:
            log.info(
                "Explaining %s with SHAP (background=%d, explain=%d)",
                name,
                len(X_background),
                len(X_explain),
            )
            pred_fn = mdl if name in _TREE_MODEL_NAMES else mdl.predict_proba
            explainer = shap.Explainer(pred_fn, X_background)
            out = explainer(X_explain)
            mat = extract_shap_matrix(out)
            mean_abs = pd.Series(
                np.abs(mat).mean(axis=0), index=X_explain.columns
            ).sort_values(ascending=False)
            single_instance = pd.Series(
                np.abs(mat[0]), index=X_explain.columns
            ).sort_values(ascending=False)
            results[name] = {
                "mean_abs_importance": mean_abs,
                "single_instance_importance": single_instance,
                "n_background": len(X_background),
                "n_explain": len(X_explain),
            }
        except Exception as e:  # noqa: BLE001 - optional step, see module docstring
            msg = f"SHAP for {name} failed: {e}"
            if strict:
                raise ValueError(msg) from e
            log.warning(msg)
    return results
