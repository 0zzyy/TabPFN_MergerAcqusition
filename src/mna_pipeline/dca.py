"""Decision Curve Analysis (A2).

This is the highest-priority confirmed defect in the whole audit and is
reproduced verbatim in the thesis's own Annex D log output:

    DCA computed for models: ['attitude_Friendly', 'realized_return',
    'friendly', 'completion_time', 'attitude_Hostile', 'spread',
    'attitude_Neutral', 'historical_avg_completion_time',
    'base_equity_value', 'day_premium', 'LR', 'RF', 'GB', 'SVM', 'MLP',
    'TabPFN']

`df_dca` was built as `X_test_p.copy()` (the raw, top-10 selected input
features) with 6 model-probability columns appended, and `modelnames`
was then computed as `[n for n in df_dca.columns if n != "status"]`,
which is 10 raw input features plus the 6 real models: every reported
decision curve analysis result includes raw features passed into
`dca()` as if they were prediction models.

Fix, at two independent layers so the bug class cannot recur silently:

1. `build_dca_frame` never starts from a copy of the feature matrix. It
   builds `df_dca` from `status` plus exactly the model-probability
   columns supplied in `model_probs`, and validates every key against
   CANONICAL_MODEL_NAMES, so a caller cannot accidentally smuggle a
   feature column in under a model's name.
2. `run_dca` never derives `modelnames` as "all columns except status".
   It always uses the explicit, hardcoded list returned by
   `build_dca_frame`, and independently asserts (via
   `validate_probability_columns`) that every one of those columns is a
   valid probability in [0, 1] for a sample of rows, raising ValueError
   before calling `dca()` if not.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# The six models under comparison in this study. This tuple is the only
# source of truth for which columns are allowed into df_dca as "model"
# columns; it must never be replaced by "all dataframe columns" logic.
CANONICAL_MODEL_NAMES: Tuple[str, ...] = ("LR", "RF", "GB", "SVM", "MLP", "TabPFN-3")


def build_dca_frame(
    y_test: pd.Series,
    model_probs: Dict[str, np.ndarray],
    expected_models: Sequence[str] = CANONICAL_MODEL_NAMES,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a DCA-ready dataframe from outcomes and model probabilities only.

    Input: y_test, the ground-truth outcome (0/1) aligned with every
    array in model_probs; model_probs, a dict mapping model name to a
    1D array of predicted positive-class probabilities aligned with
    y_test; expected_models, the whitelist a model name must belong to
    (default CANONICAL_MODEL_NAMES).
    Output: (df_dca, modelnames) where df_dca has a "status" column plus
    exactly one column per entry in model_probs, and modelnames is the
    explicit, order-preserving list of those model columns, filtered to
    `expected_models` (models that were not fit, and so are absent from
    model_probs, are simply omitted, never backfilled from other
    columns).
    Raises ValueError if model_probs contains a key outside
    expected_models, or if any probability array's length does not
    match y_test.
    Assumes every array in model_probs is a predicted probability
    (already `predict_proba(...)[:, 1]`), not a raw feature.
    """
    unexpected = set(model_probs) - set(expected_models)
    if unexpected:
        raise ValueError(
            f"build_dca_frame received unexpected model name(s) {sorted(unexpected)}. "
            f"DCA model columns must be restricted to {list(expected_models)} exactly, "
            "never derived from 'all dataframe columns except status'."
        )
    y_arr = np.asarray(y_test)
    df = pd.DataFrame({"status": y_arr})
    modelnames = [m for m in expected_models if m in model_probs]
    for name in modelnames:
        probs = np.asarray(model_probs[name])
        if len(probs) != len(y_arr):
            raise ValueError(
                f"Probability array for model {name!r} has length {len(probs)}, "
                f"expected {len(y_arr)} to match y_test."
            )
        df[name] = probs
    return df, modelnames


def validate_probability_columns(
    df: pd.DataFrame, modelnames: Sequence[str], sample_size: Optional[int] = None, seed: int = 42
) -> None:
    """Assert every column named in modelnames holds valid [0, 1] probabilities.

    Input: df, a dataframe expected to contain one column per name in
    modelnames; modelnames, the columns about to be passed to dca();
    sample_size, number of rows to check (default: all rows); seed for
    reproducible sampling.
    Output: None. Raises ValueError naming the offending column and
    example out-of-range values if any column is missing or contains
    values outside [0, 1] (this is exactly the failure mode a raw
    feature column like `spread` or `day_premium` would trigger).
    Assumes df has already been built (for example by build_dca_frame).
    """
    sample = df if sample_size is None or sample_size >= len(df) else df.sample(
        n=sample_size, random_state=seed
    )
    for name in modelnames:
        if name not in df.columns:
            raise ValueError(
                f"modelnames contains {name!r}, which is not a column of df_dca."
            )
        vals = pd.to_numeric(sample[name], errors="coerce").to_numpy()
        if np.isnan(vals).any() or not np.all((vals >= 0) & (vals <= 1)):
            bad_mask = np.isnan(vals) | (vals < 0) | (vals > 1)
            bad_examples = vals[bad_mask][:5]
            raise ValueError(
                f"Column {name!r} was passed to dca() as a model probability but "
                f"contains values outside [0, 1] (examples: {bad_examples.tolist()}). "
                "This is the exact failure mode of A2: a raw input feature column "
                "being mistaken for a model probability column."
            )


def run_dca(
    df_dca: pd.DataFrame,
    outcome: str,
    modelnames: Sequence[str],
    thresholds: Optional[np.ndarray] = None,
    strict: bool = False,
):
    """Run dcurves.dca after validating modelnames are genuine probabilities.

    Input: df_dca, outcome column name, modelnames (must be the explicit
    list from build_dca_frame, never re-derived from df_dca.columns),
    thresholds (defaults to np.linspace(0.01, 0.99, 50), matching the
    Annex D script), strict (whether a dca() failure raises instead of
    warning, per A9).
    Output: whatever dcurves.dca returns, or None if it failed and
    strict=False.
    Raises ValueError from validate_probability_columns if any
    modelnames column is not a valid probability; re-raises dca()
    failures when strict=True.
    Assumes df_dca and modelnames were produced together by
    build_dca_frame (or an equivalently validated pair).
    """
    validate_probability_columns(df_dca, modelnames)
    log.info("DCA modelnames: %s", list(modelnames))

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)

    from dcurves import dca as _dca

    try:
        result = _dca(
            data=df_dca,
            outcome=outcome,
            modelnames=list(modelnames),
            thresholds=thresholds,
        )
        log.info("DCA computed for models: %s", list(modelnames))
        return result
    except Exception as e:  # noqa: BLE001 - optional/explanatory step, see A9
        msg = f"DCA failed: {e}"
        if strict:
            raise ValueError(msg) from e
        log.warning(msg)
        return None
