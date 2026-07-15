"""A2 regression tests: DCA modelnames must never be derived from
'all columns except status', and must be exactly the model probability
columns. This directly reproduces the bug found in the thesis's own
Annex D log output:

    DCA computed for models: ['attitude_Friendly', 'realized_return',
    'friendly', 'completion_time', 'attitude_Hostile', 'spread',
    'attitude_Neutral', 'historical_avg_completion_time',
    'base_equity_value', 'day_premium', 'LR', 'RF', 'GB', 'SVM', 'MLP',
    'TabPFN']
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mna_pipeline.dca import (
    CANONICAL_MODEL_NAMES,
    build_dca_frame,
    run_dca,
    validate_probability_columns,
)


def _rng():
    return np.random.default_rng(0)


def test_build_dca_frame_happy_path_exact_six_columns():
    rng = _rng()
    n = 100
    y_test = pd.Series(rng.integers(0, 2, n))
    model_probs = {name: rng.uniform(0, 1, n) for name in CANONICAL_MODEL_NAMES}
    df, modelnames = build_dca_frame(y_test, model_probs)
    assert modelnames == list(CANONICAL_MODEL_NAMES)
    assert set(df.columns) == {"status", *CANONICAL_MODEL_NAMES}


def test_build_dca_frame_rejects_unexpected_model_name():
    rng = _rng()
    n = 50
    y_test = pd.Series(rng.integers(0, 2, n))
    model_probs = {
        "LR": rng.uniform(0, 1, n),
        "realized_return": rng.normal(0, 1, n),  # a raw feature, not a model
    }
    with pytest.raises(ValueError):
        build_dca_frame(y_test, model_probs)


def test_build_dca_frame_omits_models_that_never_fit():
    rng = _rng()
    n = 40
    y_test = pd.Series(rng.integers(0, 2, n))
    model_probs = {"LR": rng.uniform(0, 1, n), "RF": rng.uniform(0, 1, n)}
    df, modelnames = build_dca_frame(y_test, model_probs)
    assert modelnames == ["LR", "RF"]


def test_validate_probability_columns_catches_the_historical_bug_pattern():
    """Simulate the exact original bug: modelnames derived as 'all columns
    except status' on a df that (like the real df_dca) still contains raw
    input feature columns such as spread/day_premium alongside the model
    probability columns.
    """
    rng = _rng()
    n = 60
    df_dca_contaminated = pd.DataFrame(
        {
            "status": rng.integers(0, 2, n),
            "attitude_Friendly": rng.integers(0, 2, n).astype(float),
            "spread": rng.normal(0.1, 0.05, n),  # NOT a probability
            "day_premium": rng.normal(0.3, 0.1, n),  # NOT a probability
            "LR": rng.uniform(0, 1, n),
            "RF": rng.uniform(0, 1, n),
        }
    )
    buggy_modelnames = [c for c in df_dca_contaminated.columns if c != "status"]
    assert "spread" in buggy_modelnames  # confirms this reproduces the historical bug shape

    with pytest.raises(ValueError):
        validate_probability_columns(df_dca_contaminated, buggy_modelnames)


def test_run_dca_real_call_passes_exactly_six_canonical_modelnames(caplog):
    """Runs the real dcurves.dca() (no gating/network required) and asserts
    the exact modelnames list logged and used contains only the six models.
    """
    import logging

    caplog.set_level(logging.INFO)
    rng = _rng()
    n = 200
    y_test = pd.Series(rng.integers(0, 2, n))
    model_probs = {name: rng.uniform(0.01, 0.99, n) for name in CANONICAL_MODEL_NAMES}
    df_dca, modelnames = build_dca_frame(y_test, model_probs)

    result = run_dca(df_dca, "status", modelnames)

    assert modelnames == list(CANONICAL_MODEL_NAMES)
    assert len(modelnames) == 6
    assert result is not None
    logged_lines = [r.message for r in caplog.records if "DCA computed for models" in r.message]
    assert logged_lines, "expected a log line reporting the exact modelnames passed to dca()"
    for name in CANONICAL_MODEL_NAMES:
        assert name in logged_lines[-1]
    # None of the raw feature names from the historical bug should ever appear.
    for leaked in ("spread", "day_premium", "attitude_Friendly", "realized_return"):
        assert leaked not in logged_lines[-1]


def test_run_dca_warns_by_default_and_raises_when_strict_on_failure(monkeypatch):
    rng = _rng()
    n = 30
    y_test = pd.Series(rng.integers(0, 2, n))
    model_probs = {"LR": rng.uniform(0, 1, n)}
    df_dca, modelnames = build_dca_frame(y_test, model_probs)

    import mna_pipeline.dca as dca_mod

    def broken_dca(*args, **kwargs):
        raise RuntimeError("simulated dcurves failure")

    monkeypatch.setattr("dcurves.dca", broken_dca)

    result = run_dca(df_dca, "status", modelnames, strict=False)
    assert result is None

    with pytest.raises(ValueError):
        run_dca(df_dca, "status", modelnames, strict=True)
