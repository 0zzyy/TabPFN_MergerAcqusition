"""A3 regression tests: SHAP must never silently regress to n=1 samples."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from mna_pipeline.shap_explain import (
    MIN_BACKGROUND_SAMPLES,
    MIN_EXPLAIN_SAMPLES,
    build_background,
    explain_models,
    select_explain_set,
)


def _fitted_rf():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + X["f1"] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    return rf, X


def test_build_background_and_select_explain_set_default_sizes():
    _, X = _fitted_rf()
    bg = build_background(X.iloc[:200], size=100, seed=0)
    ex = select_explain_set(X.iloc[200:], size=None, seed=0)
    assert len(bg) == 100
    assert len(ex) == min(200, len(X.iloc[200:]))
    assert len(bg) > 1
    assert len(ex) > 1


def test_explain_models_happy_path_mean_abs_importance():
    rf, X = _fitted_rf()
    bg = build_background(X.iloc[:200], size=100, seed=0)
    ex = select_explain_set(X.iloc[200:], size=80, seed=0)
    models = {"RF": {"model": rf}}
    out = explain_models(models, bg, ex)
    assert "RF" in out
    assert out["RF"]["n_background"] == 100
    assert out["RF"]["n_explain"] == 80
    assert set(out["RF"]["mean_abs_importance"].index) == set(X.columns)
    assert set(out["RF"]["single_instance_importance"].index) == set(X.columns)


def test_explain_models_skips_none_model():
    rf, X = _fitted_rf()
    bg = build_background(X.iloc[:200], size=100, seed=0)
    ex = select_explain_set(X.iloc[200:], size=80, seed=0)
    out = explain_models({"Missing": {"model": None}}, bg, ex)
    assert out == {}


def test_explain_models_undersized_warns_by_default(caplog):
    caplog.set_level(logging.WARNING)
    rf, X = _fitted_rf()
    tiny_bg = X.iloc[:1]
    tiny_ex = X.iloc[1:2]
    out = explain_models({"RF": {"model": rf}}, tiny_bg, tiny_ex, strict=False)
    assert any("too small" in rec.message for rec in caplog.records)
    # Still attempts to compute something rather than silently doing nothing.
    assert isinstance(out, dict)


def test_explain_models_undersized_raises_when_strict():
    rf, X = _fitted_rf()
    tiny_bg = X.iloc[:1]
    tiny_ex = X.iloc[1:2]
    with pytest.raises(ValueError):
        explain_models({"RF": {"model": rf}}, tiny_bg, tiny_ex, strict=True)


def test_explain_models_per_model_failure_warns_by_default_raises_when_strict(caplog):
    class Broken:
        def predict_proba(self, X):
            raise RuntimeError("boom")

    _, X = _fitted_rf()
    bg = build_background(X.iloc[:200], size=100, seed=0)
    ex = select_explain_set(X.iloc[200:], size=80, seed=0)

    caplog.set_level(logging.WARNING)
    out = explain_models({"Broken": {"model": Broken()}}, bg, ex, strict=False)
    assert out == {}
    assert any("SHAP for Broken failed" in rec.message for rec in caplog.records)

    with pytest.raises(ValueError):
        explain_models({"Broken": {"model": Broken()}}, bg, ex, strict=True)
