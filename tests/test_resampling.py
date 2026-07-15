from __future__ import annotations

import pytest

from mna_pipeline.preprocessing import fit_preprocessor, select_feature_columns, transform
from mna_pipeline.resampling import smote_and_select
from fixtures import make_synthetic_dataset


def _prepped_train(n=600, seed=1, positive_rate=0.15):
    df = make_synthetic_dataset(n=n, seed=seed, positive_rate=positive_rate)
    X, y = select_feature_columns(df)
    preproc = fit_preprocessor(X)
    Xt = transform(preproc, X)
    return Xt, y


def test_smote_and_select_happy_path_balances_and_selects_top_k():
    Xt, y = _prepped_train(n=600, seed=1, positive_rate=0.15)
    Xr, yr, top = smote_and_select(Xt, y, seed=42, sample_size=2000, top_k=10)
    assert len(top) == 10
    assert list(Xr.columns) == top
    counts = yr.value_counts()
    assert counts.min() / counts.max() > 0.9  # SMOTE balances classes


def test_smote_and_select_respects_sample_size_cap():
    Xt, y = _prepped_train(n=1000, seed=2, positive_rate=0.2)
    Xr, yr, top = smote_and_select(Xt, y, seed=42, sample_size=300, top_k=5)
    assert len(top) == 5


def test_smote_extreme_imbalance_raises_clearly():
    # Fewer minority samples than SMOTE's default k_neighbors+1 requirement.
    Xt, y = _prepped_train(n=300, seed=3, positive_rate=0.01)
    assert y.sum() < 6
    with pytest.raises(ValueError):
        smote_and_select(Xt, y, seed=42, sample_size=2000, top_k=10)


def test_smote_and_select_single_class_raises():
    Xt, y = _prepped_train(n=100, seed=4, positive_rate=0.15)
    y_all_zero = y.copy()
    y_all_zero[:] = 0
    with pytest.raises(ValueError):
        smote_and_select(Xt, y_all_zero, seed=42, sample_size=2000, top_k=10)
