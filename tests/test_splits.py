from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mna_pipeline.splits import split_random, split_temporal
from fixtures import make_synthetic_dataset


def _xy(df):
    y = df["status"]
    X = df.drop(columns=["status"])
    return X, y


def test_split_random_happy_path_ratios():
    df = make_synthetic_dataset(n=1000, seed=1)
    X, y = _xy(df)
    result = split_random(X, y, seed=42)
    n = len(df)
    assert abs(len(result.X_train) / n - 0.70) < 0.02
    assert abs(len(result.X_val) / n - 0.15) < 0.02
    assert abs(len(result.X_test) / n - 0.15) < 0.02
    assert set(result.X_train.index) & set(result.X_val.index) == set()
    assert set(result.X_train.index) & set(result.X_test.index) == set()
    assert set(result.X_val.index) & set(result.X_test.index) == set()


def test_split_random_single_class_target_completes_trivially():
    # sklearn's stratified split handles a single-class target without
    # raising (there is nothing to balance); this asserts that boundary
    # case does not crash and every partition is entirely that one class.
    df = make_synthetic_dataset(n=200, seed=2, positive_rate=0.15)
    df["status"] = 0
    X, y = _xy(df)
    result = split_random(X, y, seed=42)
    assert len(result.X_train) + len(result.X_val) + len(result.X_test) == len(df)
    assert (result.y_train == 0).all()
    assert (result.y_val == 0).all()
    assert (result.y_test == 0).all()


def test_split_random_too_few_minority_samples_raises():
    # 200 rows with only 2 positives cannot be stratified across two
    # sequential splits (sklearn requires >=2 members per class per split).
    df = make_synthetic_dataset(n=200, seed=2, positive_rate=0.15)
    X, y = _xy(df)
    y = y.copy()
    y[:] = 0
    y.iloc[:2] = 1
    with pytest.raises(ValueError):
        split_random(X, y, seed=42)


def test_split_temporal_strict_date_ordering():
    df = make_synthetic_dataset(n=1000, seed=3)
    X, y = _xy(df)
    dates = df["announcement_date"]
    result = split_temporal(X, y, dates)

    train_dates = dates.loc[result.X_train.index]
    val_dates = dates.loc[result.X_val.index]
    test_dates = dates.loc[result.X_test.index]

    assert train_dates.max() <= val_dates.min()
    assert val_dates.max() <= test_dates.min()

    n = len(df)
    assert abs(len(result.X_train) / n - 0.70) < 0.02
    assert abs(len(result.X_val) / n - 0.15) < 0.02
    assert abs(len(result.X_test) / n - 0.15) < 0.02


def test_split_temporal_rejects_null_dates():
    df = make_synthetic_dataset(n=200, seed=4)
    X, y = _xy(df)
    dates = df["announcement_date"].copy()
    dates.iloc[0] = pd.NaT
    with pytest.raises(ValueError):
        split_temporal(X, y, dates)


def test_split_temporal_extreme_class_imbalance_still_partitions():
    df = make_synthetic_dataset(n=500, seed=5, positive_rate=0.01)
    X, y = _xy(df)
    dates = df["announcement_date"]
    result = split_temporal(X, y, dates)
    assert len(result.X_train) + len(result.X_val) + len(result.X_test) == len(df)
