from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mna_pipeline.preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    load_preprocessor,
    persist_preprocessor,
    select_feature_columns,
    transform,
)
from fixtures import make_synthetic_dataset


def test_select_feature_columns_drops_leak_prone_columns():
    df = make_synthetic_dataset(n=100, seed=1)
    X, y = select_feature_columns(df)
    for col in ("status", "target_id", "acquirer_id", "announcement_date", "exit_date", "target_naics_code"):
        assert col not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_fit_and_transform_happy_path():
    df = make_synthetic_dataset(n=300, seed=2)
    X, y = select_feature_columns(df)
    X_train = X.iloc[:200]
    X_test = X.iloc[200:]

    preproc = fit_preprocessor(X_train)
    out_train = transform(preproc, X_train)
    out_test = transform(preproc, X_test)

    assert not out_train.isna().any().any()
    assert not out_test.isna().any().any()
    assert any(c.startswith("categorical__attitude_") for c in out_train.columns)
    assert list(out_train.columns) == list(out_test.columns)


def test_persist_and_load_roundtrip(tmp_path):
    df = make_synthetic_dataset(n=150, seed=3)
    X, y = select_feature_columns(df)
    preproc = fit_preprocessor(X)
    path = tmp_path / "preproc.joblib"
    persist_preprocessor(preproc, path)
    loaded = load_preprocessor(path)
    out_a = transform(preproc, X)
    out_b = transform(loaded, X)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_all_missing_numeric_column_does_not_crash():
    df = make_synthetic_dataset(n=100, seed=4)
    X, _ = select_feature_columns(df)
    X = X.copy()
    X["realized_return"] = np.nan
    # SimpleImputer legitimately warns (not an error) that an all-NaN
    # column has no observed values to impute from, then drops it. This
    # fires on both fit and transform.
    with pytest.warns(UserWarning, match="Skipping features without any observed values"):
        preproc = fit_preprocessor(X)
        out = transform(preproc, X)
    # SimpleImputer drops all-NaN columns by default; the pipeline must not crash.
    assert "numeric__realized_return" not in out.columns
    assert not out.isna().any().any()


def test_empty_dataframe_raises_clearly():
    df = make_synthetic_dataset(n=10, seed=5)
    X, _ = select_feature_columns(df)
    empty = X.iloc[0:0]
    with pytest.raises(ValueError):
        fit_preprocessor(empty)
