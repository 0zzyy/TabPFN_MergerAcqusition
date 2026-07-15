"""Leakage-safe preprocessing (A1, A10).

Confirmed defect A1/A10: the original scripts one-hot encoded `attitude`
and median-imputed numeric columns on the *full* dataframe, before any
train/val/test split, and did so with chained `X[col].fillna(...,
inplace=True)` calls that are unreliable under pandas 2.x
copy-on-write semantics. Both problems leak test-set statistics
(one-hot categories, medians) into training and are fragile.

Fix: split first (see splits.py), then fit a single `ColumnTransformer`
(SimpleImputer + OneHotEncoder) on the training partition only. The
fitted transformer is persisted with joblib so the exact training-time
imputation values and one-hot category vocabulary can be reapplied to
genuinely new data later, without ever re-fitting on data outside the
original training partition. No manual pandas `fillna` is used anywhere
in this module.
"""
from __future__ import annotations

import pathlib
from typing import List, Sequence, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LEAK_PRONE_COLUMNS = (
    "status",
    "target_id",
    "acquirer_id",
    "announcement_date",
    "exit_date",
    "target_naics_code",
)


def select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into a feature matrix X and target y.

    Input: df, the raw loaded dataframe (must contain `status`).
    Output: (X, y) where X excludes `status` and the identifier/date
    columns in LEAK_PRONE_COLUMNS, and y is `status` cast to int.
    Raises ValueError if `status` is missing.
    Assumes no upstream state; safe to call directly on `load_data`'s
    output before any split.
    """
    if "status" not in df.columns:
        raise ValueError("Column 'status' missing from data")
    y = df["status"].astype(int)
    X = df.drop(columns=[c for c in LEAK_PRONE_COLUMNS if c in df.columns])
    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Construct (but do not fit) a leakage-safe ColumnTransformer for X_train.

    Input: X_train, the training-partition feature matrix, used only to
    discover which columns are numeric vs categorical (object/category
    dtype); no values are read for fitting purposes here.
    Output: an unfitted ColumnTransformer with a median-impute branch for
    numeric columns and a most-frequent-impute + one-hot-encode branch
    for categorical columns, with pandas-labeled output.
    Assumes the caller will call `.fit` on this transformer using only
    training-partition rows.
    """
    numeric_cols: List[str] = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols: List[str] = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    transformer.set_output(transform="pandas")
    return transformer


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build and fit a ColumnTransformer using only X_train.

    Input: X_train, the training-partition feature matrix.
    Output: a fitted ColumnTransformer.
    Assumes X_train contains no rows from the validation or test
    partitions; this is the only function in the pipeline allowed to
    call `.fit` on the preprocessing transformer.
    """
    transformer = build_preprocessor(X_train)
    transformer.fit(X_train)
    return transformer


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Apply an already-fitted preprocessor to any partition (train/val/test/new).

    Input: preprocessor, a ColumnTransformer previously fit on training
    data only; X, the feature matrix to transform.
    Output: a pandas DataFrame of transformed features with column names
    derived from the fitted transformer.
    Assumes the caller has already fit `preprocessor` on train-only data;
    this function never fits.
    """
    return preprocessor.transform(X)


def persist_preprocessor(preprocessor: ColumnTransformer, path: pathlib.Path) -> None:
    """Save a fitted preprocessor to disk with joblib.

    Input: preprocessor (fitted), path (destination .joblib file).
    Output: None, writes to disk, creating parent directories if needed.
    Assumes `preprocessor` is already fitted; persisting an unfitted
    transformer is allowed but defeats the purpose of this function.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: pathlib.Path) -> ColumnTransformer:
    """Load a previously persisted fitted preprocessor.

    Input: path to a .joblib file written by `persist_preprocessor`.
    Output: the fitted ColumnTransformer.
    Assumes the file was produced by `persist_preprocessor` in this
    package (or an object joblib-compatible with the same interface).
    """
    return joblib.load(path)
