"""SMOTE oversampling and random-forest-importance feature selection.

This module operates strictly on the already-preprocessed, train-only
feature matrix, so it cannot itself introduce test-set leakage as long
as callers respect that contract (see each function's docstring).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def smote_and_select(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    sample_size: int,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Balance classes with SMOTE and select the top_k most important features.

    Input: X, y are the preprocessed *training-partition only* feature
    matrix and labels; seed for reproducibility; sample_size caps how
    many training rows are sub-sampled (stratified) before SMOTE, for
    runtime reasons; top_k is how many features a RandomForest's
    feature_importances_ selects.
    Output: (Xr_top, yr, top_feature_names) where Xr_top/yr are the
    SMOTE-resampled, top-feature-selected training data.
    Raises whatever SMOTE/RandomForestClassifier raise for degenerate
    inputs (for example a single-class y cannot be resampled by SMOTE).
    Assumes X and y contain only training-partition rows already
    transformed by a train-only-fitted preprocessor; calling this on
    validation or test data would leak those rows into feature selection
    and is a caller error, not something this function can detect.
    """
    subset = min(len(X), sample_size)
    if subset < len(X):
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=subset, stratify=y, random_state=seed
        )
    else:
        X_sub, y_sub = X, y
    sm = SMOTE(random_state=seed)
    Xr, yr = sm.fit_resample(X_sub, y_sub)
    rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(Xr, yr)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top = importances.nlargest(top_k).index.tolist()
    Xr_top = pd.DataFrame(Xr, columns=X.columns)[top]
    log.info("Top %d features selected: %s", top_k, top)
    return Xr_top, yr, top
