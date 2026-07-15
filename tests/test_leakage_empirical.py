"""Part C item 3: empirical (not structural) leakage check.

Fits the ColumnTransformer on a train partition, then perturbs the
*test* partition (shifts a numeric column's values, injects an
out-of-range / unseen category) and asserts the already-fitted
transformer's learned parameters (imputer statistics_, one-hot
categories_) are completely unaffected, proving no information from the
post-fit "test" perturbation could possibly have leaked backward into
the transformer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mna_pipeline.preprocessing import fit_preprocessor, select_feature_columns, transform
from fixtures import make_synthetic_dataset


def test_perturbing_test_set_after_fit_does_not_affect_fitted_transformer():
    df = make_synthetic_dataset(n=400, seed=7)
    X, y = select_feature_columns(df)
    X_train = X.iloc[:280].copy()
    X_test = X.iloc[280:].copy()

    preproc = fit_preprocessor(X_train)

    numeric_imputer = preproc.named_transformers_["numeric"].named_steps["impute"]
    categorical_imputer = preproc.named_transformers_["categorical"].named_steps["impute"]
    onehot = preproc.named_transformers_["categorical"].named_steps["onehot"]

    stats_before = numeric_imputer.statistics_.copy()
    cat_stats_before = [np.array(a, dtype=object).copy() for a in categorical_imputer.statistics_]
    categories_before = [c.copy() for c in onehot.categories_]

    # Perturb the test set AFTER fitting: shift a numeric column wildly out
    # of the training distribution, and inject a category never seen in
    # training.
    X_test_perturbed = X_test.copy()
    X_test_perturbed["realized_return"] = X_test_perturbed["realized_return"] + 1_000_000.0
    X_test_perturbed["attitude"] = "NeverSeenBeforeCategory"

    # Transforming the perturbed test set must not mutate the fitted
    # transformer's learned parameters.
    _ = transform(preproc, X_test_perturbed)

    stats_after = numeric_imputer.statistics_
    cat_stats_after = categorical_imputer.statistics_
    categories_after = onehot.categories_

    np.testing.assert_array_equal(stats_before, stats_after)
    for before, after in zip(cat_stats_before, cat_stats_after):
        np.testing.assert_array_equal(before, np.array(after, dtype=object))
    for before, after in zip(categories_before, categories_after):
        np.testing.assert_array_equal(before, after)

    # And the unseen category is handled via handle_unknown="ignore"
    # (all-zero one-hot row), not by silently re-fitting.
    out = transform(preproc, X_test_perturbed)
    attitude_cols = [c for c in out.columns if c.startswith("categorical__attitude_")]
    unseen_row = out.iloc[0][attitude_cols]
    assert unseen_row.sum() == 0
