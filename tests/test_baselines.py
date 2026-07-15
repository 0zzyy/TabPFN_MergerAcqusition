"""A5 regression tests: hyperparameter search must be non-trivial (Part C item 6)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from mna_pipeline.baselines import FAST_PARAM_GRIDS, FULL_PARAM_GRIDS, tune_baselines
from mna_pipeline.preprocessing import fit_preprocessor, select_feature_columns, transform
from mna_pipeline.resampling import smote_and_select
from mna_pipeline.splits import split_random
from fixtures import make_synthetic_dataset


def _prepared_split(n=700, seed=1):
    df = make_synthetic_dataset(n=n, seed=seed, positive_rate=0.2)
    X, y = select_feature_columns(df)
    split = split_random(X, y, seed=42)
    preproc = fit_preprocessor(split.X_train)
    X_train_t = transform(preproc, split.X_train)
    X_val_t = transform(preproc, split.X_val)
    X_test_t = transform(preproc, split.X_test)
    Xr, yr, top = smote_and_select(X_train_t, split.y_train, seed=42, sample_size=2000, top_k=8)
    X_val_p = X_val_t[top]
    X_test_p = X_test_t[top]
    scaler = StandardScaler().fit(Xr)
    return (
        scaler.transform(Xr),
        yr.values,
        scaler.transform(X_val_p),
        split.y_val.values,
        scaler.transform(X_test_p),
        split.y_test.values,
    )


def _unique_combo_count(cv_results_):
    return len({tuple(sorted(p.items())) for p in cv_results_["params"]})


def test_full_grids_have_multiple_hyperparameters_and_candidates():
    for name, grid in FULL_PARAM_GRIDS.items():
        assert len(grid) >= 2, f"{name} grid should vary >=2 hyperparameters"
        for param, values in grid.items():
            assert len(values) >= 3, f"{name}.{param} should offer >=3 candidate values"


def test_tune_baselines_full_mode_evaluates_nontrivial_search_for_every_model():
    Xr_s, yr, Xv_s, yv, Xt_s, yt = _prepared_split()
    results = tune_baselines(Xr_s, yr, Xv_s, yv, Xt_s, yt, seed=42, search_mode="full", n_iter=20, cv_folds=5)
    assert set(results) == {"LR", "RF", "GB", "SVM", "MLP"}
    for name, res in results.items():
        n_combos = _unique_combo_count(res["cv_results_"])
        assert n_combos > 1, f"{name} search evaluated only {n_combos} unique combination(s)"
        assert np.isfinite(res["AUC"])
        assert np.isfinite(res["val_AUC"])


def test_tune_baselines_fast_mode_uses_old_toy_grids():
    Xr_s, yr, Xv_s, yv, Xt_s, yt = _prepared_split()
    results = tune_baselines(Xr_s, yr, Xv_s, yv, Xt_s, yt, seed=42, search_mode="fast")
    # GB's fast grid is intentionally a single candidate (preserved verbatim
    # from the pre-fix code); this is only reachable via search_mode="fast",
    # never the default.
    assert _unique_combo_count(results["GB"]["cv_results_"]) == 1


def test_tune_baselines_single_class_target_raises():
    Xr_s, yr, Xv_s, yv, Xt_s, yt = _prepared_split()
    yr_bad = np.zeros_like(yr)
    with pytest.raises(Exception):
        tune_baselines(Xr_s, yr_bad, Xv_s, yv, Xt_s, yt, seed=42, search_mode="full")
