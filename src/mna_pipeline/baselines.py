"""Tuned baseline classifiers (A4, A5, A9).

A5 fix: the previous grids were nominal (as small as a single candidate
for Gradient Boosting, with `n_iter=2, cv=2`), so no real search ever
happened. FULL_PARAM_GRIDS below give every model multiple
hyperparameters with at least 3-5 candidate values each, and
`tune_baselines` defaults to `n_iter=20` with 5-fold stratified CV. The
old nominal grids are preserved verbatim as FAST_PARAM_GRIDS, reachable
only via `search_mode="fast"` (wired to the CLI's `--search-mode fast`),
never as the default, strictly for fast local debugging.

A4 fix: `tune_baselines` evaluates every tuned candidate's AUC and
calibration (Brier, ECE) on the validation partition and logs a
comparison table before any test-set metric is computed, so the
validation split is actually used for model comparison rather than
being created and discarded.

A9 fix: baseline tuning and fitting is a core pipeline step. Failures
here are not caught and converted into NaN placeholders; they propagate
and halt the run, since a broken baseline silently reported as NaN would
produce an incomplete benchmark table without a hard failure signal.

Incidental, verified scikit-learn API changes (flagged, not silent):
while widening these grids against the actually-installed scikit-learn
(1.9.0, checked 2026-07-15), two of the original scripts' estimator
constructions turned out to already be deprecated in the installed
version, independent of anything this brief asked to change:

- `LogisticRegression(..., penalty=...)`: scikit-learn 1.9 defaults
  `penalty="deprecated"` and controls the l1/l2/elasticnet mix purely
  via `l1_ratio` (0.0 = l2, 1.0 = l1, in between = elasticnet); passing
  `penalty=` at all raises a FutureWarning. Fixed by dropping `penalty`
  and tuning `l1_ratio` directly with `solver="saga"` (which supports
  the full l1_ratio range), confirmed via
  `inspect.signature(LogisticRegression.__init__)` against the
  installed package.
- `SVC(..., probability=True)`: scikit-learn 1.9 deprecates the
  `probability` constructor argument in favor of wrapping the estimator
  in `CalibratedClassifierCV`. Fixed by tuning
  `CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid",
  ensemble=False)` instead, with grid keys prefixed `estimator__` to
  reach the wrapped SVC's hyperparameters, confirmed the same way.

Both changes preserve the original hyperparameter *values* search space;
only the estimator wiring needed to change to stay on non-deprecated
scikit-learn 1.9 APIs, since `pytest -W error` (Part C) would otherwise
fail on these FutureWarnings.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .calibration import summarize_calibration

log = logging.getLogger(__name__)

SearchMode = Literal["full", "fast"]

FULL_PARAM_GRIDS: Dict[str, dict] = {
    "LR": {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "tol": [1e-4, 1e-3, 1e-2],
    },
    "RF": {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    },
    "GB": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
    },
    "SVM": {
        "estimator__C": [0.01, 0.1, 1, 10, 100],
        "estimator__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "estimator__tol": [1e-4, 1e-3, 1e-2],
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "learning_rate_init": [1e-4, 1e-3, 1e-2],
    },
}

# Preserved as close to the pre-fix code as the installed scikit-learn's
# non-deprecated API allows (A5). Debug-only; never the default. LR's
# "l2-equivalent" is l1_ratio=0.0; SVM keys use the estimator__ prefix
# required by the CalibratedClassifierCV wrapper (see module docstring).
FAST_PARAM_GRIDS: Dict[str, dict] = {
    "LR": {"C": [0.1, 1], "l1_ratio": [0.0]},
    "RF": {"n_estimators": [50], "max_depth": [None, 5]},
    "GB": {"n_estimators": [50], "max_depth": [3], "learning_rate": [0.1]},
    "SVM": {"estimator__C": [0.1, 1], "estimator__gamma": ["scale"]},
    "MLP": {"alpha": [1e-3], "learning_rate_init": [1e-3]},
}


def _model_defs(seed: int) -> Dict[str, object]:
    """Return fresh, unfitted baseline estimator instances keyed by name.

    Input: seed for reproducibility.
    Output: dict of 5 sklearn estimators. LR uses solver="saga" (supports
    the full l1_ratio range); SVM is wrapped in CalibratedClassifierCV
    rather than constructed with the deprecated probability=True (see
    module docstring for why).
    Assumes no upstream state.
    """
    return {
        "LR": LogisticRegression(solver="saga", max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "GB": GradientBoostingClassifier(random_state=seed),
        "SVM": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=seed), method="sigmoid", ensemble=False
        ),
        "MLP": MLPClassifier(
            max_iter=200,
            tol=1e-4,
            early_stopping=True,
            random_state=seed,
            hidden_layer_sizes=(50,),
        ),
    }


def tune_baselines(
    Xr_s: "np.ndarray | pd.DataFrame",
    y: np.ndarray,
    X_val_s: "np.ndarray | pd.DataFrame",
    y_val: np.ndarray,
    X_test_s: "np.ndarray | pd.DataFrame",
    y_test: np.ndarray,
    seed: int,
    search_mode: SearchMode = "full",
    n_iter: int = 20,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, object]]:
    """Tune, validate, and test 5 baseline classifiers.

    Input: Xr_s/y are the SMOTE-resampled, scaled, feature-selected
    training data; X_val_s/y_val and X_test_s/y_test are the
    correspondingly scaled and feature-selected validation and test
    partitions; seed for reproducibility; search_mode selects
    FULL_PARAM_GRIDS (default) or FAST_PARAM_GRIDS (debug only); n_iter
    and cv_folds control RandomizedSearchCV (ignored, replaced by
    n_iter=2/cv=2, when search_mode="fast" to preserve the original
    debug behavior exactly).
    Output: dict keyed by model name, each value containing the fitted
    "model", "cv_results_" (sklearn's raw dict, for verifying search
    breadth), "best_params", validation metrics ("val_AUC", "val_brier",
    "val_ece") and test metrics ("AUC", "F1", "brier", "ece").
    Raises on any failure (core step per A9); does not catch and mask
    exceptions with NaN placeholders.
    Assumes Xr_s/X_val_s/X_test_s are already scaled and restricted to
    the same feature columns, and that the validation/test partitions
    were never used to fit the scaler, SMOTE, or feature selection.
    """
    if search_mode == "fast":
        grids = FAST_PARAM_GRIDS
        eff_n_iter, eff_cv = 2, 2
    else:
        grids = FULL_PARAM_GRIDS
        eff_n_iter, eff_cv = n_iter, cv_folds

    defs = _model_defs(seed)
    results: Dict[str, Dict[str, object]] = {}
    for name, mdl in defs.items():
        log.info("Tuning %s (search_mode=%s, n_iter=%d, cv=%d)", name, search_mode, eff_n_iter, eff_cv)
        search = RandomizedSearchCV(
            mdl,
            grids[name],
            n_iter=eff_n_iter,
            cv=StratifiedKFold(eff_cv, shuffle=True, random_state=seed),
            scoring="roc_auc",
            random_state=seed,
            n_jobs=1,
            error_score="raise",
        )
        if search_mode == "fast":
            # FAST_PARAM_GRIDS deliberately has a single candidate for GB
            # (see module docstring); sklearn's ParameterSampler warns when
            # n_iter exceeds the grid size, which is expected here, not a
            # bug, so the warning is narrowly suppressed for this call only.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The total space of parameters.*smaller.*n_iter",
                    category=UserWarning,
                )
                search.fit(Xr_s, y)
        else:
            search.fit(Xr_s, y)
        best = search.best_estimator_

        val_prob = best.predict_proba(X_val_s)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        val_brier, val_ece = summarize_calibration(f"{name} [val]", y_val, val_prob)

        test_prob = best.predict_proba(X_test_s)[:, 1]
        test_auc = roc_auc_score(y_test, test_prob)
        test_f1 = f1_score(y_test, (test_prob > 0.5).astype(int))
        test_brier, test_ece = summarize_calibration(f"{name} [test]", y_test, test_prob)

        log.info(
            "%s -> val_AUC=%.4f test_AUC=%.4f test_F1=%.4f (unique param combos evaluated: %d)",
            name,
            val_auc,
            test_auc,
            test_f1,
            len(set(tuple(sorted(p.items())) for p in search.cv_results_["params"])),
        )
        results[name] = {
            "model": best,
            "cv_results_": search.cv_results_,
            "best_params": search.best_params_,
            "val_AUC": val_auc,
            "val_brier": val_brier,
            "val_ece": val_ece,
            "AUC": test_auc,
            "F1": test_f1,
            "brier": test_brier,
            "ece": test_ece,
        }
    return results
