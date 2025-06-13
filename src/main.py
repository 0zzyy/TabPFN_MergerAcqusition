#!/usr/bin/env python3
"""Lightweight pipeline for merger & acquisition outcome prediction.

This script trains tuned baseline models and TabPFN on tabular data,
then explains predictions with SHAP and optionally performs Decision
Curve Analysis.  It mirrors the earlier monolithic script but is
refactored for clarity and easier maintenance.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier
from dcurves import dca
import shap

log = logging.getLogger(__name__)


@dataclass
class Args:
    input_csv: pathlib.Path
    output_cache: pathlib.Path
    seed: int = 42
    smote_sample_size: int = 2000
    top_k: int = 10
    tabpfn_time: int = 10


def parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    p = argparse.ArgumentParser(description="Train baselines, TabPFN, and compute SHAP explanations")
    p.add_argument("--input-csv", type=pathlib.Path, required=True)
    p.add_argument("--output-cache", type=pathlib.Path, default=pathlib.Path("cache_light_rq"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smote-sample-size", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tabpfn-time", type=int, default=10)
    ns = p.parse_args(argv)
    return Args(**vars(ns))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
    y_true = np.asarray(y_true)
    y_prob = np.nan_to_num(np.clip(y_prob, 0.0, 1.0), nan=0.5)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        prop = mask.mean()
        if prop:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += abs(acc - conf) * prop
    return ece


def summarize_calibration(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    log.info("%s → Brier=%.4f, ECE=%.4f", name, brier, ece)


def extract_shap_values(result) -> np.ndarray:
    """Return a 1D array of SHAP values for the positive class."""
    arr = getattr(result, "values", result)
    if isinstance(arr, list):
        arr = arr[1]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 1]
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_data(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, index_col=0, low_memory=False)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "status" not in df.columns:
        raise ValueError("Column 'status' missing from data")
    y = df["status"].astype(int)
    X = df.drop(columns=[c for c in [
        "status",
        "target_id",
        "acquirer_id",
        "announcement_date",
        "exit_date",
        "target_naics_code",
    ] if c in df.columns])
    if "attitude" in X.columns:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        atts = enc.fit_transform(X[["attitude"]])
        cols = enc.get_feature_names_out(["attitude"])
        X = X.drop(columns=["attitude"])
        X = pd.concat([X, pd.DataFrame(atts, columns=cols, index=X.index)], axis=1)
    for col in X.select_dtypes(include="number"):
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
    X.fillna(0, inplace=True)
    return X, y


def smote_and_select(X: pd.DataFrame, y: pd.Series, seed: int, sample_size: int, k: int) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    subset = min(len(X), sample_size)
    if subset < len(X):
        X_sub, _, y_sub, _ = train_test_split(X, y, train_size=subset, stratify=y, random_state=seed)
    else:
        X_sub, y_sub = X, y
    sm = SMOTE(random_state=seed)
    Xr, yr = sm.fit_resample(X_sub, y_sub)
    rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(Xr, yr)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top = importances.nlargest(k).index.tolist()
    Xr_top = pd.DataFrame(Xr, columns=X.columns)[top]
    return Xr_top, yr, top


def tune_baselines(Xr_s: np.ndarray, y: np.ndarray, X_test_s: np.ndarray, y_test: np.ndarray, seed: int) -> Dict[str, Dict[str, object]]:
    defs = {
        "LR": LogisticRegression(solver="liblinear", max_iter=500, random_state=seed),
        "RF": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "GB": GradientBoostingClassifier(random_state=seed),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "MLP": MLPClassifier(max_iter=200, tol=1e-4, early_stopping=True, random_state=seed, hidden_layer_sizes=(50,)),
    }
    grids = {
        "LR": {"C": [0.1, 1], "penalty": ["l2"]},
        "RF": {"n_estimators": [50], "max_depth": [None, 5]},
        "GB": {"n_estimators": [50], "max_depth": [3], "learning_rate": [0.1]},
        "SVM": {"C": [0.1, 1], "gamma": ["scale"]},
        "MLP": {"alpha": [1e-3], "learning_rate_init": [1e-3]},
    }
    results: Dict[str, Dict[str, object]] = {}
    for name, mdl in defs.items():
        log.info("Tuning %s", name)
        try:
            search = RandomizedSearchCV(
                mdl,
                grids[name],
                n_iter=2,
                cv=StratifiedKFold(2, shuffle=True, random_state=seed),
                scoring="roc_auc",
                random_state=seed,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(Xr_s, y)
            best = search.best_estimator_
            y_prob = best.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, (y_prob > 0.5).astype(int))
            summarize_calibration(name, y_test, y_prob)
            log.info("%s → AUC=%.4f, F1=%.4f", name, auc, f1)
            results[name] = {"model": best, "AUC": auc, "F1": f1}
        except Exception as e:  # pragma: no cover
            log.error("%s tuning failed: %s", name, e)
            log.debug("%s", traceback.format_exc())
            results[name] = {"model": None, "AUC": np.nan, "F1": np.nan}
    return results


def train_tabpfn(Xr: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, seed: int, cache: pathlib.Path, max_time: int) -> Dict[str, object]:
    cache.mkdir(exist_ok=True)
    pkl = cache / "tabpfn_rq.joblib"
    model = None
    if pkl.exists():
        try:
            model = joblib.load(pkl)
            log.info("Loaded cached TabPFN model")
        except Exception:
            model = None
    if model is None:
        try:
            model = TabPFNClassifier(device="cpu", max_time=max_time)
            model.fit(Xr.values.astype(np.float32), y.values.astype(np.int64))
            joblib.dump(model, pkl)
            log.info("Trained and cached TabPFN")
        except Exception as e:
            log.error("TabPFN training failed: %s", e)
            log.debug("%s", traceback.format_exc())
            return {"model": None, "AUC": np.nan, "F1": np.nan}
    y_prob = model.predict_proba(X_test.values.astype(np.float32))[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, (y_prob > 0.5).astype(int))
    summarize_calibration("TabPFN", y_test, y_prob)
    log.info("TabPFN → AUC=%.4f, F1=%.4f", auc, f1)
    return {"model": model, "AUC": auc, "F1": f1}


def explain_models(models: Dict[str, Dict[str, object]], X_ref: pd.DataFrame, X_explain: pd.DataFrame) -> Dict[str, pd.Series]:
    imps: Dict[str, pd.Series] = {}
    for name, res in models.items():
        mdl = res.get("model")
        if mdl is None:
            continue
        try:
            log.info("Explaining %s", name)
            pred_fn = mdl if name in {"RF", "GB"} else mdl.predict_proba
            expl = shap.Explainer(pred_fn, X_ref)
            out = expl(X_explain)
            arr = extract_shap_values(out)
            if arr.ndim == 2:
                arr = arr[0]
            imp = pd.Series(np.abs(arr), index=X_explain.columns).sort_values(ascending=False)
            imps[name] = imp
        except Exception as e:
            log.warning("SHAP for %s failed: %s", name, e)
            log.debug("%s", traceback.format_exc())
    return imps


def run_dca(df: pd.DataFrame, outcome: str, model_cols: List[str]) -> None:
    try:
        dca(df, outcome=outcome, modelnames=model_cols)
        log.info("DCA computed for models: %s", model_cols)
    except Exception as e:  # pragma: no cover
        log.warning("DCA failed: %s", e)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.captureWarnings(True)
    np.random.seed(args.seed)

    t0 = time.time()
    log.info("=== PIPELINE START ===")

    df = load_data(args.input_csv)
    X, y = preprocess(df)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=args.seed)
    log.info("Train=%d, Val=%d, Test=%d", len(X_train), len(X_val), len(X_test))

    Xr_p, yr, features = smote_and_select(X_train, y_train, args.seed, args.smote_sample_size, args.top_k)
    X_train_p = X_train[features]
    X_test_p = X_test[features]

    scaler = StandardScaler().fit(Xr_p)
    Xr_s = scaler.transform(Xr_p)
    X_test_s = scaler.transform(X_test_p)

    baseline_results = tune_baselines(Xr_s, yr.values, X_test_s, y_test.values, args.seed)
    tabpfn_result = train_tabpfn(Xr_p, yr, X_test_p, y_test, args.seed, args.output_cache, args.tabpfn_time)
    all_models = {**baseline_results, "TabPFN": tabpfn_result}

    shap_bg = pd.DataFrame(Xr_p).sample(1, random_state=args.seed)
    shap_ts = X_test_p.iloc[:1].reset_index(drop=True)
    shap_imps = explain_models(all_models, shap_bg, shap_ts)
    for name, imp in shap_imps.items():
        log.info("Top SHAP feature for %s: %s (%.4f)", name, imp.index[0], imp.iloc[0])

    dca_df = shap_ts.copy()
    dca_df["status"] = y_test.values[:1]
    for name, res in all_models.items():
        mdl = res.get("model")
        if mdl is None:
            continue
        try:
            prob = mdl.predict_proba(X_test_s if name != "TabPFN" else X_test_p)[:, 1]
            dca_df[name] = prob
        except Exception:
            pass
    run_dca(dca_df, "status", [c for c in dca_df.columns if c != "status"])

    log.info("=== FINAL METRICS ===")
    for name, res in all_models.items():
        log.info("%6s → AUC=%.4f, F1=%.4f", name, res["AUC"], res["F1"])

    log.info("Total time: %.1fs", time.time() - t0)
    log.info("=== COMPLETE ===")


if __name__ == "__main__":  # pragma: no cover
    main()
