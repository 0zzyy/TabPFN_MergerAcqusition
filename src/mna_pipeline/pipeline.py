"""Pipeline orchestration.

A6 fix, reporting side by side: this module always computes the full
benchmark (SMOTE, feature selection, baseline tuning, TabPFN-3) under
BOTH the `random` and `temporal` split strategies whenever
`compare_splits=True` (the default), and logs a side-by-side comparison
table, because a lower AUC under `temporal` is an expected and important
finding, not something to hide by only ever running one strategy. This
roughly doubles pipeline runtime (hyperparameter search and TabPFN
training each run twice); `compare_splits=False` (wired to
`--single-split` on the CLI) restricts the run to only
`args.split_strategy`, for fast iteration or CI. This trade-off is a
deliberate, flagged design choice, not a silent shortcut: the brief
requires both results reported side by side, and the only way to do
that honestly is to actually run both.

`args.split_strategy` (default "temporal") selects which strategy is
"primary": its fitted models drive the SHAP explanation and Decision
Curve Analysis steps and are the ones written to run_metadata.json.

Additional correctness fix found while implementing A2/A3 (flagged,
outside the audited A1-A11/Part B list): the baseline models (LR, RF,
GB, SVM, MLP) are fit and evaluated on StandardScaler-transformed
features throughout this module, but TabPFN-3 is fit and evaluated on
the raw (unscaled) feature-selected data, matching TabPFN's own
guidance not to pre-scale inputs. SHAP and DCA must therefore feed each
model the same representation it was trained on: scaled features (as
labeled DataFrames, so column names survive StandardScaler) for the
five baseline models, and raw features for TabPFN-3. Feeding a baseline
model raw (unscaled) inputs at explanation time, as the original
scripts' SHAP section did, would silently apply thresholds and
coefficients learned in standardized space to unstandardized values and
produce meaningless SHAP/DCA numbers for those five models.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .baselines import tune_baselines
from .data import DATE_COLUMN, load_data
from .dca import build_dca_frame, run_dca
from .metadata import RunMetadata, collect_environment
from .preprocessing import (
    fit_preprocessor,
    persist_preprocessor,
    select_feature_columns,
    transform,
)
from .resampling import smote_and_select
from .shap_explain import (
    DEFAULT_BACKGROUND_SIZE,
    DEFAULT_EXPLAIN_SIZE_CAP,
    build_background,
    explain_models,
    select_explain_set,
)
from .splits import SplitResult, describe_split, split_random, split_temporal
from .tabpfn_model import (
    TABPFN3_CHECKPOINT,
    ClassifierFactory,
    build_tabpfn_classifier,
    train_tabpfn,
)

log = logging.getLogger(__name__)

TABPFN_MODEL_KEY = "TabPFN-3"


@dataclass
class PipelineArgs:
    """CLI-facing configuration for one pipeline run. See cli.py for parsing."""

    input_csv: pathlib.Path
    output_dir: pathlib.Path = pathlib.Path("cache_light_rq")
    seed: int = 42
    smote_sample_size: int = 2000
    top_k: int = 10
    split_strategy: str = "temporal"
    compare_splits: bool = True
    search_mode: str = "full"
    n_iter: int = 20
    cv_folds: int = 5
    strict: bool = False
    tabpfn_device: str = "cpu"
    tabpfn_checkpoint: str = TABPFN3_CHECKPOINT
    allow_cpu_large_dataset: bool = False
    shap_background_size: int = DEFAULT_BACKGROUND_SIZE
    shap_explain_size: int = DEFAULT_EXPLAIN_SIZE_CAP


@dataclass
class StrategyRun:
    """Everything produced by running the benchmark under one split strategy."""

    strategy: str
    split: SplitResult
    features: List[str]
    Xr_p: pd.DataFrame
    yr: pd.Series
    X_test_p: pd.DataFrame
    Xr_s: pd.DataFrame
    X_test_s: pd.DataFrame
    all_models: Dict[str, Dict[str, object]]


@dataclass
class PipelineResult:
    """Top-level result of a full pipeline run."""

    runs_by_strategy: Dict[str, StrategyRun]
    shap_importances: Dict[str, Dict[str, object]]
    dca_modelnames: List[str]
    dca_result: object
    metadata: RunMetadata


def _run_single_strategy(
    strategy: str,
    X_raw: pd.DataFrame,
    y: pd.Series,
    dates: Optional[pd.Series],
    args: PipelineArgs,
    classifier_factory: ClassifierFactory,
) -> StrategyRun:
    """Run the full benchmark (split -> preprocess -> SMOTE -> tune -> TabPFN-3) once.

    Input: strategy ("random" or "temporal"), X_raw/y as returned by
    select_feature_columns, dates (announcement_date Series, required
    for strategy="temporal"), args (PipelineArgs), classifier_factory
    (real or test-double TabPFN-3 constructor).
    Output: a StrategyRun with the fitted split, selected features, and
    every model's fitted estimator plus validation/test metrics.
    Raises ValueError if strategy="temporal" and dates is None; raises
    whatever any core step (split, preprocessing fit, SMOTE, baseline
    tuning, TabPFN-3 training) raises, per A9.
    Assumes X_raw/y/dates share the same index.
    """
    if strategy == "temporal":
        if dates is None:
            raise ValueError("temporal split requested but announcement_date is unavailable")
        split = split_temporal(X_raw, y, dates)
    elif strategy == "random":
        split = split_random(X_raw, y, args.seed)
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    describe_split(split, dates)

    preproc = fit_preprocessor(split.X_train)
    persist_preprocessor(
        preproc, args.output_dir / f"preprocessor_{strategy}.joblib"
    )
    X_train_t = transform(preproc, split.X_train)
    X_val_t = transform(preproc, split.X_val)
    X_test_t = transform(preproc, split.X_test)

    Xr_p, yr, features = smote_and_select(
        X_train_t, split.y_train, args.seed, args.smote_sample_size, args.top_k
    )
    X_val_p = X_val_t[features]
    X_test_p = X_test_t[features]

    scaler = StandardScaler().fit(Xr_p)
    Xr_s = pd.DataFrame(scaler.transform(Xr_p), columns=features)
    X_val_s = pd.DataFrame(scaler.transform(X_val_p), columns=features, index=X_val_p.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test_p), columns=features, index=X_test_p.index)

    baseline_results = tune_baselines(
        Xr_s,
        yr.values,
        X_val_s,
        split.y_val.values,
        X_test_s,
        split.y_test.values,
        args.seed,
        search_mode=args.search_mode,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
    )
    tabpfn_result = train_tabpfn(
        Xr_p,
        yr,
        X_test_p,
        split.y_test,
        args.seed,
        cache=args.output_dir / f"tabpfn_{strategy}",
        device=args.tabpfn_device,
        checkpoint=args.tabpfn_checkpoint,
        allow_cpu_large_dataset=args.allow_cpu_large_dataset,
        classifier_factory=classifier_factory,
    )

    all_models = {**baseline_results, TABPFN_MODEL_KEY: tabpfn_result}
    return StrategyRun(strategy, split, features, Xr_p, yr, X_test_p, Xr_s, X_test_s, all_models)


def _log_comparison_table(runs_by_strategy: Dict[str, StrategyRun]) -> None:
    """Log a side-by-side AUC/F1/calibration table across split strategies (A6)."""
    log.info("=== SPLIT-STRATEGY COMPARISON (A6) ===")
    for strategy, run in runs_by_strategy.items():
        for name, res in run.all_models.items():
            log.info(
                "[%s] %-10s -> AUC=%.4f F1=%.4f Brier=%.4f ECE=%.4f",
                strategy,
                name,
                res.get("AUC", float("nan")),
                res.get("F1", float("nan")),
                res.get("brier", float("nan")),
                res.get("ece", float("nan")),
            )


def run_pipeline(
    args: PipelineArgs,
    classifier_factory: ClassifierFactory = build_tabpfn_classifier,
) -> PipelineResult:
    """Run the end-to-end M&A outcome prediction benchmark.

    Input: args (PipelineArgs); classifier_factory, defaults to the real
    TabPFN-3 constructor but can be swapped for a test double so this
    function is exercisable without gated model weights.
    Output: a PipelineResult with per-strategy benchmark results, SHAP
    importances and DCA output for the primary strategy
    (args.split_strategy), and run metadata (A7).
    Raises on any core-step failure (data loading, split, preprocessing
    fit, SMOTE, baseline tuning, TabPFN-3 training, final metric
    computation); SHAP and DCA failures warn by default and raise only
    when args.strict=True (A9).
    Assumes args.input_csv points to a readable CSV with a `status`
    column; assumes announcement_date is present if any requested
    strategy is "temporal".
    """
    metadata = collect_environment(args.seed, args.split_strategy)

    df = load_data(args.input_csv)
    dates = df[DATE_COLUMN] if DATE_COLUMN in df.columns else None
    X_raw, y = select_feature_columns(df)

    strategies = [args.split_strategy]
    if args.compare_splits:
        for other in ("temporal", "random"):
            if other not in strategies and (other != "temporal" or dates is not None):
                strategies.append(other)

    runs_by_strategy: Dict[str, StrategyRun] = {}
    for strategy in strategies:
        log.info("=== Running benchmark under split_strategy=%s ===", strategy)
        runs_by_strategy[strategy] = _run_single_strategy(
            strategy, X_raw, y, dates, args, classifier_factory
        )

    if args.compare_splits:
        _log_comparison_table(runs_by_strategy)

    primary = runs_by_strategy[args.split_strategy]

    # Each model must be explained/predicted on the same representation it
    # was fit on: scaled features for the 5 baseline models, raw features
    # for TabPFN-3 (see module docstring "Additional correctness fix").
    baseline_models = {k: v for k, v in primary.all_models.items() if k != TABPFN_MODEL_KEY}
    tabpfn_models = {k: v for k, v in primary.all_models.items() if k == TABPFN_MODEL_KEY}

    shap_importances: Dict[str, Dict[str, object]] = {}
    if baseline_models:
        bg_s = build_background(primary.Xr_s, args.shap_background_size, args.seed)
        explain_s = select_explain_set(primary.X_test_s, args.shap_explain_size, args.seed)
        shap_importances.update(
            explain_models(baseline_models, bg_s, explain_s, strict=args.strict)
        )
    if tabpfn_models:
        bg_raw = build_background(primary.Xr_p, args.shap_background_size, args.seed)
        explain_raw = select_explain_set(primary.X_test_p, args.shap_explain_size, args.seed)
        shap_importances.update(
            explain_models(tabpfn_models, bg_raw, explain_raw, strict=args.strict)
        )
    for name, imp in shap_importances.items():
        top = imp["mean_abs_importance"]
        if len(top):
            log.info("Top mean|SHAP| feature for %s: %s (%.4f)", name, top.index[0], top.iloc[0])

    model_probs: Dict[str, np.ndarray] = {}
    for name, res in primary.all_models.items():
        mdl = res.get("model")
        if mdl is None:
            continue
        X_for_predict = (
            primary.X_test_p.values.astype(np.float32) if name == TABPFN_MODEL_KEY else primary.X_test_s
        )
        model_probs[name] = mdl.predict_proba(X_for_predict)[:, 1]

    df_dca, dca_modelnames = build_dca_frame(primary.split.y_test, model_probs)
    dca_result = run_dca(df_dca, "status", dca_modelnames, strict=args.strict)

    tabpfn_res = primary.all_models.get(TABPFN_MODEL_KEY, {})
    metadata.tabpfn_checkpoint = tabpfn_res.get("checkpoint")
    metadata.tabpfn_package_version = tabpfn_res.get("package_version")
    metadata.tabpfn_model_version_tag = tabpfn_res.get("model_version_tag")
    metadata.tabpfn_device = args.tabpfn_device
    metadata.write(args.output_dir / "run_metadata.json")
    log.info(
        "Model identity (A7): tabpfn package=%s checkpoint=%s device=%s",
        metadata.tabpfn_package_version,
        metadata.tabpfn_checkpoint,
        metadata.tabpfn_device,
    )

    log.info("=== FINAL METRICS (primary strategy=%s) ===", args.split_strategy)
    for name, res in primary.all_models.items():
        log.info("%10s -> AUC=%.4f F1=%.4f", name, res.get("AUC", float("nan")), res.get("F1", float("nan")))

    return PipelineResult(runs_by_strategy, shap_importances, dca_modelnames, dca_result, metadata)
