"""Shared synthetic-data fixtures and test doubles for the test suite.

None of these require the real dataset or network access. The synthetic
schema mirrors the columns verifiable from the original scripts and the
Annex D log (status, attitude, realized_return, spread,
historical_avg_completion_time, base_equity_value, day_premium,
completion_time, friendly, plus identifier/date columns that get
dropped), with a handful of extra numeric noise columns so top-k feature
selection is non-trivial.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def make_synthetic_dataset(
    n: int = 800,
    seed: int = 42,
    positive_rate: float = 0.15,
    with_missing: bool = True,
) -> pd.DataFrame:
    """Build a synthetic M&A-shaped dataframe for tests.

    Input: n (row count), seed, positive_rate (fraction of status=1),
    with_missing (whether to inject NaNs in numeric/categorical columns
    to exercise imputation).
    Output: a DataFrame with the columns described in the module
    docstring, indexed 0..n-1 (an explicit index column, matching
    load_data's index_col=0 convention when written to CSV).
    Assumes nothing upstream; this is a fixture generator.
    """
    rng = np.random.default_rng(seed)

    realized_return = rng.normal(0.05, 0.2, n)
    spread = rng.normal(0.1, 0.05, n)
    day_premium = rng.normal(0.3, 0.1, n)
    completion_time = rng.integers(30, 400, n).astype(float)
    historical_avg_completion_time = completion_time + rng.normal(0, 20, n)
    base_equity_value = rng.lognormal(mean=8, sigma=1.2, size=n)
    friendly = rng.integers(0, 2, n).astype(float)

    noise_cols = {f"extra_feature_{i}": rng.normal(0, 1, n) for i in range(5)}

    attitude = rng.choice(["Friendly", "Hostile", "Neutral"], size=n, p=[0.6, 0.15, 0.25])

    logit = (
        2.5 * realized_return
        - 4.0 * spread
        + 1.5 * day_premium
        + 0.8 * friendly
        - 0.001 * completion_time
        + rng.normal(0, 1.0, n)
    )
    threshold = np.quantile(logit, 1 - positive_rate)
    status = (logit >= threshold).astype(int)

    start = pd.Timestamp("2000-01-12")
    end = pd.Timestamp("2024-07-11")
    span_days = (end - start).days
    offsets = np.sort(rng.integers(0, span_days, n))
    announcement_date = [start + pd.Timedelta(days=int(d)) for d in offsets]
    exit_date = [d + pd.Timedelta(days=int(rng.integers(10, 400))) for d in announcement_date]

    df = pd.DataFrame(
        {
            "target_id": np.arange(n),
            "acquirer_id": rng.integers(1000, 9999, n),
            "announcement_date": announcement_date,
            "exit_date": exit_date,
            "target_naics_code": rng.integers(1000, 9999, n),
            "attitude": attitude,
            "realized_return": realized_return,
            "friendly": friendly,
            "completion_time": completion_time,
            "spread": spread,
            "historical_avg_completion_time": historical_avg_completion_time,
            "base_equity_value": base_equity_value,
            "day_premium": day_premium,
            **noise_cols,
            "status": status,
        }
    )

    if with_missing:
        miss_idx = rng.choice(n, size=max(1, n // 20), replace=False)
        df.loc[miss_idx, "realized_return"] = np.nan
        miss_idx2 = rng.choice(n, size=max(1, n // 25), replace=False)
        df.loc[miss_idx2, "attitude"] = np.nan

    df.index.name = "id"
    return df


def write_synthetic_csv(path: pathlib.Path, **kwargs) -> pathlib.Path:
    """Write make_synthetic_dataset(**kwargs) to `path` as CSV with an index column.

    Input: path, destination CSV path; kwargs forwarded to
    make_synthetic_dataset.
    Output: the same path, for chaining.
    Assumes the parent directory exists or is creatable.
    """
    df = make_synthetic_dataset(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


class FakeTabPFNClassifier:
    """A fast, offline, sklearn-compatible stand-in for tabpfn.TabPFNClassifier.

    Used only in tests via dependency injection (pipeline.run_pipeline's
    classifier_factory parameter), because the real TabPFN-3 weights are
    gated behind Prior Labs' license acceptance and require a
    TABPFN_TOKEN this test environment does not have. It implements the
    same fit/predict_proba contract using a plain LogisticRegression
    internally, so pipeline-level logic (caching, evaluation, SHAP, DCA)
    is exercised end to end without ever claiming to reproduce TabPFN-3's
    actual numeric predictions.
    """

    def __init__(self, seed: int = 42, **_ignored):
        self._seed = seed
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(max_iter=1000, random_state=seed)

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(np.asarray(X))
        self._clf.fit(Xs, y)
        return self

    def predict_proba(self, X):
        Xs = self._scaler.transform(np.asarray(X))
        return self._clf.predict_proba(Xs)


def fake_tabpfn_factory(device: str = "cpu", seed: int = 42, checkpoint: str | None = None, **_):
    """classifier_factory-compatible constructor returning a FakeTabPFNClassifier.

    Input/Output: matches tabpfn_model.ClassifierFactory's call signature
    (device, seed, checkpoint keyword args; extra kwargs ignored).
    Assumes it is only ever used in tests.
    """
    return FakeTabPFNClassifier(seed=seed)


class AlwaysFailsClassifier:
    """A classifier whose predict_proba always raises, for A9 fail-loud tests."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        raise RuntimeError("synthetic failure for testing")


def failing_tabpfn_factory(device: str = "cpu", seed: int = 42, checkpoint: str | None = None, **_):
    """classifier_factory returning a model whose fit() raises, for A9 core-step tests."""

    class _Fails:
        def fit(self, X, y):
            raise RuntimeError("synthetic TabPFN-3 fit failure for testing")

        def predict_proba(self, X):  # pragma: no cover - never reached
            raise RuntimeError("synthetic TabPFN-3 predict failure for testing")

    return _Fails()
