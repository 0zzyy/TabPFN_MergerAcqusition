"""Train/validation/test splitting strategies (A4, A6).

Split ratio and rationale (A4): the pipeline uses a 70/15/15
train/validation/test split. Train (70%) feeds SMOTE + feature selection
and hyperparameter search; validation (15%) is used for genuine model
selection, meaning every tuned candidate model's AUC and calibration
(Brier, ECE) are computed and logged on the validation set before the
test set is touched for final reporting, so validation is not "dead"
weight; test (15%) is reserved exclusively for the final, one-shot
reported metrics per model. 70/15/15 was chosen because it is the ratio
already used by the original scripts (both the GitHub `src/main.py` and
the thesis's Annex D script), so switching to it here does not change
the amount of data available to any downstream step, isolating the
effect of the other fixes in this pipeline from a confound of a changed
split ratio.

Split strategies (A6): `random` stratifies purely on the target label
and ignores deal chronology, which risks look-ahead bias, later market
regimes (financing conditions, deal-review practices, macro cycles)
leaking into a test set that is supposed to represent "unseen future
deals." `temporal` sorts by `announcement_date` and assigns the earliest
~70% of deals to train, the next ~15% to validation, and the most recent
~15% to test, which is the correct protocol for a model meant to predict
outcomes of new deals going forward. `temporal` is the default split
strategy; `random` is retained and both are reported side by side (per
A6) because a lower AUC under `temporal` is an expected and important
finding about the difficulty of forecasting future, unseen market
regimes, not a bug to hide.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


@dataclass
class SplitResult:
    """Container for one train/val/test split.

    Fields are the row-aligned DataFrame/Series pairs for each partition.
    Assumes the caller will index features and target consistently.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    strategy: str


def _class_balance(y: pd.Series) -> dict:
    """Return a dict of class -> proportion for the given labels.

    Input: y, a pandas Series of 0/1 labels. Output: dict mapping the
    label value to its fraction of len(y). Assumes y is non-empty;
    raises ZeroDivisionError otherwise by design, since an empty split
    is a structural bug the caller should see immediately.
    """
    counts = y.value_counts(normalize=True).sort_index()
    return counts.to_dict()


def split_random(
    X: pd.DataFrame, y: pd.Series, seed: int = 42
) -> SplitResult:
    """Stratified random 70/15/15 train/val/test split.

    Input: X (feature/raw columns, pre-preprocessing), y (target labels),
    seed for reproducibility. Output: a SplitResult with strategy="random".
    Raises whatever sklearn's train_test_split raises for degenerate
    inputs (for example a single-class y cannot be stratified).
    Assumes X and y share the same index and are already row-aligned;
    assumes no upstream state.
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(VAL_FRAC + TEST_FRAC), stratify=y, random_state=seed
    )
    val_share_of_tmp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=(1 - val_share_of_tmp),
        stratify=y_tmp,
        random_state=seed,
    )
    return SplitResult(X_train, y_train, X_val, y_val, X_test, y_test, "random")


def split_temporal(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series
) -> SplitResult:
    """Chronological 70/15/15 train/val/test split by announcement_date.

    Input: X (feature/raw columns), y (target labels), dates (a
    pandas Series of datetimes, same index as X/y, typically
    `df["announcement_date"]`). Output: a SplitResult with
    strategy="temporal", where every date in X_train <= every date in
    X_val <= every date in X_test (ties at a boundary are broken by a
    stable sort, so rows sharing the boundary date can land in either
    adjacent partition, but partitions are otherwise strictly ordered
    by row count).
    Raises ValueError if `dates` contains nulls, since an unordered
    or unknown announcement date makes temporal ordering meaningless.
    Assumes X, y, and dates share the same index; assumes no upstream
    state.
    """
    if dates.isna().any():
        raise ValueError(
            "Cannot perform a temporal split: announcement_date contains "
            "null values for some rows."
        )
    order = dates.sort_values(kind="stable").index
    n = len(order)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return SplitResult(
        X.loc[train_idx],
        y.loc[train_idx],
        X.loc[val_idx],
        y.loc[val_idx],
        X.loc[test_idx],
        y.loc[test_idx],
        "temporal",
    )


def describe_split(result: SplitResult, dates: pd.Series | None = None) -> None:
    """Log split sizes, class balance, and (for temporal splits) date ranges.

    Input: result, the SplitResult to describe; dates, the same Series
    passed to split_temporal (or None for a random split, where date
    ranges are still logged if provided, for side-by-side comparison).
    Output: None; writes INFO log lines.
    Assumes result's partitions share their original index with `dates`
    when `dates` is provided.
    """
    log.info(
        "[%s split] Train=%d Val=%d Test=%d",
        result.strategy,
        len(result.X_train),
        len(result.X_val),
        len(result.X_test),
    )
    log.info(
        "[%s split] class balance train=%s val=%s test=%s",
        result.strategy,
        _class_balance(result.y_train),
        _class_balance(result.y_val),
        _class_balance(result.y_test),
    )
    if dates is not None:
        for name, idx in (
            ("train", result.X_train.index),
            ("val", result.X_val.index),
            ("test", result.X_test.index),
        ):
            part = dates.loc[idx]
            if len(part):
                log.info(
                    "[%s split] %s date range: %s to %s",
                    result.strategy,
                    name,
                    part.min(),
                    part.max(),
                )
