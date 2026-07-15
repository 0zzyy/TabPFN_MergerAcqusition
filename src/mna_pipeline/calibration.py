"""Calibration metrics: Expected Calibration Error and Brier score."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.metrics import brier_score_loss

log = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error over n_bins equal-width bins.

    Input: y_true (0/1 labels), y_prob (predicted probabilities of the
    positive class, any values are clipped to [0, 1] and NaNs are
    treated as 0.5), n_bins (number of equal-width probability bins).
    Output: a single float ECE value in [0, 1].
    Assumes y_true and y_prob are 1D and the same length; assumes no
    upstream state.
    """
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


def summarize_calibration(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Compute and log Brier score and ECE for one model's predictions.

    Input: name (label for logging), y_true, y_prob as in compute_ece.
    Output: (brier, ece) tuple of floats.
    Assumes no upstream state.
    """
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    log.info("%s -> Brier=%.4f, ECE=%.4f", name, brier, ece)
    return brier, ece
