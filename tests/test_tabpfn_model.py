"""Part B guardrail tests + A9 core-step-must-raise tests for TabPFN-3 training."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mna_pipeline.tabpfn_model import (
    TABPFN3_CPU_HARD_LIMIT_SAMPLES,
    TABPFN3_CPU_WARN_LIMIT_SAMPLES,
    CPUDatasetTooLargeError,
    guard_cpu_dataset_size,
    train_tabpfn,
)
from fixtures import FakeTabPFNClassifier, fake_tabpfn_factory, failing_tabpfn_factory


def test_guard_cpu_dataset_size_under_limit_ok(caplog):
    guard_cpu_dataset_size(TABPFN3_CPU_HARD_LIMIT_SAMPLES - 1, device="cpu")


def test_guard_cpu_dataset_size_over_hard_limit_raises():
    with pytest.raises(CPUDatasetTooLargeError):
        guard_cpu_dataset_size(TABPFN3_CPU_HARD_LIMIT_SAMPLES + 1, device="cpu")


def test_guard_cpu_dataset_size_between_warn_and_hard_limit_warns(caplog):
    caplog.set_level(logging.WARNING)
    n = (TABPFN3_CPU_WARN_LIMIT_SAMPLES + TABPFN3_CPU_HARD_LIMIT_SAMPLES) // 2
    guard_cpu_dataset_size(n, device="cpu")
    assert any("may be slow" in rec.message for rec in caplog.records)


def test_guard_cpu_dataset_size_override_bypasses_hard_limit():
    guard_cpu_dataset_size(TABPFN3_CPU_HARD_LIMIT_SAMPLES + 1, device="cpu", allow_override=True)


def test_guard_cpu_dataset_size_gpu_device_ignores_limit():
    guard_cpu_dataset_size(TABPFN3_CPU_HARD_LIMIT_SAMPLES + 5000, device="cuda")


def _small_train_test():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    y_train = pd.Series(rng.integers(0, 2, 60))
    X_test = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
    y_test = pd.Series(rng.integers(0, 2, 20))
    return X_train, y_train, X_test, y_test


def test_train_tabpfn_happy_path_with_fake_classifier(tmp_path):
    X_train, y_train, X_test, y_test = _small_train_test()
    result = train_tabpfn(
        X_train, y_train, X_test, y_test,
        seed=42, cache=tmp_path, device="cpu",
        classifier_factory=fake_tabpfn_factory,
    )
    assert result["model"] is not None
    assert np.isfinite(result["AUC"])
    assert np.isfinite(result["F1"])
    assert result["checkpoint"]
    assert (tmp_path / "tabpfn3_rq.joblib").exists()


def test_train_tabpfn_uses_cache_on_second_call(tmp_path):
    X_train, y_train, X_test, y_test = _small_train_test()
    calls = {"n": 0}

    def counting_factory(**kwargs):
        calls["n"] += 1
        return FakeTabPFNClassifier(seed=kwargs.get("seed", 42))

    train_tabpfn(X_train, y_train, X_test, y_test, seed=42, cache=tmp_path, classifier_factory=counting_factory)
    train_tabpfn(X_train, y_train, X_test, y_test, seed=42, cache=tmp_path, classifier_factory=counting_factory)
    assert calls["n"] == 1


def test_train_tabpfn_cpu_oversized_raises_before_touching_factory(tmp_path):
    rng = np.random.default_rng(1)
    n = TABPFN3_CPU_HARD_LIMIT_SAMPLES + 10
    X_train = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y_train = pd.Series(rng.integers(0, 2, n))
    X_test = pd.DataFrame(rng.normal(size=(5, 3)), columns=list("abc"))
    y_test = pd.Series(rng.integers(0, 2, 5))

    called = {"n": 0}

    def factory(**kwargs):
        called["n"] += 1
        return FakeTabPFNClassifier()

    with pytest.raises(CPUDatasetTooLargeError):
        train_tabpfn(X_train, y_train, X_test, y_test, seed=42, cache=tmp_path, device="cpu", classifier_factory=factory)
    assert called["n"] == 0


def test_train_tabpfn_core_failure_propagates_not_swallowed(tmp_path):
    """A9: TabPFN training is a core step; a fit failure must raise, not become NaN."""
    X_train, y_train, X_test, y_test = _small_train_test()
    with pytest.raises(RuntimeError):
        train_tabpfn(
            X_train, y_train, X_test, y_test,
            seed=42, cache=tmp_path, device="cpu",
            classifier_factory=failing_tabpfn_factory,
        )
