"""A9 / Part C item 10: --strict demonstrated both ways at the full pipeline level.

DCA is an optional/explanatory step (like SHAP), so a failure there
should warn and let the run complete when strict=False, and should halt
the run when strict=True. This is exercised through the real
run_pipeline entry point, not just the unit-level dca() function.
"""
from __future__ import annotations

import pytest

from mna_pipeline.pipeline import PipelineArgs, run_pipeline
from fixtures import fake_tabpfn_factory, write_synthetic_csv


def _args(tmp_path, strict: bool) -> PipelineArgs:
    csv_path = write_synthetic_csv(tmp_path / "synthetic.csv", n=400, seed=21, positive_rate=0.2)
    return PipelineArgs(
        input_csv=csv_path,
        output_dir=tmp_path / "out",
        seed=42,
        smote_sample_size=800,
        top_k=6,
        split_strategy="temporal",
        compare_splits=False,
        search_mode="fast",
        strict=strict,
        shap_background_size=25,
        shap_explain_size=25,
    )


def test_strict_false_lets_optional_dca_failure_warn_and_complete(tmp_path, monkeypatch):
    monkeypatch.setattr("dcurves.dca", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated")))
    args = _args(tmp_path, strict=False)
    result = run_pipeline(args, classifier_factory=fake_tabpfn_factory)
    assert result.dca_result is None  # failed but did not halt the run


def test_strict_true_converts_optional_dca_failure_into_hard_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("dcurves.dca", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated")))
    args = _args(tmp_path, strict=True)
    with pytest.raises(ValueError):
        run_pipeline(args, classifier_factory=fake_tabpfn_factory)


def test_strict_true_also_escalates_shap_failures(tmp_path, monkeypatch):
    import mna_pipeline.shap_explain as shap_mod

    def broken_explainer(*args, **kwargs):
        raise RuntimeError("simulated shap failure")

    monkeypatch.setattr(shap_mod.shap, "Explainer", broken_explainer)
    args = _args(tmp_path, strict=True)
    with pytest.raises(ValueError):
        run_pipeline(args, classifier_factory=fake_tabpfn_factory)


def test_strict_false_shap_failure_warns_and_completes(tmp_path, monkeypatch, caplog):
    import logging

    import mna_pipeline.shap_explain as shap_mod

    def broken_explainer(*args, **kwargs):
        raise RuntimeError("simulated shap failure")

    monkeypatch.setattr(shap_mod.shap, "Explainer", broken_explainer)
    caplog.set_level(logging.WARNING)
    args = _args(tmp_path, strict=False)
    result = run_pipeline(args, classifier_factory=fake_tabpfn_factory)
    assert result.shap_importances == {}
