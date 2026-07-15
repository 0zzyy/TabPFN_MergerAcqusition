"""Part D: full end-to-end integration test on a small synthetic dataset.

Uses a fixed seed and does not require the real CSV. TabPFN-3's real
weights are gated (see LICENSE_NOTES.md / README Known Limitations), so
`classifier_factory=fake_tabpfn_factory` is injected to exercise every
other pipeline stage for real: leakage-safe preprocessing, temporal and
random splitting, SMOTE + feature selection, baseline hyperparameter
search, SHAP explanation, DCA, and run metadata.
"""
from __future__ import annotations

import numpy as np
import pytest

from mna_pipeline.dca import CANONICAL_MODEL_NAMES
from mna_pipeline.pipeline import PipelineArgs, run_pipeline
from fixtures import fake_tabpfn_factory, write_synthetic_csv


def test_pipeline_end_to_end_synthetic_single_split(tmp_path):
    csv_path = write_synthetic_csv(tmp_path / "synthetic.csv", n=600, seed=11, positive_rate=0.2)
    args = PipelineArgs(
        input_csv=csv_path,
        output_dir=tmp_path / "out",
        seed=42,
        smote_sample_size=1000,
        top_k=8,
        split_strategy="temporal",
        compare_splits=False,
        search_mode="fast",
        strict=False,
        shap_background_size=40,
        shap_explain_size=40,
    )
    result = run_pipeline(args, classifier_factory=fake_tabpfn_factory)

    assert set(result.runs_by_strategy) == {"temporal"}
    primary = result.runs_by_strategy["temporal"]
    assert set(primary.all_models) == {"LR", "RF", "GB", "SVM", "MLP", "TabPFN-3"}
    for name, res in primary.all_models.items():
        assert np.isfinite(res["AUC"]), f"{name} AUC is not finite"
        assert np.isfinite(res["F1"]), f"{name} F1 is not finite"

    assert result.dca_modelnames == list(CANONICAL_MODEL_NAMES)
    assert (args.output_dir / "run_metadata.json").exists()
    assert result.metadata.tabpfn_checkpoint is not None


def test_pipeline_end_to_end_compares_both_split_strategies(tmp_path):
    csv_path = write_synthetic_csv(tmp_path / "synthetic2.csv", n=600, seed=12, positive_rate=0.2)
    args = PipelineArgs(
        input_csv=csv_path,
        output_dir=tmp_path / "out2",
        seed=42,
        smote_sample_size=1000,
        top_k=8,
        split_strategy="temporal",
        compare_splits=True,
        search_mode="fast",
        strict=False,
        shap_background_size=30,
        shap_explain_size=30,
    )
    result = run_pipeline(args, classifier_factory=fake_tabpfn_factory)
    assert set(result.runs_by_strategy) == {"temporal", "random"}
    for strategy, run in result.runs_by_strategy.items():
        for name, res in run.all_models.items():
            assert np.isfinite(res["AUC"]), f"[{strategy}] {name} AUC is not finite"
