"""TabPFN-3 integration (Part B migration, A7, A9).

Migration verification (Part B item 1): before writing this module, the
installed `tabpfn` package was inspected directly (`pip show tabpfn` ->
8.1.0, installed 2026-07-15) rather than assuming the TabPFN v2.x API
in the original scripts still applied. Concretely:

- `TabPFNClassifier.__init__` in tabpfn==8.1.0 has no `max_time`
  parameter at all (confirmed via `inspect.signature`); the old
  `device="cpu", max_time=max_time` call from both the GitHub
  `src/main.py` and the thesis's Annex D script would raise
  `TypeError: unexpected keyword argument 'max_time'` unchanged. The
  ensemble/compute budget is now controlled by `n_estimators` (default
  8) together with `auto_scale_n_estimators` (default True).
- `n_jobs` is deprecated in favor of `n_preprocessing_jobs`.
- `random_state` now defaults to `0` (deterministic), a behavior change
  from v2.x's non-deterministic default.
- `tabpfn.settings.TabPFNSettings.model_version` defaults to
  `ModelVersion.V3`, confirming `pip install tabpfn` today installs
  TabPFN-3 by default, matching Prior Labs' documentation
  (https://docs.priorlabs.ai/models, checked 2026-07-15) and the
  TabPFN-3 technical report (arXiv:2605.13986, May 2026).

Checkpoint pin (Part B item 2): `tabpfn.model_loading.ModelSource.
get_classifier_v3()` (installed package source, v8.1.0) lists the
checkpoints shipped for TabPFN-3 classification under the HuggingFace
repo `Prior-Labs/tabpfn_3`:
  tabpfn-v3-classifier-v3_default.ckpt
  tabpfn-v3-classifier-v3_20260417_binary.ckpt
  tabpfn-v3-classifier-v3_20260417_multiclass.ckpt
  tabpfn-v3-classifier-v3_20260506_ood.ckpt
`status` (the M&A deal outcome target) is binary, so this module pins
the binary-classification-specific checkpoint rather than the generic
default or the multiclass/OOD variants, so results are reproducible
against one named model artifact rather than "whatever v3 default
happens to resolve to."

CPU guardrail (Part B item 3): the old "CPU rows <~1000" number in prior
guidance was TabPFN-2-specific and must not be reused blindly. This
module's guardrail constants are read directly from the *installed*
tabpfn==8.1.0 source, `tabpfn/validation.py::_validate_num_samples_for_cpu`,
checked 2026-07-15: CPU inference raises `RuntimeError` above 1000
samples by default (overridable via `ignore_pretraining_limits=True` or
the `TABPFN_ALLOW_CPU_LARGE_DATASET` environment variable) and emits a
performance warning above 200 samples. The numeric value happens to
match the old v2 guidance, but it is independently re-verified here
against TabPFN-3's own current guardrail code, not carried over on the
assumption that it did not change.

License (Part B item 4): TabPFN-3 model weights are gated and licensed
under Prior Labs' "TABPFN-3 License v1.0" (non-commercial; permits
research, evaluation, and internal benchmarking; prohibits commercial or
production use), distinct from the Apache-2.0-derived Prior Labs code
license that covers the `tabpfn` Python package itself. See
LICENSE_NOTES.md for the full analysis. Downloading the weights requires
one-time interactive license acceptance at https://ux.priorlabs.ai or a
`TABPFN_TOKEN` environment variable for non-interactive environments.

A9 fix: TabPFN training is a core benchmark step (one of the six models
under comparison), so `train_tabpfn` does not catch and mask fit/predict
failures with a NaN placeholder; it raises and halts the run. The one
exception is a failed load of a *cached* prior model, which is narrowly
caught since falling back to retraining is a genuine non-core
convenience, not a silent corruption of a reported result.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .calibration import summarize_calibration
from .metadata import get_tabpfn_package_version

log = logging.getLogger(__name__)

# Source: tabpfn==8.1.0 (checked 2026-07-15), tabpfn/validation.py,
# _validate_num_samples_for_cpu(). Re-verified for TabPFN-3 specifically,
# not assumed to carry over from TabPFN-2 guidance.
TABPFN3_CPU_HARD_LIMIT_SAMPLES = 1000
TABPFN3_CPU_WARN_LIMIT_SAMPLES = 200

# Source: tabpfn==8.1.0 (checked 2026-07-15),
# tabpfn.model_loading.ModelSource.get_classifier_v3(), repo
# "Prior-Labs/tabpfn_3". Binary-classification-specific checkpoint,
# matching this project's binary `status` target.
TABPFN3_CHECKPOINT = "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
TABPFN3_MODEL_VERSION_TAG = "v3"

ClassifierFactory = Callable[..., object]


class CPUDatasetTooLargeError(ValueError):
    """Raised when a CPU run would exceed the TabPFN-3 CPU guardrail."""


def guard_cpu_dataset_size(
    n_samples: int, device: str, allow_override: bool = False
) -> None:
    """Enforce the TabPFN-3 CPU sample-count guardrail before training.

    Input: n_samples (rows to be fit), device (a torch device string,
    guardrail only applies when it starts with "cpu"), allow_override
    (mirrors `ignore_pretraining_limits` / `TABPFN_ALLOW_CPU_LARGE_DATASET`).
    Output: None. Raises CPUDatasetTooLargeError if n_samples exceeds
    TABPFN3_CPU_HARD_LIMIT_SAMPLES on CPU and allow_override is False;
    logs a warning (does not raise) if n_samples exceeds
    TABPFN3_CPU_WARN_LIMIT_SAMPLES.
    Assumes no upstream state; safe to call before any model
    construction so the failure is immediate and clearly attributed to
    this guardrail rather than surfacing later as an installed-library
    RuntimeError.
    """
    if not device.startswith("cpu"):
        return
    if allow_override:
        return
    if n_samples > TABPFN3_CPU_HARD_LIMIT_SAMPLES:
        raise CPUDatasetTooLargeError(
            f"Refusing to run TabPFN-3 on CPU with {n_samples} samples: "
            f"exceeds the documented guardrail of "
            f"{TABPFN3_CPU_HARD_LIMIT_SAMPLES} samples (source: tabpfn==8.1.0, "
            "tabpfn/validation.py::_validate_num_samples_for_cpu, checked "
            "2026-07-15). Use a GPU device, pass allow_cpu_large_dataset=True, "
            "or set TABPFN_ALLOW_CPU_LARGE_DATASET=1."
        )
    if n_samples > TABPFN3_CPU_WARN_LIMIT_SAMPLES:
        log.warning(
            "Running TabPFN-3 on CPU with %d samples (> %d) may be slow.",
            n_samples,
            TABPFN3_CPU_WARN_LIMIT_SAMPLES,
        )


def build_tabpfn_classifier(
    device: str = "cpu",
    seed: int = 42,
    checkpoint: str = TABPFN3_CHECKPOINT,
    n_estimators: int = 8,
    ignore_pretraining_limits: bool = False,
) -> object:
    """Construct a real tabpfn.TabPFNClassifier pinned to the binary checkpoint.

    Input: device ("cpu"/"cuda"/"auto"), seed (random_state), checkpoint
    (a filename resolvable by tabpfn's model loader, default
    TABPFN3_CHECKPOINT), n_estimators (ensemble size), and
    ignore_pretraining_limits (passed straight through to TabPFN-3 to
    permit CPU runs above its own internal guardrail; independent of, but
    consistent with, guard_cpu_dataset_size above).
    Output: an unfitted tabpfn.TabPFNClassifier instance.
    Raises ImportError if tabpfn is not installed.
    Assumes no upstream state. Not called directly by tests, which
    inject a fake classifier_factory instead, since downloading real
    TabPFN-3 weights requires a TABPFN_TOKEN this environment does not
    have (see LICENSE_NOTES.md and README "Known Limitations").
    """
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        model_path=checkpoint,
        device=device,
        random_state=seed,
        n_estimators=n_estimators,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )


def train_tabpfn(
    Xr: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
    cache: pathlib.Path,
    device: str = "cpu",
    checkpoint: str = TABPFN3_CHECKPOINT,
    allow_cpu_large_dataset: bool = False,
    classifier_factory: ClassifierFactory = build_tabpfn_classifier,
) -> Dict[str, object]:
    """Train (or load a cached) TabPFN-3 classifier and evaluate it on the test set.

    Input: Xr/y are SMOTE-resampled, feature-selected training data
    (unscaled; TabPFN-3, like TabPFN-2, does not want manual
    standardization; see the migration notes above); X_test/y_test are
    the correspondingly feature-selected test partition; seed, cache
    (directory for the cached fitted model), device, checkpoint,
    allow_cpu_large_dataset as documented above; classifier_factory
    defaults to `build_tabpfn_classifier` but can be replaced with a
    test double implementing `.fit`/`.predict_proba`.
    Output: dict with "model", "AUC", "F1", "brier", "ece",
    "checkpoint", and "package_version" (the last two feed A7's
    metadata logging).
    Raises CPUDatasetTooLargeError before any model is constructed if
    the CPU guardrail would be violated; raises whatever the underlying
    fit/predict raise otherwise (core step per A9, not caught).
    Assumes Xr/X_test contain only the previously-selected top-k feature
    columns and that Xr contains only training-partition (post-SMOTE)
    rows.
    """
    guard_cpu_dataset_size(len(Xr), device, allow_override=allow_cpu_large_dataset)

    cache.mkdir(parents=True, exist_ok=True)
    pkl = cache / "tabpfn3_rq.joblib"
    model: Optional[object] = None
    if pkl.exists():
        try:
            model = joblib.load(pkl)
            log.info("Loaded cached TabPFN-3 model from %s", pkl)
        except Exception as exc:  # noqa: BLE001 - narrow, documented fallback
            log.warning("Cached TabPFN-3 model at %s could not be loaded (%s); retraining.", pkl, exc)
            model = None

    if model is None:
        model = classifier_factory(device=device, seed=seed, checkpoint=checkpoint)
        model.fit(Xr.values.astype(np.float32), y.values.astype(np.int64))
        joblib.dump(model, pkl)
        log.info("Trained and cached TabPFN-3 (checkpoint=%s)", checkpoint)

    y_prob = model.predict_proba(X_test.values.astype(np.float32))[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, (y_prob > 0.5).astype(int))
    brier, ece = summarize_calibration("TabPFN-3", y_test, y_prob)
    log.info(
        "TabPFN-3 (package=%s, checkpoint=%s) -> AUC=%.4f, F1=%.4f",
        get_tabpfn_package_version(),
        checkpoint,
        auc,
        f1,
    )
    return {
        "model": model,
        "AUC": auc,
        "F1": f1,
        "brier": brier,
        "ece": ece,
        "checkpoint": checkpoint,
        "package_version": get_tabpfn_package_version(),
        "model_version_tag": TABPFN3_MODEL_VERSION_TAG,
    }
