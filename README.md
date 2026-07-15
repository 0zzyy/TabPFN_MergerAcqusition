# TabPFN_MergerAcqusition

Pipeline for predicting the outcome of US stock mergers and acquisitions,
developed as part of a master's thesis at University College Dublin. The
pipeline trains tuned baseline classifiers (Logistic Regression, Random
Forest, Gradient Boosting, SVM, MLP) together with **TabPFN-3**, explains
predictions via SHAP, and runs Decision Curve Analysis (DCA).

This version is a defect-audit remediation and TabPFN-2 -> TabPFN-3
migration of the original single-script pipeline (`src/main.py` on
GitHub, and the "Annex D" script appended to the thesis). See
"Confirmed defects and fixes" and "TabPFN-3 migration" below for what
changed and why, and "Known Limitations" for what is intentionally left
unresolved.

## Dataset

4,400 historical U.S. M&A transactions. The exact `announcement_date`
and `exit_date` ranges reported in different sections of the thesis text
disagreed with each other; this repository does not repeat either prose
claim. The canonical range must be read directly from the CSV's
`announcement_date` column (the pipeline logs the min/max date per split
on every run; see "Temporal split" below). The real CSV is not included
in this repository and was not available when this remediation was
implemented and tested; see "Known Limitations".

## Module structure

The former monolithic `src/main.py` is now `src/mna_pipeline/`, a
modular, independently tested package. `src/main.py` remains as a thin,
backward-compatible CLI wrapper (`python src/main.py --input-csv ...`
still works).

| Module | Responsibility |
|---|---|
| `mna_pipeline/data.py` | Load the raw CSV (`load_data`). |
| `mna_pipeline/splits.py` | Random and temporal train/val/test splitting (A6). |
| `mna_pipeline/preprocessing.py` | Leakage-safe `ColumnTransformer` (impute + one-hot), fit train-only (A1, A10). |
| `mna_pipeline/resampling.py` | SMOTE + random-forest-importance feature selection. |
| `mna_pipeline/baselines.py` | Tuned LR/RF/GB/SVM/MLP via `RandomizedSearchCV` (A4, A5). |
| `mna_pipeline/tabpfn_model.py` | TabPFN-3 integration, CPU guardrail, checkpoint pin (Part B). |
| `mna_pipeline/calibration.py` | Brier score, Expected Calibration Error. |
| `mna_pipeline/shap_explain.py` | SHAP background/explanation sampling and mean-|SHAP| importance (A3). |
| `mna_pipeline/dca.py` | Decision Curve Analysis with an explicit, validated model-column whitelist (A2). |
| `mna_pipeline/metadata.py` | Run metadata: package version, checkpoint identifier, environment (A7). |
| `mna_pipeline/pipeline.py` | Orchestration: runs the full benchmark, optionally under both split strategies (A6). |
| `mna_pipeline/cli.py` | Argument parsing and process entry point. |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python version is pinned in `.python-version` (3.11.15) and
`pyproject.toml` (`>=3.11,<3.12`). `requirements.txt` pins every direct
dependency to the exact version this codebase was implemented and tested
against; `requirements.lock` is the full transitive closure captured via
`pip freeze` in a clean virtual environment on 2026-07-15 (A8). To
verify a fresh install matches:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip freeze | sort > /tmp/got.txt
grep -v '^#' requirements.lock | sort > /tmp/want.txt
diff /tmp/got.txt /tmp/want.txt   # should be empty
```

## Usage

```bash
python src/main.py \
  --input-csv /path/to/Model\ Training\ Data.csv \
  --output-dir cache_light_rq \
  --seed 42 \
  --smote-sample-size 2000 \
  --top-k 10 \
  --split-strategy temporal
```

| Flag | Description | Default |
|---|---|---|
| `--input-csv` | Path to input CSV | **required** |
| `--output-dir` | Directory for cached models, persisted preprocessors, and `run_metadata.json` | `cache_light_rq` |
| `--seed` | Random seed | `42` |
| `--smote-sample-size` | Max training rows sub-sampled before SMOTE | `2000` |
| `--top-k` | Number of top features selected by RF importance | `10` |
| `--split-strategy` | `temporal` or `random` (A6) | `temporal` |
| `--single-split` | Only run `--split-strategy`, skip the side-by-side comparison run | off (both run by default) |
| `--search-mode` | `full` (widened grids, A5) or `fast` (old nominal debug grids) | `full` |
| `--n-iter` / `--cv-folds` | RandomizedSearchCV budget in `full` mode | `20` / `5` |
| `--strict` | Convert optional-step warnings (SHAP per-model, DCA) into hard failures (A9) | off |
| `--tabpfn-device` | `cpu`, `cuda`, or `auto` | `cpu` |
| `--tabpfn-checkpoint` | Override the pinned TabPFN-3 checkpoint filename | binary classification checkpoint (see below) |
| `--allow-cpu-large-dataset` | Override the TabPFN-3 CPU sample-count guardrail (Part B item 3) | off |
| `--shap-background-size` | SHAP background sample size (A3) | `100` |
| `--shap-explain-size` | SHAP explained-subset size, capped at `len(X_test)` (A3) | `200` |

`--tabpfn-time` (the old TabPFN-2.x time-budget flag) is accepted but
deprecated and ignored with a warning: TabPFN-3's constructor has no
`max_time` parameter (see "TabPFN-3 migration" below).

Results, cached models, persisted preprocessors, and `run_metadata.json`
are written under `--output-dir`.

## Confirmed defects and fixes (Part A)

| ID | Defect | Fix | Where | Test |
|---|---|---|---|---|
| A1 | One-hot/median-impute fit on full data before split (leakage) | `ColumnTransformer` (SimpleImputer+OneHotEncoder) fit on train only, persisted with joblib | `preprocessing.py` | `test_preprocessing.py`, `test_leakage_empirical.py` |
| A2 | DCA `modelnames` derived as "all columns except status", passing raw features into `dca()` | Explicit 6-model whitelist (`CANONICAL_MODEL_NAMES`) + `[0,1]` probability assertion before calling `dca()` | `dca.py` | `test_dca.py` |
| A3 | SHAP background and explained set both hardcoded to 1 sample | Configurable background (default 100, `shap.sample`) and explained subset (default `min(200, len(X_test))`); mean\|SHAP\| is the primary output | `shap_explain.py` | `test_shap_explain.py` |
| A4 | Validation split created and never used | Val AUC/Brier/ECE logged for every tuned candidate before test metrics are computed | `baselines.py` (module docstring documents the 70/15/15 rationale) | `test_baselines.py` |
| A5 | `RandomizedSearchCV(n_iter=2, cv=2)` over grids as small as 1 candidate | Widened grids (3-5+ candidates, multiple hyperparameters per model), `n_iter=20`, 5-fold stratified CV by default; old grids kept behind `--search-mode fast` | `baselines.py` | `test_baselines.py` |
| A6 | Random-stratified split only; no temporal validation | `--split-strategy {temporal,random}`, `temporal` is the new default; both are always computed and reported side by side | `splits.py`, `pipeline.py` | `test_splits.py`, `test_pipeline_integration.py` |
| A7 | Thesis text says "TabPFN v1"; code actually ran v2.x; no machine-verified identifier printed | Installed `tabpfn` version and exact checkpoint filename logged at run start and written to `run_metadata.json` | `metadata.py`, `tabpfn_model.py` | `test_metadata.py` |
| A8 | `requirements.txt` pinned nothing but a floor | Exact pins in `requirements.txt`, full closure in `requirements.lock`, Python pinned in `.python-version`/`pyproject.toml` | `requirements.txt`, `requirements.lock`, `.python-version` | Part C item 1 |
| A9 | Broad `except Exception: warn; continue` could mask a broken core step | Core steps (load, split, preprocessing fit, SMOTE, baseline tuning, TabPFN-3 training, final metrics) raise and halt; only SHAP-per-model and DCA warn by default; `--strict` escalates those to hard failures | throughout; policy centered in `pipeline.py`/`baselines.py`/`tabpfn_model.py` | `test_strict_mode.py`, `test_tabpfn_model.py::test_train_tabpfn_core_failure_propagates_not_swallowed` |
| A10 | `X[col].fillna(..., inplace=True)` chained assignment, unreliable under pandas 2.x copy-on-write | Eliminated entirely by the A1 `ColumnTransformer` | `preprocessing.py` | same as A1 |
| A11 | No tests | Full suite below | `tests/` | this repository's `tests/` directory |

Two additional, related defects were found and fixed while implementing
the above (flagged here explicitly, since they were not in the original
audited list):

- **SHAP/DCA scale mismatch**: the 5 baseline models are fit on
  `StandardScaler`-transformed features, but the original scripts' SHAP
  section (and an early draft of this fix) explained them using
  unscaled feature values, which would silently apply coefficients/split
  thresholds learned in standardized space to unstandardized inputs.
  Fixed in `pipeline.py` by explaining/predicting each model on the
  representation it was actually fit on (scaled for the 5 baselines, raw
  for TabPFN-3, which is not scaled per TabPFN's own guidance).
- Two scikit-learn 1.9-specific deprecations were hit while widening the
  hyperparameter grids (`LogisticRegression(penalty=...)` and
  `SVC(probability=True)`, both deprecated in the actually-installed
  scikit-learn 1.9.0) and fixed by using `l1_ratio` directly and
  `CalibratedClassifierCV` respectively; see `baselines.py`'s module
  docstring for the full explanation.

## TabPFN-3 migration (Part B)

Verified directly against the installed package (`pip show tabpfn` ->
`8.1.0`, installed and checked 2026-07-15) and its own source, not
assumed to be a drop-in version bump from TabPFN-2.x:

1. **API surface**: `TabPFNClassifier.__init__` in `tabpfn==8.1.0` has
   no `max_time` parameter at all (confirmed via
   `inspect.signature`); the ensemble/compute budget is now controlled
   by `n_estimators` (default 8) and `auto_scale_n_estimators`. `n_jobs`
   is deprecated in favor of `n_preprocessing_jobs`. `random_state`
   defaults to `0` (deterministic) rather than v2.x's non-deterministic
   default. `tabpfn.settings.TabPFNSettings.model_version` defaults to
   `ModelVersion.V3`, confirming `pip install tabpfn` installs TabPFN-3
   by default today, consistent with Prior Labs' documentation
   (https://docs.priorlabs.ai/models, checked 2026-07-15) and the
   TabPFN-3 technical report (arXiv:2605.13986, May 2026).

2. **Checkpoint**: pinned to
   `tabpfn-v3-classifier-v3_20260417_binary.ckpt`, the
   binary-classification-specific checkpoint shipped in the
   `Prior-Labs/tabpfn_3` HuggingFace repo (per
   `tabpfn.model_loading.ModelSource.get_classifier_v3()` in the
   installed package), matching this project's binary `status` target
   rather than the generic default or multiclass/OOD variants. Logged
   and written to `run_metadata.json` on every run (A7).

3. **CPU guardrail**: `TABPFN3_CPU_HARD_LIMIT_SAMPLES = 1000` and
   `TABPFN3_CPU_WARN_LIMIT_SAMPLES = 200` in `tabpfn_model.py`, sourced
   directly from the installed `tabpfn==8.1.0` source
   (`tabpfn/validation.py::_validate_num_samples_for_cpu`, checked
   2026-07-15), not carried over from old TabPFN-2 guidance. The
   pipeline enforces this itself (`guard_cpu_dataset_size`, raising
   `CPUDatasetTooLargeError`) before ever constructing a TabPFN-3
   classifier, in addition to the installed library's own internal
   check. Empirically, this guardrail fires organically on real runs
   whenever SMOTE-balanced training data exceeds 1,000 rows on the
   default CPU device (see Part C verification log).

4. **License**: see `LICENSE_NOTES.md` for the full analysis. Summary:
   the `tabpfn` package's code is Apache-2.0-derived (Prior Labs
   License v1.2); the TabPFN-3 model *weights* are separately licensed
   under the "TABPFN-3 License v1.0" (non-commercial; research/
   evaluation permitted). Academic thesis use is compliant under that
   description; `LICENSE_NOTES.md` flags the one caveat (this repository
   could not directly fetch and quote the full legal text from this
   sandboxed environment) and what to re-check before any non-academic
   reuse.

5. **Benchmark re-run**: see "Known Limitations" — the real TabPFN-3
   weights are gated behind Prior Labs' license acceptance and require a
   `TABPFN_TOKEN`, which was not available in the environment this
   remediation was implemented in, so the AUC=0.9168/ECE=0.0676 numbers
   previously reported for TabPFN-2.x could not be directly reproduced
   or compared against real TabPFN-3 numbers here. The full pipeline
   (including the real `TabPFNClassifier` construction, checkpoint
   resolution, and CPU guardrail) was verified end-to-end up to the
   point where weight download requires that token; see Part C below
   for the exact commands and log output. Whoever has a `TABPFN_TOKEN`
   available (`export TABPFN_TOKEN=...` before running) can complete
   this re-run with no code changes.

6. **Version consistency**: every place that used to say "TabPFN" or
   "TabPFN (v2.x)" without a verifiable identifier now says "TabPFN-3"
   and logs/records the exact package version and checkpoint filename
   (this README, `run_metadata.json`, and every log line that reports a
   TabPFN-3 metric).

## Verification protocol (Part C)

All of the following were executed against this repository; commands
and the resulting log output are preserved in this PR's description and
in `tests/`. Two items could not be run against the real dataset (no
real CSV in this environment) and are marked accordingly.

1. **Clean-environment install**: verified — fresh venv,
   `pip install -r requirements.txt`, `pip freeze` matches
   `requirements.lock` exactly (diff is empty).
2. **Full test suite**: verified — `pytest -W error -v` passes all 56
   tests, 0 failures, 0 warnings, 0 skips.
3. **Leakage check (empirical)**: verified —
   `tests/test_leakage_empirical.py` fits the preprocessor on a train
   partition, then perturbs a "test" partition (shifts a numeric column
   by +1,000,000, injects a never-seen category) and asserts the fitted
   transformer's imputer statistics and one-hot categories are
   byte-for-byte unchanged.
4. **DCA fix, real data**: run against a real, non-degenerate synthetic
   dataset (real CSV unavailable; see "Known Limitations"). Logged
   `DCA modelnames: ['LR', 'RF', 'GB', 'SVM', 'MLP', 'TabPFN-3']` -
   exactly the six model names, nothing else. Also covered by
   `tests/test_dca.py::test_run_dca_real_call_passes_exactly_six_canonical_modelnames`,
   which calls the real `dcurves.dca()` (no network/gating required).
5. **SHAP sample sizes**: verified in a real pipeline run — logged
   `n_background=100`, `n_explain=200` for all six models.
6. **Hyperparameter search non-triviality**: verified — a real
   `--search-mode full` run logged `unique param combos evaluated: 20`
   for every one of LR/RF/GB/SVM/MLP, including GB (the model the
   original audit specifically flagged as having only 1 candidate).
7. **TabPFN-3 CPU guardrail**: verified against a real oversized
   condition — a real run with 2,306 SMOTE-resampled training rows on
   `--tabpfn-device cpu` raised
   `CPUDatasetTooLargeError: Refusing to run TabPFN-3 on CPU with 2306
   samples: exceeds the documented guardrail of 1000 samples (source:
   tabpfn==8.1.0, tabpfn/validation.py::_validate_num_samples_for_cpu,
   checked 2026-07-15)`. The warning tier was also observed for real at
   820 samples: `Running TabPFN-3 on CPU with 820 samples (> 200) may be
   slow.`
8. **Temporal split ordering**: verified on a realistic synthetic date
   range (real CSV unavailable) — logged train/val/test date ranges
   with strict, non-overlapping ordering (train max <= val min <= val
   max <= test min).
9. **End-to-end sane metrics, all six models**: verified on synthetic
   data with the real pipeline logic and a dependency-injected TabPFN-3
   stand-in (see "Known Limitations" for why); all six models produced
   finite AUC/F1 in a non-degenerate range (~0.64-0.79) under both
   `random` and `temporal` splits, no unexplained NaN.
10. **Strict mode, both ways**: verified —
    `tests/test_strict_mode.py` runs the real `run_pipeline` with an
    injected DCA/SHAP failure; `strict=False` completes with a warning
    logged, `strict=True` raises. Also observed for real (not a test
    double) when the actual TabPFN-3 weight download hit the license
    gate: the pipeline halted with a full traceback rather than
    reporting a NaN metric, since TabPFN-3 training is a core step (A9).
11. **Model identity traceability**: verified — the real run's log
    (`Downloading model to
    /root/.cache/tabpfn/tabpfn-v3-classifier-v3_20260417_binary.ckpt.`)
    matches the checkpoint pinned in `tabpfn_model.py`, logged at
    runtime, written to `run_metadata.json`
    (`tabpfn_package_version: "8.1.0"`,
    `tabpfn_checkpoint: "tabpfn-v3-classifier-v3_20260417_binary.ckpt"`),
    and documented in this README's "TabPFN-3 migration" section above.

## Test suite (A11 / Part D)

`tests/` covers every module with at least one happy-path and one
boundary-case test (empty input, single-class target, all-missing
column, extreme class imbalance, as applicable per function), plus:

- `tests/test_leakage_empirical.py`: A1 empirical leakage check.
- `tests/test_dca.py`: A2 regression tests, including a real (non-mocked)
  `dcurves.dca()` call.
- `tests/test_shap_explain.py`: A3 minimum-sample-size regression tests.
- `tests/test_baselines.py`: A5 non-trivial-search regression tests.
- `tests/test_splits.py`: A6 temporal-ordering tests.
- `tests/test_strict_mode.py`: A9 fail-loud tests, both ways.
- `tests/test_pipeline_integration.py`: full end-to-end integration test
  on a small, fixed-seed synthetic dataset (`tests/fixtures.py`),
  asserting the pipeline completes and produces non-null metrics for
  every model, without requiring the real CSV.

Run with:

```bash
pytest -W error -v
```

## Known Limitations

- **Real dataset unavailable in this environment**: the 4,400-row CSV
  referenced by the thesis was not present in this repository or
  execution environment. Every fix was implemented against the real
  code paths and verified either against a realistic synthetic dataset
  (`tests/fixtures.py::make_synthetic_dataset`, matching every column
  name and semantic verifiable from the original scripts and the Annex D
  log) or, where possible, run for real up to the point where the real
  CSV or gated weights were required. Anyone with access to the real CSV
  should re-run `pytest -W error -v` plus the Part C commands against
  it; no code changes should be required.
- **TabPFN-3 weights are gated and were not downloaded**: Prior Labs
  requires one-time interactive license acceptance or a `TABPFN_TOKEN`
  to download TabPFN-3 checkpoints, and neither was available in this
  sandboxed environment. The pipeline's real integration with the
  `tabpfn` package (checkpoint resolution, CPU guardrail, construction,
  and the fit call) was verified for real up to that gate (see Part C
  item 7/10/11); the test suite exercises the rest of the pipeline
  through a dependency-injected stand-in classifier
  (`tests/fixtures.py::FakeTabPFNClassifier`), clearly labeled as such
  everywhere it is used. Set `TABPFN_TOKEN` to complete a real run.
- **TabPFN-2.x vs TabPFN-3 numeric comparison not completed**: because
  real TabPFN-3 weights could not be downloaded here, the AUC=0.9168 /
  ECE=0.0676 numbers previously reported for TabPFN-2.x could not be
  directly compared against a real TabPFN-3 run in this environment
  (Part B item 5 is implemented, not empirically verified with real
  weights).
- **`--compare-splits` (A6) roughly doubles runtime**: computing both
  `random` and `temporal` benchmarks in one run means hyperparameter
  search and TabPFN-3 training each happen twice. `--single-split` is
  available for fast iteration; this is a deliberate, documented
  trade-off (see `pipeline.py` module docstring), not a shortcut around
  the requirement to report both.
- **Full-text TABPFN-3 License v1.0 not directly quoted**: this
  repository's license analysis (`LICENSE_NOTES.md`) is based on Prior
  Labs' documentation and indexed summaries, not a direct reading of the
  full legal text, because the canonical license page was not reachable
  from this environment. Flagged explicitly in `LICENSE_NOTES.md` with a
  recommendation to re-check before non-academic reuse.
- **`--compare-splits` model-selection scope**: A4's validation-based
  comparison logs AUC/Brier/ECE per candidate on the validation set but
  does not drop or re-rank models based on it, since this is a
  benchmark-comparison study (all six models are reported), not a
  single-model deployment pipeline. Documented in `baselines.py`'s
  module docstring.
