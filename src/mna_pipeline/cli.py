"""Command-line interface."""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
import warnings
from typing import Iterable, Optional

import numpy as np

from .pipeline import PipelineArgs, run_pipeline

log = logging.getLogger(__name__)


def parse_args(argv: Optional[Iterable[str]] = None) -> PipelineArgs:
    """Parse CLI arguments into a PipelineArgs.

    Input: argv, an optional iterable of CLI argument strings (defaults
    to sys.argv[1:] via argparse). Output: a populated PipelineArgs.
    Raises SystemExit via argparse on invalid input.
    Assumes no upstream state.
    """
    p = argparse.ArgumentParser(
        description="Train tuned baselines and TabPFN-3, explain with SHAP, and run DCA."
    )
    p.add_argument("--input-csv", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("cache_light_rq"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smote-sample-size", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)

    p.add_argument(
        "--split-strategy",
        choices=["temporal", "random"],
        default="temporal",
        help="Primary split strategy (A6). Default temporal: sort by announcement_date, "
        "earliest ~70%% train / next ~15%% val / most recent ~15%% test.",
    )
    p.add_argument(
        "--single-split",
        action="store_true",
        help="Only run the strategy given by --split-strategy, instead of computing both "
        "random and temporal for side-by-side comparison (A6). Faster; use for debugging/CI.",
    )

    p.add_argument(
        "--search-mode",
        choices=["full", "fast"],
        default="full",
        help="full (A5 default): widened grids, n_iter>=20, 5-fold CV. fast: the old "
        "nominal debug-only grids/settings, never the default.",
    )
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--cv-folds", type=int, default=5)

    p.add_argument(
        "--strict",
        action="store_true",
        help="Convert optional-step warnings (SHAP per-model, DCA) into hard failures (A9).",
    )

    p.add_argument("--tabpfn-device", type=str, default="cpu")
    p.add_argument(
        "--tabpfn-checkpoint",
        type=str,
        default=None,
        help="Override the TabPFN-3 checkpoint filename. Defaults to the pinned binary "
        "classification checkpoint (see tabpfn_model.TABPFN3_CHECKPOINT).",
    )
    p.add_argument(
        "--allow-cpu-large-dataset",
        action="store_true",
        help="Override the TabPFN-3 CPU sample-count guardrail (Part B item 3).",
    )

    p.add_argument("--shap-background-size", type=int, default=100)
    p.add_argument("--shap-explain-size", type=int, default=200)

    p.add_argument(
        "--tabpfn-time",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    ns = p.parse_args(list(argv) if argv is not None else None)

    if ns.tabpfn_time is not None:
        warnings.warn(
            "--tabpfn-time is deprecated and ignored: TabPFN-3's constructor "
            "(tabpfn==8.1.0) has no max_time parameter (see tabpfn_model.py migration "
            "notes). Use --search-mode/--n-iter or TabPFNClassifier's own n_estimators "
            "instead if you need to control compute budget.",
            stacklevel=2,
        )

    from .tabpfn_model import TABPFN3_CHECKPOINT

    return PipelineArgs(
        input_csv=ns.input_csv,
        output_dir=ns.output_dir,
        seed=ns.seed,
        smote_sample_size=ns.smote_sample_size,
        top_k=ns.top_k,
        split_strategy=ns.split_strategy,
        compare_splits=not ns.single_split,
        search_mode=ns.search_mode,
        n_iter=ns.n_iter,
        cv_folds=ns.cv_folds,
        strict=ns.strict,
        tabpfn_device=ns.tabpfn_device,
        tabpfn_checkpoint=ns.tabpfn_checkpoint or TABPFN3_CHECKPOINT,
        allow_cpu_large_dataset=ns.allow_cpu_large_dataset,
        shap_background_size=ns.shap_background_size,
        shap_explain_size=ns.shap_explain_size,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point: parse args, run the pipeline, log total runtime.

    Input: argv, optional CLI args (for testing); Output: None (side
    effects: logging, files written under args.output_dir).
    Raises whatever run_pipeline raises for core-step failures (A9);
    this function does not catch exceptions, so a failing run exits
    with a non-zero status and a visible traceback.
    Assumes no upstream state; this is the process entry point.
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.captureWarnings(True)
    np.random.seed(args.seed)

    t0 = time.time()
    log.info("=== PIPELINE START ===")
    run_pipeline(args)
    log.info("Total time: %.1fs", time.time() - t0)
    log.info("=== COMPLETE ===")


if __name__ == "__main__":  # pragma: no cover
    main()
