#!/usr/bin/env python3
"""CLI entry point for the M&A outcome prediction pipeline.

This file is a thin, backward-compatible wrapper. The implementation
lives in the modular src/mna_pipeline package (data loading, splitting,
leakage-safe preprocessing, resampling, baseline tuning, TabPFN-3
training, SHAP explanation, and Decision Curve Analysis are each their
own module; see README.md for the module map). Kept as `src/main.py` so
`python src/main.py --input-csv ...` continues to work unchanged.
"""
from __future__ import annotations

from mna_pipeline.cli import main

if __name__ == "__main__":  # pragma: no cover
    main()
