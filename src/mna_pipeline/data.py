"""Data loading.

Schema notes (verifiable only from the existing code, README, and the
Annex D script's own log output; nothing beyond this is assumed):

- The target column is `status` (binary, 0/1).
- Identifier/leak-prone columns dropped before modeling: `target_id`,
  `acquirer_id`, `announcement_date`, `exit_date`, `target_naics_code`.
  `announcement_date` is kept aside (not dropped from the loaded frame)
  because the temporal split (A6) needs it; it is dropped from the
  feature matrix later, in preprocessing.
- `attitude` is a categorical column (values observed in the Annex D log:
  Friendly, Hostile, Neutral) requiring one-hot encoding.
- The Annex D log records X.shape=(4400, 36) after one-hot encoding for
  the real dataset, which is not available in this environment. The
  exact full column list beyond what appears in code/logs is not
  fabricated here.
"""
from __future__ import annotations

import pathlib

import pandas as pd

REQUIRED_COLUMNS = ("status",)
DATE_COLUMN = "announcement_date"


def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load the raw M&A dataset from a CSV file.

    Input: path to a CSV file whose first column is an index column
    (matching the original script's `index_col=0` convention) and which
    contains at least a `status` column.
    Output: a pandas DataFrame with the raw columns, `announcement_date`
    parsed as datetime if present.
    Raises: FileNotFoundError if the path does not exist, ValueError if
    `status` is missing.
    Assumes no upstream state; this is the pipeline's entry point.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, index_col=0, low_memory=False)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from data")
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="raise")
    return df
