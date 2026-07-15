from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from mna_pipeline.data import load_data
from fixtures import write_synthetic_csv


def test_load_data_happy_path(tmp_path: pathlib.Path):
    csv_path = write_synthetic_csv(tmp_path / "data.csv", n=50, seed=1)
    df = load_data(csv_path)
    assert "status" in df.columns
    assert len(df) == 50
    assert pd.api.types.is_datetime64_any_dtype(df["announcement_date"])


def test_load_data_missing_file(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


def test_load_data_missing_status_column(tmp_path: pathlib.Path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.index.name = "id"
    path = tmp_path / "no_status.csv"
    df.to_csv(path)
    with pytest.raises(ValueError):
        load_data(path)
