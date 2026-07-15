from __future__ import annotations

import pathlib

import pytest

from mna_pipeline.cli import parse_args


def test_parse_args_defaults(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("id,status\n0,1\n")
    args = parse_args(["--input-csv", str(csv)])
    assert args.split_strategy == "temporal"
    assert args.compare_splits is True
    assert args.search_mode == "full"
    assert args.strict is False


def test_parse_args_single_split_and_fast_mode(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("id,status\n0,1\n")
    args = parse_args(
        ["--input-csv", str(csv), "--single-split", "--search-mode", "fast", "--strict"]
    )
    assert args.compare_splits is False
    assert args.search_mode == "fast"
    assert args.strict is True


def test_parse_args_deprecated_tabpfn_time_warns(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("id,status\n0,1\n")
    with pytest.warns(UserWarning, match="deprecated"):
        parse_args(["--input-csv", str(csv), "--tabpfn-time", "10"])
