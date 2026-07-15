from __future__ import annotations

import json

import tabpfn

from mna_pipeline.metadata import collect_environment, get_tabpfn_package_version


def test_get_tabpfn_package_version_matches_installed_package():
    assert get_tabpfn_package_version() == tabpfn.__version__


def test_collect_environment_happy_path():
    meta = collect_environment(seed=123, split_strategy="temporal")
    assert meta.seed == 123
    assert meta.split_strategy == "temporal"
    assert meta.tabpfn_package_version == tabpfn.__version__


def test_run_metadata_write_roundtrip(tmp_path):
    meta = collect_environment(seed=1, split_strategy="random")
    meta.tabpfn_checkpoint = "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
    meta.tabpfn_device = "cpu"
    path = tmp_path / "sub" / "run_metadata.json"
    meta.write(path)
    data = json.loads(path.read_text())
    assert data["tabpfn_checkpoint"] == "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
    assert data["tabpfn_package_version"] == tabpfn.__version__
    assert data["seed"] == 1
