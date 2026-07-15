"""Run metadata: model identity and environment capture (A7).

Confirmed defect A7 (model identity confusion): the thesis text claimed
"TabPFN v1" while the code actually ran v2.x, and nothing printed a
machine-verified checkpoint identifier alongside reported metrics. This
module makes that ambiguity structurally impossible to repeat: every run
logs and persists the exact installed tabpfn package version and the
exact checkpoint filename/identifier used, plus enough environment
detail to reproduce a result.
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class RunMetadata:
    """Machine-verified identity of the model and environment for one run.

    Inputs: none required at construction; fields are filled in by
    `collect_environment` and by the caller after resolving the TabPFN
    checkpoint actually used.
    Outputs: a JSON-serializable record written to <output_dir>/run_metadata.json.
    Assumes nothing about upstream state; this is the root of the metadata chain.
    """

    tabpfn_package_version: Optional[str] = None
    tabpfn_checkpoint: Optional[str] = None
    tabpfn_model_version_tag: Optional[str] = None
    tabpfn_device: Optional[str] = None
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform_string: str = field(default_factory=platform.platform)
    seed: Optional[int] = None
    split_strategy: Optional[str] = None
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Return a plain dict representation. No inputs. No upstream assumptions."""
        return asdict(self)

    def write(self, path: pathlib.Path) -> None:
        """Write this metadata as pretty JSON to `path`.

        Input: path is the destination file (parent directories must exist
        or will be created). Output: none, writes to disk.
        Assumes the caller has finished populating every field it cares about;
        this only serializes the current state.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


def get_tabpfn_package_version() -> Optional[str]:
    """Return the installed tabpfn package version string, or None if not installed.

    No inputs. Output: version string (e.g. "8.1.0") or None.
    Assumes nothing; safe to call even if tabpfn is not importable.
    """
    try:
        import tabpfn

        return getattr(tabpfn, "__version__", None)
    except ImportError:
        return None


def collect_environment(seed: int, split_strategy: str) -> RunMetadata:
    """Build a RunMetadata populated with package/environment facts.

    Input: seed (int) and split_strategy ("random" or "temporal") used for
    this run. Output: a RunMetadata with tabpfn_package_version,
    python_version, platform_string, seed and split_strategy filled in;
    tabpfn_checkpoint/tabpfn_device/tabpfn_model_version_tag are left for
    the caller to fill in once the TabPFN model is actually constructed,
    since only the model-construction code knows which checkpoint resolved.
    Assumes no upstream state.
    """
    return RunMetadata(
        tabpfn_package_version=get_tabpfn_package_version(),
        seed=seed,
        split_strategy=split_strategy,
    )
