"""Load machine-specific external data paths for the curtailment workflow.

Tracked scripts deliberately keep local SAPN/EVM/BOM locations out of Git.
Create `local_paths.py` next to this file by copying `local_paths.example.py`
and filling in the paths for your machine.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
LOCAL_PATHS_FILE = SCRIPTS_DIR / "local_paths.py"
LOCAL_PATHS_EXAMPLE_FILE = SCRIPTS_DIR / "local_paths.example.py"


def _setup_message() -> str:
    return (
        f"Create {LOCAL_PATHS_FILE} from {LOCAL_PATHS_EXAMPLE_FILE} and fill in the "
        "machine-specific SAPN/EVM/BOM paths before running this script."
    )


@lru_cache(maxsize=1)
def _local_paths_module() -> ModuleType:
    if not LOCAL_PATHS_FILE.exists():
        raise RuntimeError(_setup_message())

    spec = importlib.util.spec_from_file_location(
        "sapn2022_curtailment_local_paths",
        LOCAL_PATHS_FILE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load local path settings from {LOCAL_PATHS_FILE}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_local_path(name: str, description: str) -> Path:
    """Return one required local path from `local_paths.py`.

    Paths are validated for presence here so each script can fail with the same
    clear setup guidance instead of committing machine-specific defaults.
    """
    module = _local_paths_module()
    value = getattr(module, name, None)
    if value is None:
        raise RuntimeError(f"{_setup_message()} Missing `{name}`: {description}")
    return Path(value)
