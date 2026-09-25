"""Machine-specific path configuration.

Reads configs/paths.local.yaml (gitignored) and falls back to
configs/paths.example.yaml with a warning. Import get_dataset_dir()
instead of hardcoding dataset paths.
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
LOCAL_CONFIG = CONFIG_DIR / "paths.local.yaml"
EXAMPLE_CONFIG = CONFIG_DIR / "paths.example.yaml"


def _load_paths() -> dict:
    if LOCAL_CONFIG.exists():
        source = LOCAL_CONFIG
    else:
        source = EXAMPLE_CONFIG
        print(
            f"WARNING: {LOCAL_CONFIG} not found; falling back to {EXAMPLE_CONFIG.name}, "
            "which contains placeholder paths.\n"
            f"Copy {EXAMPLE_CONFIG.name} to {LOCAL_CONFIG.name} and set dataset_dir "
            "to your real dataset location.",
            file=sys.stderr,
        )
    with open(source, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_dataset_dir() -> Path:
    """Absolute path to the folder holding train_source1.tsv, test_source1.tsv, etc."""
    return Path(_load_paths()["dataset_dir"])


def get_mock_dir() -> Path:
    """Mock data directory; relative paths resolve against the repo root."""
    path = Path(_load_paths().get("mock_dir", "data/mock"))
    return path if path.is_absolute() else REPO_ROOT / path
