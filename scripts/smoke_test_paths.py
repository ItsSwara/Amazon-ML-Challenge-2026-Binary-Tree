"""Confirm the configured dataset directory exists and the source1 TSVs are readable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entity_resolution.config import get_dataset_dir  # noqa: E402

REQUIRED = ["train/train_source1.tsv", "test/test_source1.tsv"]


def main() -> int:
    dataset_dir = get_dataset_dir()
    print(f"dataset_dir: {dataset_dir}")
    if not dataset_dir.is_dir():
        print("FAIL: dataset_dir does not exist or is not a directory")
        return 1

    ok = True
    for name in REQUIRED:
        path = dataset_dir / name
        try:
            with open(path, encoding="utf-8") as f:
                header = f.readline().rstrip("\n")
            print(f"OK   {name}: header = {header[:100]!r}")
        except OSError as e:
            print(f"FAIL {name}: {e}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
