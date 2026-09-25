"""Write the first N data rows of each of the 7 dataset files to data/mock/.

Bytes are copied verbatim (header + first N lines, split on b"\\n" only), so
filenames, schema, encoding and any control characters are identical to the
originals. The originals are opened read-only.

Run:  python src/entity_resolution/data/make_mock.py [--rows 10000]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from entity_resolution.config import get_dataset_dir, get_mock_dir  # noqa: E402

# Path of each file relative to dataset_dir.
DATASET_FILES = [
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
]


def head_lines(src: Path, dst: Path, n_rows: int) -> int:
    """Copy the header plus the first n_rows data lines; return data rows written."""
    written = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(fin.readline())  # header
        for line in fin:
            fout.write(line)
            written += 1
            if written == n_rows:
                break
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=10_000, help="data rows per file")
    args = parser.parse_args()

    dataset_dir = get_dataset_dir()
    mock_dir = get_mock_dir()
    mock_dir.mkdir(parents=True, exist_ok=True)

    for rel in DATASET_FILES:
        src = dataset_dir / rel
        dst = mock_dir / Path(rel).name
        n = head_lines(src, dst, args.rows)
        print(f"{rel:32s} -> {dst}  ({n:,} rows)")


if __name__ == "__main__":
    main()
