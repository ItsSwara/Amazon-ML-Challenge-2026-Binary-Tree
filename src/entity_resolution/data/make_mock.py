"""Build a linked mock sample of the dataset in data/mock/.

Train side (genuinely linked):
  * the N lowest-numbered train_source1 references,
  * their train_ground_truth rows, verbatim,
  * every S2/S3 row those references match, plus an equal number of randomly
    drawn S2/S3 rows matched to none of the N references (real negatives).

Test side (no ground truth, so linkage is NOT guaranteed):
  * the N lowest-numbered test_source1 rows,
  * a uniform random sample of test_source2/3 of size N * (file rows / test_source1 rows).

Rows are copied byte-for-byte (same header, encoding, control characters) and
the output files keep the original filenames. Originals are read-only.
Output row order is shuffled (seeded) so file position carries no signal.

Run:  python src/entity_resolution/data/make_mock.py [--refs 2000] [--seed 42]
"""

import argparse
import heapq
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from entity_resolution.config import get_dataset_dir, get_mock_dir  # noqa: E402

NL = b"\n"


def numeric_id(line: bytes) -> int:
    """'S1-965667\\t...' -> 965667"""
    return int(line.split(b"\t", 1)[0].split(b"-", 1)[1])


def line_id(line: bytes) -> str:
    return line.split(b"\t", 1)[0].decode("utf-8")


def read_header(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.readline()


def count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(NL)
    return n - 1  # header


def data_lines(path: Path):
    """Yield data lines as bytes, each ending in a newline."""
    with open(path, "rb") as f:
        f.readline()
        for line in f:
            yield line if line.endswith(NL) else line + NL


def lowest_ids(path: Path, n: int) -> list[bytes]:
    """The n data lines with the smallest numeric entity_id."""
    return heapq.nsmallest(n, data_lines(path), key=numeric_id)


def select_lines(path: Path, keep: set[str], n_random: int, rng: random.Random) -> tuple[list[bytes], list[bytes]]:
    """One pass over path.

    Returns (rows whose id is in keep, reservoir sample of n_random rows whose id is not in keep).
    """
    matched: list[bytes] = []
    reservoir: list[bytes] = []
    seen = 0
    for line in data_lines(path):
        if line_id(line) in keep:
            matched.append(line)
            continue
        seen += 1
        if len(reservoir) < n_random:
            reservoir.append(line)
        else:
            j = rng.randrange(seen)
            if j < n_random:
                reservoir[j] = line
    return matched, reservoir


def write_lines(path: Path, header: bytes, lines: list[bytes], rng: random.Random) -> None:
    lines = list(lines)
    rng.shuffle(lines)
    with open(path, "wb") as f:
        f.write(header)
        f.writelines(lines)


def build_train(train_dir: Path, mock_dir: Path, n_refs: int, rng: random.Random) -> None:
    s1_path, gt_path = train_dir / "train_source1.tsv", train_dir / "train_ground_truth.tsv"
    s1_lines = lowest_ids(s1_path, n_refs)
    refs = {line_id(line) for line in s1_lines}

    gt_lines = [line for line in data_lines(gt_path) if line_id(line) in refs]
    assert len(gt_lines) == len(refs), "every reference must have a ground-truth row"
    matched_ids: set[str] = set()
    for line in gt_lines:
        matched_ids.update(i for i in line.rstrip(NL).split(b"\t")[1].decode("utf-8").split(",") if i)

    write_lines(mock_dir / "train_source1.tsv", read_header(s1_path), s1_lines, rng)
    write_lines(mock_dir / "train_ground_truth.tsv", read_header(gt_path), gt_lines, rng)

    for src in ("source2", "source3"):
        path = train_dir / f"train_{src}.tsv"
        prefix = "S2-" if src == "source2" else "S3-"
        want = {i for i in matched_ids if i.startswith(prefix)}
        pos, neg = select_lines(path, want, len(want), rng)
        assert len(pos) == len(want), f"{src}: {len(want) - len(pos)} matched IDs missing from file"
        write_lines(mock_dir / path.name, read_header(path), pos + neg, rng)
        print(f"train_{src}.tsv: {len(pos):,} matched + {len(neg):,} non-matching = {len(pos) + len(neg):,} rows")

    n_zero = sum(1 for line in gt_lines if not line.rstrip(NL).split(b"\t")[1])
    print(f"train_source1.tsv: {len(s1_lines):,} references ({n_zero:,} with no matches); "
          f"train_ground_truth.tsv: {len(gt_lines):,} rows, {len(matched_ids):,} matched IDs")


def build_test(test_dir: Path, mock_dir: Path, n_rows: int, rng: random.Random) -> None:
    s1_path = test_dir / "test_source1.tsv"
    s1_lines = lowest_ids(s1_path, n_rows)
    write_lines(mock_dir / "test_source1.tsv", read_header(s1_path), s1_lines, rng)
    total_s1 = count_rows(s1_path)
    print(f"test_source1.tsv: {len(s1_lines):,} of {total_s1:,} rows")

    for src in ("source2", "source3"):
        path = test_dir / f"test_{src}.tsv"
        k = round(count_rows(path) * n_rows / total_s1)
        _, sample = select_lines(path, set(), k, rng)
        write_lines(mock_dir / path.name, read_header(path), sample, rng)
        print(f"test_{src}.tsv: {len(sample):,} random rows (proportional; linkage not guaranteed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refs", type=int, default=2000, help="train references / test source1 rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = get_dataset_dir()
    mock_dir = get_mock_dir()
    mock_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    build_train(dataset_dir / "train", mock_dir, args.refs, rng)
    build_test(dataset_dir / "test", mock_dir, args.refs, rng)
    print(f"wrote {mock_dir}")


if __name__ == "__main__":
    main()
