"""Build the reference-level train/validation split from train_ground_truth.tsv.

Each reference (source1_entity_id) is one unit, so its full matched-ID set
always lands on one side of the split. References are bucketed by
(country, match_count_bucket in {0, 1, 2+}) and split 80/20 within each
bucket, so both sides keep the real country and match-count proportions.

Writes train_reference_ids.txt and val_reference_ids.txt (one ID per line).

Run:  python src/entity_resolution/data/build_split.py --source mock
      python src/entity_resolution/data/build_split.py --source real
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from entity_resolution.config import REPO_ROOT, get_dataset_dir, get_mock_dir  # noqa: E402

REAL_OUT_DIR = Path(__file__).resolve().parent
BUCKETS = ["0", "1", "2+"]


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
        quoting=csv.QUOTE_NONE, encoding="utf-8", **kwargs,
    )


def match_counts(gt: pd.DataFrame) -> pd.Series:
    """Number of distinct matched IDs per reference, indexed like gt.

    If a reference appears on several rows, its matched sets are unioned first.
    """
    matched = gt["matched_entity_ids"]
    if gt["source1_entity_id"].is_unique:
        return pd.Series(
            np.where(matched == "", 0, matched.str.count(",") + 1), index=gt.index
        )
    union = (
        gt.assign(ids=matched.str.split(","))
        .groupby("source1_entity_id")["ids"]
        .agg(lambda lists: len({i for lst in lists for i in lst if i}))
    )
    return gt["source1_entity_id"].map(union)


def bucket_of(counts: pd.Series) -> pd.Series:
    return pd.Series(np.select([counts == 0, counts == 1], BUCKETS[:2], BUCKETS[2]), index=counts.index)


def stratified_split(strata: pd.Series, val_frac: float, seed: int) -> np.ndarray:
    """Boolean is_val mask; each stratum contributes round(val_frac * size) rows to val."""
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(strata), dtype=bool)
    for key in sorted(strata.unique()):
        rows = np.flatnonzero((strata == key).to_numpy())
        n_val = int(round(val_frac * len(rows)))
        is_val[rng.permutation(rows)[:n_val]] = True
    return is_val


def print_summary(df: pd.DataFrame) -> None:
    n_train, n_val = (~df["is_val"]).sum(), df["is_val"].sum()
    print(f"\nreferences: {len(df):,}  train: {n_train:,} ({n_train / len(df):.2%})  "
          f"val: {n_val:,} ({n_val / len(df):.2%})\n")
    hdr = (f"{'country':<8}{'bucket':<7}{'total':>11}{'train':>11}{'val':>10}{'val%':>8}"
           f"{'share all':>11}{'share train':>13}{'share val':>11}")
    print(hdr)
    print("-" * len(hdr))
    for country in sorted(df["country"].unique()):
        sub = df[df["country"] == country]
        n_all, n_tr, n_va = len(sub), (~sub["is_val"]).sum(), sub["is_val"].sum()
        for b in BUCKETS:
            s = sub[sub["bucket"] == b]
            tot, va = len(s), int(s["is_val"].sum())
            tr = tot - va
            print(f"{country:<8}{b:<7}{tot:>11,}{tr:>11,}{va:>10,}{(va / tot if tot else 0):>8.2%}"
                  f"{(tot / n_all if n_all else 0):>11.2%}{(tr / n_tr if n_tr else 0):>13.2%}"
                  f"{(va / n_va if n_va else 0):>11.2%}")
        print(f"{country:<8}{'ALL':<7}{n_all:>11,}{n_tr:>11,}{n_va:>10,}{n_va / n_all:>8.2%}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["mock", "real"], default="mock")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path,
                        help="default: src/entity_resolution/data (real) or data/processed/mock_split (mock)")
    args = parser.parse_args()

    mock = args.source == "mock"
    if mock:
        s1_path, gt_path = get_mock_dir() / "train_source1.tsv", get_mock_dir() / "train_ground_truth.tsv"
    else:
        d = get_dataset_dir() / "train"
        s1_path, gt_path = d / "train_source1.tsv", d / "train_ground_truth.tsv"
    out_dir = args.out_dir or (REPO_ROOT / "data" / "processed" / "mock_split" if mock else REAL_OUT_DIR)

    gt = read_tsv(gt_path)
    country_of = read_tsv(s1_path, usecols=["entity_id", "country"]).set_index("entity_id")["country"]

    df = pd.DataFrame({"ref": gt["source1_entity_id"]})
    df["country"] = df["ref"].map(country_of).fillna("UNKNOWN")
    df["bucket"] = bucket_of(match_counts(gt))

    n_unknown = int((df["country"] == "UNKNOWN").sum())
    if n_unknown:
        msg = f"{n_unknown:,} references have no row in train_source1 (country=UNKNOWN)"
        if not mock:
            raise SystemExit("ERROR: " + msg)
        print(f"WARNING: {msg}. Expected for the mock: its ground truth rows and "
              "source1 rows are different 10k samples.")

    # Whole reference = one unit, so a matched ID can only be split if two references share it.
    ids = gt["matched_entity_ids"].str.split(",").explode()
    dup_targets = int(ids[ids != ""].duplicated().sum())
    if dup_targets:
        print(f"WARNING: {dup_targets:,} matched IDs are claimed by more than one reference; "
              "those references can straddle train/val")

    df = df.drop_duplicates("ref").reset_index(drop=True)
    df["is_val"] = stratified_split(df["country"] + "|" + df["bucket"], args.val_frac, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in (("train", ~df["is_val"]), ("val", df["is_val"])):
        path = out_dir / f"{name}_reference_ids.txt"
        path.write_text("\n".join(df.loc[mask, "ref"]) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {int(mask.sum()):,} IDs -> {path}")

    print_summary(df)


if __name__ == "__main__":
    main()
