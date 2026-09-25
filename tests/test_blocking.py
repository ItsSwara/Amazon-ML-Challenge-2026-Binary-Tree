"""Checks for the Stage 3 blocker on tiny synthetic data.

Run:  python -m pytest tests/test_blocking.py
  or  python tests/test_blocking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from entity_resolution.blocking import blocker  # noqa: E402

WORDS = [
    "acme", "global", "trading", "pacific", "steel", "harbor", "bright", "logistics",
    "summit", "green", "valley", "farms", "iron", "works", "blue", "river", "cafe",
]


def make_data(n_s1=20, n_candidates=120, seed=0):
    rng = np.random.default_rng(seed)

    def rows(prefix, n):
        names = [" ".join(rng.choice(WORDS, 2)) for _ in range(n)]
        streets = [f"{rng.integers(1, 999)} {rng.choice(WORDS)} street" for _ in range(n)]
        return pd.DataFrame(
            {
                "entity_id": [f"{prefix}-{i}" for i in range(n)],
                "name_norm": names,
                "address_norm": streets,
            }
        )

    s1 = rows("S1", n_s1)
    candidates = pd.concat(
        [rows("S2", n_candidates // 2), rows("S3", n_candidates // 2)],
        ignore_index=True,
    )

    return s1, candidates


def retrieve(top_k, **kwargs):
    s1, candidates = make_data()
    retrieval = blocker.retrieve_candidates(
        s1, candidates, top_k=top_k, svd_dim=16, index_type="flat", **kwargs
    )
    return s1, candidates, retrieval


def test_default_top_k_is_50():
    assert blocker.TOP_K == 50


def test_top_k_cap_is_respected():
    for top_k, ann_k in [(5, None), (7, 3), (1, None)]:
        s1, _, retrieval = retrieve(top_k, ann_k=ann_k)
        counts = np.bincount(retrieval.s1_idx, minlength=len(s1))
        assert counts.max() <= top_k
        assert (counts == top_k).all()  # pool is far larger than K, so every row fills up


def test_scores_descend_and_pairs_are_unique():
    _, _, retrieval = retrieve(10)
    pairs = set(zip(retrieval.s1_idx.tolist(), retrieval.cand_idx.tolist()))
    assert len(pairs) == len(retrieval.score)
    for i in np.unique(retrieval.s1_idx):
        scores = retrieval.score[retrieval.s1_idx == i]
        assert (np.diff(scores) <= 1e-7).all()


def test_recall_counts_validation_ids_only():
    ground_truth = {
        "S1-val": {"S2-1", "S2-2"},
        "S1-train": {"S2-3", "S2-4"},
    }
    retrieved = {
        "S1-val": {"S2-1"},  # 1 of 2 found
        "S1-train": {"S2-3", "S2-4"},  # perfect, but must not count
    }

    result = blocker.calculate_blocking_recall(retrieved, ground_truth, {"S1-val"})
    assert (result.found, result.total, result.n_refs) == (1, 2, 1)
    assert result.recall == 0.5

    # Reference IDs absent from the ground truth are ignored, not counted as misses.
    result = blocker.calculate_blocking_recall(retrieved, ground_truth, {"S1-val", "S1-other"})
    assert (result.found, result.total, result.n_refs) == (1, 2, 1)

    # A validation reference with nothing retrieved counts as fully missed.
    result = blocker.calculate_blocking_recall({}, ground_truth, {"S1-val"})
    assert (result.found, result.total) == (0, 2)


def test_official_format_one_row_per_s1_with_valid_ids():
    s1, candidates, retrieval = retrieve(5)

    official = blocker.official_frame(
        retrieval, s1["entity_id"].to_numpy(), candidates["entity_id"].to_numpy()
    )

    assert list(official.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert official["source1_entity_id"].tolist() == s1["entity_id"].tolist()

    valid = set(candidates["entity_id"])
    for cell in official["candidate_entity_ids"]:
        ids = cell.split(",")
        assert len(ids) == 5
        assert len(set(ids)) == len(ids)
        assert set(ids) <= valid
        assert all(x == x.strip() and x for x in ids)


def test_official_format_empty_string_when_no_candidates():
    s1_ids = np.array(["S1-0", "S1-1", "S1-2"], dtype=object)
    candidate_ids = np.array(["S2-0", "S2-1", "S3-0"], dtype=object)
    retrieval = blocker.Retrieval(
        np.array([0, 0, 2]), np.array([1, 0, 2]), np.array([0.9, 0.5, 0.3], dtype=np.float32)
    )

    official = blocker.official_frame(retrieval, s1_ids, candidate_ids)

    assert official["candidate_entity_ids"].tolist() == ["S2-1,S2-0", "", "S3-0"]


def test_official_file_roundtrip():
    import tempfile

    s1, candidates, retrieval = retrieve(5)
    s1_ids = s1["entity_id"].to_numpy()
    candidate_ids = candidates["entity_id"].to_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        official_path = blocker.save_official_candidate_pairs(
            retrieval, s1_ids, candidate_ids, out_dir
        )
        scored_path = blocker.save_candidate_pairs(retrieval, s1_ids, candidate_ids, out_dir)

        official = pd.read_csv(official_path, sep="\t", dtype=str, keep_default_na=False)
        scored = pd.read_csv(scored_path, sep="\t", dtype={"source1_entity_id": str})

    assert len(official) == len(s1) == official["source1_entity_id"].nunique()
    assert list(scored.columns) == ["source1_entity_id", "candidate_entity_id", "score"]
    assert len(scored) == 5 * len(s1)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {exc!r}")
    raise SystemExit(1 if failed else 0)
