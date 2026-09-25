"""Stage 3: candidate retrieval (blocking).

For every Source-1 reference, retrieve the Top-K most similar rows from the
pooled Source-2 + Source-3 candidates, using char n-gram TF-IDF on the
normalized name and address plus a token-overlap bonus.

Pipeline per split (train and test are run separately, never pooled):

    1. Fit one TF-IDF vectorizer per field (name, address) on the split's
       candidate pool (S2 + S3 together) and transform S1 with it.
    2. Reduce each TF-IDF matrix with TruncatedSVD to a dense, L2-normalized
       vector and index the candidates with FAISS (GPU exact search when a CUDA
       FAISS build is present, otherwise CPU HNSW).
    3. Query the name index and the address index for each S1 row and take the
       union of the hits as the shortlist.
    4. Rescore the shortlist with the exact sparse TF-IDF cosine (so the score
       is not distorted by the SVD), add the token bonus, keep the Top-K.

Score semantics (the contract for the features/matcher stages):

    score = max(name_cosine, address_cosine) + 0.05 * shares_any_token

  * Higher score = better match.
  * S2 and S3 candidates are scored identically, with the same shared
    vectorizer, so scores are directly comparable across sources.
  * Scores are NOT bounded to [0, 1]: cosine is in [0, 1] and the bonus adds up
    to 0.05, so the maximum is 1.05.
  * The score mixes two fields into one number (whichever of name or address
    agrees better) and is a ranking signal, not a probability.
  * IDF is fit per split, so scores are comparable within a split, not
    numerically identical between train and test.

Run:  python src/entity_resolution/blocking/blocker.py --source mock
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from entity_resolution.config import REPO_ROOT, get_dataset_dir, get_mock_dir  # noqa: E402

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TOP_K = 50
TOKEN_BONUS = 0.05

# ANN stage: each field's index returns ann_k hits per S1 row (default
# ANN_K_FACTOR * top_k) and the shortlist is rescored exactly. SVD_DIM and
# ANN_K_FACTOR trade recall against time and memory (measured on the mock:
# 128 dims / factor 1 -> 98.0% recall, 256 / 2 -> 99.2%, exact brute force 99.5%).
SVD_DIM = 256
ANN_K_FACTOR = 2
SVD_FIT_ROWS = 200_000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
BATCH_SIZE = 2_000

SPLIT_DIR = Path(__file__).resolve().parents[1] / "data"
SPLITS = ("train", "test")


# --------------------------------------------------
# Load data
# --------------------------------------------------

def normalized_dir(source):
    return REPO_ROOT / "data" / "processed" / (
        "normalized_mock" if source == "mock" else "normalized"
    )


def output_dir(source, split):
    return REPO_ROOT / "output" / source / split


def ground_truth_path(source):
    if source == "mock":
        return get_mock_dir() / "train_ground_truth.tsv"
    return get_dataset_dir() / "train" / "train_ground_truth.tsv"


def load_split(source, split):
    """S1 and the pooled S2+S3 candidates for one split; train and test never mix."""
    base = normalized_dir(source)
    s1 = pd.read_parquet(base / f"{split}_source1.parquet")
    s2 = pd.read_parquet(base / f"{split}_source2.parquet")
    s3 = pd.read_parquet(base / f"{split}_source3.parquet")
    return s1, s2, s3


def build_candidate_pool(s2, s3):
    return pd.concat(
        [s2, s3],
        ignore_index=True
    )


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

def build_tfidf(s1, candidates, column):
    s1_text = s1[column].fillna("")
    candidate_text = candidates[column].fillna("")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4)
    )

    candidate_matrix = vectorizer.fit_transform(
        candidate_text
    )

    s1_matrix = vectorizer.transform(
        s1_text
    )

    return s1_matrix, candidate_matrix


# --------------------------------------------------
# Token inverted index
# --------------------------------------------------

def get_row_tokens(name, address):
    name = "" if pd.isna(name) else str(name)
    address = "" if pd.isna(address) else str(address)

    return set(
        (name + " " + address).split()
    )


def build_token_index(candidates):
    token_index = defaultdict(set)

    rows = zip(
        candidates.index,
        candidates["name_norm"],
        candidates["address_norm"]
    )

    for idx, name, address in rows:
        for token in get_row_tokens(name, address):
            if token:
                token_index[token].add(idx)

    return token_index


def shares_token(row_tokens, candidate_indexes, token_index):
    """Bool per candidate: does it share at least one token with the S1 row?

    Same relation as "candidate is in the union of token_index[t]", checked
    per shortlisted candidate instead of materializing that union, which for
    common tokens is a large share of the whole pool.
    """
    postings = [
        token_index[token]
        for token in row_tokens
        if token in token_index
    ]

    return np.fromiter(
        (
            any(idx in posting for posting in postings)
            for idx in candidate_indexes
        ),
        dtype=bool,
        count=len(candidate_indexes)
    )


# --------------------------------------------------
# ANN index (FAISS)
# --------------------------------------------------

def gpu_available():
    return hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0


class AnnField:
    """SVD projection + FAISS index over one field's candidate TF-IDF matrix.

    FAISS only searches dense vectors, so the sparse TF-IDF rows are projected
    to `dim` dimensions and L2-normalized; inner product then approximates the
    TF-IDF cosine. The index only proposes a shortlist. Exact scores are
    recomputed on the sparse matrices in retrieve_top_k.
    """

    def __init__(self, candidate_matrix, dim=SVD_DIM, index_type="auto",
                 fit_rows=SVD_FIT_ROWS, seed=0):
        n_rows, n_features = candidate_matrix.shape
        components = max(1, min(dim, n_features - 1, n_rows - 1))

        rng = np.random.default_rng(seed)
        fit_idx = (
            np.sort(rng.choice(n_rows, fit_rows, replace=False))
            if n_rows > fit_rows
            else slice(None)
        )

        self.svd = TruncatedSVD(
            n_components=components,
            algorithm="randomized",
            random_state=seed
        ).fit(candidate_matrix[fit_idx].astype(np.float32))

        vectors = self.project(candidate_matrix)

        if index_type == "auto":
            index_type = "flat" if gpu_available() else "hnsw"

        self.index_type = index_type
        self.on_gpu = False

        if index_type == "flat":
            index = faiss.IndexFlatIP(components)
            if gpu_available():
                index = faiss.index_cpu_to_all_gpus(index)
                self.on_gpu = True
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(
                components, HNSW_M, faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        else:
            raise ValueError(f"unknown index_type {index_type!r}")

        index.add(vectors)
        self.index = index

    def project(self, matrix, chunk=100_000):
        out = np.empty((matrix.shape[0], self.svd.n_components), dtype=np.float32)

        for start in range(0, matrix.shape[0], chunk):
            out[start:start + chunk] = self.svd.transform(
                matrix[start:start + chunk].astype(np.float32)
            )

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)

        return np.ascontiguousarray(out)

    def search(self, matrix, k):
        """Candidate row positions, shape (rows, k); -1 where fewer than k exist."""
        if self.index_type == "hnsw":
            self.index.hnsw.efSearch = max(64, 2 * k)

        _, hits = self.index.search(self.project(matrix), k)

        return hits


# --------------------------------------------------
# Retrieve Top-K + create candidate pairs
# --------------------------------------------------

class Retrieval(NamedTuple):
    """Candidate pairs as parallel arrays, sorted by S1 row then descending score.

    s1_idx / cand_idx are row positions into S1 and the candidate pool.
    """
    s1_idx: np.ndarray
    cand_idx: np.ndarray
    score: np.ndarray


def pair_cosine(s1_matrix, candidate_matrix, s1_rows, candidate_rows):
    """Exact TF-IDF cosine for each (s1_row, candidate_row) pair (rows are L2-normalized)."""
    return np.asarray(
        s1_matrix[s1_rows].multiply(candidate_matrix[candidate_rows]).sum(axis=1)
    ).ravel()


def retrieve_top_k(
    s1,
    candidates,
    s1_name_matrix,
    candidate_name_matrix,
    s1_address_matrix,
    candidate_address_matrix,
    name_ann,
    address_ann,
    token_index,
    top_k=TOP_K,
    ann_k=None,
    batch_size=BATCH_SIZE
):
    ann_k = max(ann_k or ANN_K_FACTOR * top_k, top_k)
    n_candidates = len(candidates)

    s1_idx_parts, cand_idx_parts, score_parts = [], [], []

    s1_names = s1["name_norm"].to_numpy()
    s1_addresses = s1["address_norm"].to_numpy()

    for start in range(0, len(s1), batch_size):
        stop = min(start + batch_size, len(s1))

        name_batch = s1_name_matrix[start:stop]
        address_batch = s1_address_matrix[start:stop]

        # Shortlist = union of name hits and address hits. Every row of the
        # exact Top-K by max(name, address) is in the name Top-K or the
        # address Top-K, so nothing is lost by querying the fields separately.
        hits = np.concatenate(
            [
                name_ann.search(name_batch, ann_k),
                address_ann.search(address_batch, ann_k)
            ],
            axis=1
        )

        local_rows = np.repeat(np.arange(stop - start), hits.shape[1])
        flat_hits = hits.ravel()
        valid = flat_hits >= 0

        keys = np.unique(
            local_rows[valid].astype(np.int64) * n_candidates + flat_hits[valid]
        )
        rows = keys // n_candidates
        cols = keys % n_candidates

        # Exact scores on the shortlist
        scores = np.maximum(
            pair_cosine(name_batch, candidate_name_matrix, rows, cols),
            pair_cosine(address_batch, candidate_address_matrix, rows, cols)
        )

        # Token-match bonus
        bonus = np.zeros(len(rows), dtype=bool)
        bounds = np.searchsorted(rows, np.arange(stop - start + 1))

        for i in range(stop - start):
            lo, hi = bounds[i], bounds[i + 1]
            row_tokens = get_row_tokens(
                s1_names[start + i],
                s1_addresses[start + i]
            )
            if hi > lo and row_tokens:
                bonus[lo:hi] = shares_token(
                    row_tokens,
                    cols[lo:hi],
                    token_index
                )

        scores = scores + TOKEN_BONUS * bonus

        # Top-K per row (ties broken by candidate position for determinism)
        order = np.lexsort((cols, -scores, rows))
        rows, cols, scores = rows[order], cols[order], scores[order]

        rank = np.arange(len(rows)) - np.searchsorted(rows, rows, side="left")
        keep = rank < top_k

        s1_idx_parts.append(rows[keep] + start)
        cand_idx_parts.append(cols[keep])
        score_parts.append(scores[keep].astype(np.float32))

        if (stop // batch_size) % 5 == 0 or stop == len(s1):
            print(f"Processed {stop}/{len(s1)} S1 records")

    return Retrieval(
        np.concatenate(s1_idx_parts),
        np.concatenate(cand_idx_parts),
        np.concatenate(score_parts)
    )


def retrieve_candidates(
    s1,
    candidates,
    top_k=TOP_K,
    ann_k=None,
    svd_dim=SVD_DIM,
    index_type="auto",
    batch_size=BATCH_SIZE,
    verbose=False
):
    """Full retrieval for one split: TF-IDF, ANN indexes, token index, Top-K."""
    def log(message, t0):
        if verbose:
            print(f"{message} ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    s1_name, cand_name = build_tfidf(s1, candidates, "name_norm")
    s1_addr, cand_addr = build_tfidf(s1, candidates, "address_norm")
    log("TF-IDF (name + address) ready", t0)

    t0 = time.perf_counter()
    name_ann = AnnField(cand_name, svd_dim, index_type)
    address_ann = AnnField(cand_addr, svd_dim, index_type)
    log(
        f"ANN indexes ready: {name_ann.index_type}"
        f"{' (GPU)' if name_ann.on_gpu else ''}, "
        f"{name_ann.svd.n_components} dims",
        t0
    )

    t0 = time.perf_counter()
    token_index = build_token_index(candidates)
    log(f"Token index ready: {len(token_index)} unique tokens", t0)

    t0 = time.perf_counter()
    retrieval = retrieve_top_k(
        s1, candidates,
        s1_name, cand_name, s1_addr, cand_addr,
        name_ann, address_ann, token_index,
        top_k=top_k, ann_k=ann_k, batch_size=batch_size
    )
    log(f"Retrieved Top-{top_k}", t0)

    return retrieval


def retrieved_sets(retrieval, s1_ids, candidate_ids):
    """dict[source1_entity_id -> set of retrieved candidate IDs]."""
    retrieved = {s1_id: set() for s1_id in s1_ids}

    s1_col = np.asarray(s1_ids, dtype=object)[retrieval.s1_idx]
    cand_col = np.asarray(candidate_ids, dtype=object)[retrieval.cand_idx]

    for s1_id, cand_id in zip(s1_col, cand_col):
        retrieved[s1_id].add(cand_id)

    return retrieved


# --------------------------------------------------
# Blocking recall
# --------------------------------------------------

class RecallResult(NamedTuple):
    recall: float
    found: int
    total: int
    n_refs: int


def load_ground_truth(path):
    ground_truth = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE
    )

    truth = defaultdict(set)

    for s1_id, matched in zip(
        ground_truth["source1_entity_id"],
        ground_truth["matched_entity_ids"]
    ):
        truth[s1_id].update(
            x.strip() for x in matched.split(",") if x.strip()
        )

    return dict(truth)


def load_id_set(path):
    return set(
        Path(path).read_text(encoding="utf-8").split()
    )


def calculate_blocking_recall(retrieved, ground_truth, reference_ids):
    """Recall of retrieved candidates over the true matches of `reference_ids` only.

    `reference_ids` scopes the measurement to one split (validation for the
    headline number); ground-truth rows for any other reference are ignored.
    """
    total_true_matches = 0
    retrieved_true_matches = 0
    n_refs = 0

    for s1_id, true_ids in ground_truth.items():

        if s1_id not in reference_ids:
            continue

        n_refs += 1
        total_true_matches += len(true_ids)

        found_ids = retrieved.get(
            s1_id,
            set()
        )

        retrieved_true_matches += len(
            true_ids.intersection(found_ids)
        )

    recall = (
        retrieved_true_matches / total_true_matches
        if total_true_matches
        else 0
    )

    return RecallResult(
        recall,
        retrieved_true_matches,
        total_true_matches,
        n_refs
    )


# --------------------------------------------------
# Save candidate pairs
# --------------------------------------------------

SCORED_NAME = "candidate_pairs_scored.tsv"
OFFICIAL_NAME = "candidate_pairs.tsv"


def save_candidate_pairs(retrieval, s1_ids, candidate_ids, out_dir, chunk=5_000_000):
    """Long format (source1_entity_id, candidate_entity_id, score): one row per pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SCORED_NAME

    s1_ids = np.asarray(s1_ids, dtype=object)
    candidate_ids = np.asarray(candidate_ids, dtype=object)

    for start in range(0, len(retrieval.score), chunk):
        part = slice(start, start + chunk)

        pd.DataFrame(
            {
                "source1_entity_id": s1_ids[retrieval.s1_idx[part]],
                "candidate_entity_id": candidate_ids[retrieval.cand_idx[part]],
                "score": retrieval.score[part]
            }
        ).to_csv(
            path,
            sep="\t",
            index=False,
            float_format="%.6f",
            mode="w" if start == 0 else "a",
            header=start == 0
        )

    if len(retrieval.score) == 0:
        pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "score"]
        ).to_csv(path, sep="\t", index=False)

    return path


def official_frame(retrieval, s1_ids, candidate_ids):
    """Submission format: one row per S1 entity, candidates comma-joined by descending score.

    An entity with no candidates gets an empty string.
    """
    n_s1 = len(s1_ids)
    candidate_ids = np.asarray(candidate_ids, dtype=object)

    counts = np.bincount(retrieval.s1_idx, minlength=n_s1)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    ordered_ids = candidate_ids[retrieval.cand_idx]

    return pd.DataFrame(
        {
            "source1_entity_id": list(s1_ids),
            "candidate_entity_ids": [
                ",".join(ordered_ids[bounds[i]:bounds[i + 1]])
                for i in range(n_s1)
            ]
        }
    )


def save_official_candidate_pairs(retrieval, s1_ids, candidate_ids, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / OFFICIAL_NAME

    official_frame(retrieval, s1_ids, candidate_ids).to_csv(
        path,
        sep="\t",
        index=False
    )

    return path


# --------------------------------------------------
# Main
# --------------------------------------------------

def run_split(args, split):
    print()
    print("====================================")
    print(f"SPLIT: {split.upper()}")
    print("====================================")

    t_split = time.perf_counter()

    print("Loading normalized data...")

    s1, s2, s3 = load_split(args.source, split)

    candidates = build_candidate_pool(s2, s3)

    assert s1["entity_id"].is_unique, "duplicate S1 entity_id"

    print("Source 1:", len(s1))
    print("Source 2:", len(s2))
    print("Source 3:", len(s3))
    print("Candidate pool:", len(candidates))

    print()

    retrieval = retrieve_candidates(
        s1,
        candidates,
        top_k=args.top_k,
        ann_k=args.ann_k,
        svd_dim=args.svd_dim,
        index_type=args.index_type,
        batch_size=args.batch_size,
        verbose=True
    )

    s1_ids = s1["entity_id"].to_numpy()
    candidate_ids = candidates["entity_id"].to_numpy()

    out_dir = output_dir(args.source, split)

    scored_path = save_candidate_pairs(
        retrieval, s1_ids, candidate_ids, out_dir
    )
    official_path = save_official_candidate_pairs(
        retrieval, s1_ids, candidate_ids, out_dir
    )

    print()
    print("Candidate pair rows:", len(retrieval.score))
    print("Official rows (one per S1):", len(s1_ids))
    print("Saved:", scored_path)
    print("Saved:", official_path)

    if split == "train":

        retrieved = retrieved_sets(retrieval, s1_ids, candidate_ids)

        ground_truth = load_ground_truth(ground_truth_path(args.source))

        val_ids = load_id_set(args.split_dir / "val_reference_ids.txt")
        train_ids = load_id_set(args.split_dir / "train_reference_ids.txt")

        val = calculate_blocking_recall(retrieved, ground_truth, val_ids)
        train = calculate_blocking_recall(retrieved, ground_truth, train_ids)

        print()
        print("FINAL BLOCKER: TF-IDF + FAISS ANN + TOKEN INDEX")
        print("------------------------------------")
        print(f"Validation references evaluated: {val.n_refs}")
        print("True matches retrieved:", val.found)
        print("True matches missed:", val.total - val.found)
        print("Total true matches:", val.total)
        print(f"Blocking Recall (validation): {val.recall:.2%}")
        print(
            f"(train-set recall, informational: {train.recall:.2%} "
            f"over {train.n_refs} references, {train.found}/{train.total})"
        )

    else:
        print()
        print("Blocking recall: not computed (no test ground truth)")

    print(f"{split} split time: {time.perf_counter() - t_split:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["mock", "real"], default="mock")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--ann-k", type=int, default=None,
                        help=f"hits per field index before rescoring (default: {ANN_K_FACTOR} x top-k)")
    parser.add_argument("--svd-dim", type=int, default=SVD_DIM)
    parser.add_argument("--index-type", choices=["auto", "hnsw", "flat"], default="auto",
                        help="auto = exact GPU flat index if FAISS sees a GPU, else CPU HNSW")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR,
                        help="folder with val_reference_ids.txt and train_reference_ids.txt")
    args = parser.parse_args()

    print(
        f"FAISS {faiss.__version__}, "
        f"GPUs visible to FAISS: {faiss.get_num_gpus() if hasattr(faiss, 'get_num_gpus') else 0}"
    )

    t0 = time.perf_counter()

    for split in args.splits:
        run_split(args, split)

    print()
    print(f"Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
