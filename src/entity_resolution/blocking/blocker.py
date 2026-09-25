from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Paths
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
NORMALIZED_DIR = REPO_ROOT / "data" / "processed" / "normalized_mock"
MOCK_DIR = REPO_ROOT / "data" / "mock"
OUTPUT_DIR = REPO_ROOT / "output"

TOP_K = 50


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():
    s1 = pd.read_parquet(NORMALIZED_DIR / "train_source1.parquet")
    s2 = pd.read_parquet(NORMALIZED_DIR / "train_source2.parquet")
    s3 = pd.read_parquet(NORMALIZED_DIR / "train_source3.parquet")

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

def build_token_index(candidates):
    token_index = defaultdict(set)

    for idx, row in candidates.iterrows():

        name = row["name_norm"]
        address = row["address_norm"]

        name = "" if pd.isna(name) else str(name)
        address = "" if pd.isna(address) else str(address)

        tokens = set(
            (name + " " + address).split()
        )

        for token in tokens:
            if token:
                token_index[token].add(idx)

    return token_index


def get_token_candidates(row, token_index):

    name = row["name_norm"]
    address = row["address_norm"]

    name = "" if pd.isna(name) else str(name)
    address = "" if pd.isna(address) else str(address)

    tokens = set(
        (name + " " + address).split()
    )

    candidate_indexes = set()

    for token in tokens:
        candidate_indexes.update(
            token_index.get(token, set())
        )

    return candidate_indexes


# --------------------------------------------------
# Retrieve Top-K + create candidate pairs
# --------------------------------------------------

def retrieve_top_k(
    s1,
    candidates,
    s1_name_matrix,
    candidate_name_matrix,
    s1_address_matrix,
    candidate_address_matrix,
    token_index,
    top_k=TOP_K
):

    retrieved = {}

    candidate_pair_rows = []

    for i in range(len(s1)):

        # Name similarity
        name_scores = cosine_similarity(
            s1_name_matrix[i],
            candidate_name_matrix
        ).flatten()

        # Address similarity
        address_scores = cosine_similarity(
            s1_address_matrix[i],
            candidate_address_matrix
        ).flatten()

        # Keep strongest TF-IDF signal
        combined_scores = np.maximum(
            name_scores,
            address_scores
        )

        # Token inverted index
        token_candidates = get_token_candidates(
            s1.iloc[i],
            token_index
        )

        # Token-match bonus
        if token_candidates:

            token_indexes = np.fromiter(
                token_candidates,
                dtype=int
            )

            combined_scores[token_indexes] += 0.05

        # Get Top-K candidates
        top_indices = combined_scores.argsort()[::-1][:top_k]

        s1_id = s1.iloc[i]["entity_id"]

        candidate_ids = (
            candidates.iloc[top_indices]["entity_id"].tolist()
        )

        retrieved[s1_id] = set(candidate_ids)

        # Save every candidate pair
        for candidate_idx in top_indices:

            candidate_pair_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidates.iloc[
                        candidate_idx
                    ]["entity_id"],
                    "score": combined_scores[candidate_idx]
                }
            )

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(s1)} S1 records"
            )

    candidate_pairs = pd.DataFrame(
        candidate_pair_rows
    )

    return retrieved, candidate_pairs


# --------------------------------------------------
# Blocking recall
# --------------------------------------------------

def calculate_blocking_recall(retrieved):

    ground_truth = pd.read_csv(
        MOCK_DIR / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    total_true_matches = 0
    retrieved_true_matches = 0

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]

        true_ids = {
            x.strip()
            for x in row["matched_entity_ids"].split(",")
            if x.strip()
        }

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

    return (
        recall,
        retrieved_true_matches,
        total_true_matches
    )


# --------------------------------------------------
# Save candidate_pairs.tsv
# --------------------------------------------------

def save_candidate_pairs(candidate_pairs):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR / "candidate_pairs.tsv"
    )

    candidate_pairs.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    return output_path


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading normalized data...")

    s1, s2, s3 = load_data()

    candidates = build_candidate_pool(
        s2,
        s3
    )

    print("Source 1:", len(s1))
    print("Source 2:", len(s2))
    print("Source 3:", len(s3))
    print("Candidate pool:", len(candidates))

    # --------------------------------------------------
    # Name TF-IDF
    # --------------------------------------------------

    print()
    print("Building NAME TF-IDF index...")

    (
        s1_name_matrix,
        candidate_name_matrix
    ) = build_tfidf(
        s1,
        candidates,
        "name_norm"
    )

    print("NAME TF-IDF index ready.")

    # --------------------------------------------------
    # Address TF-IDF
    # --------------------------------------------------

    print()
    print("Building ADDRESS TF-IDF index...")

    (
        s1_address_matrix,
        candidate_address_matrix
    ) = build_tfidf(
        s1,
        candidates,
        "address_norm"
    )

    print("ADDRESS TF-IDF index ready.")

    # --------------------------------------------------
    # Token inverted index
    # --------------------------------------------------

    print()
    print("Building TOKEN inverted index...")

    token_index = build_token_index(
        candidates
    )

    print("TOKEN inverted index ready.")
    print("Unique tokens:", len(token_index))

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    print()
    print("Retrieving Top-50 candidates...")

    retrieved, candidate_pairs = retrieve_top_k(
        s1,
        candidates,
        s1_name_matrix,
        candidate_name_matrix,
        s1_address_matrix,
        candidate_address_matrix,
        token_index,
        top_k=TOP_K
    )

    # --------------------------------------------------
    # Save candidate pairs
    # --------------------------------------------------

    print()
    print("Saving candidate_pairs.tsv...")

    output_path = save_candidate_pairs(
        candidate_pairs
    )

    print("Saved:", output_path)
    print("Candidate pair rows:", len(candidate_pairs))

    # --------------------------------------------------
    # Recall
    # --------------------------------------------------

    print()
    print("Calculating blocking recall...")

    recall, found, total = calculate_blocking_recall(
        retrieved
    )

    missed = total - found

    print()
    print("====================================")
    print("FINAL BLOCKER: TF-IDF + TOKEN INDEX")
    print("====================================")

    print("True matches retrieved:", found)
    print("Total true matches:", total)
    print("True matches missed:", missed)
    print(f"Blocking Recall: {recall:.2%}")

    print()
    print("candidate_pairs.tsv:", output_path)