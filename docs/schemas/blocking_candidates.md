# Blocking candidate pairs

Contract for the output of `src/entity_resolution/blocking/blocker.py` (Stage 3). Consumed by the
features, matcher and submission stages.

Run: `python src/entity_resolution/blocking/blocker.py --source mock|real [--splits train test]`.
Needs the normalized parquets from `normalization/normalize.py --source ...` first.

## What it does

Per split, S1 is matched against the pooled S2 + S3 rows **of that split only** (train S1 against
train S2/S3, test S1 against test S2/S3; the pools are never mixed).

1. Char 2-4-gram TF-IDF on `name_norm` and on `address_norm`, one vectorizer per field fit on the
   split's candidate pool (S2 and S3 together) and used to transform S1.
2. Each candidate TF-IDF matrix is projected to `--svd-dim` dense dimensions (TruncatedSVD, default
   256) and L2-normalized, then indexed with FAISS: exact inner-product search on GPU when FAISS
   sees a CUDA GPU, otherwise CPU HNSW (`--index-type auto|hnsw|flat`).
3. The name index and the address index each return `--ann-k` hits per S1 row (default 2 x top-k).
   The union is the shortlist.
4. The shortlist is rescored with the **exact sparse TF-IDF cosine** (the SVD only proposes
   candidates, it does not change scores), the token bonus is added, and the Top-K
   (`--top-k`, default 50) are kept.

The token bonus uses the token inverted index over `name_norm + address_norm` tokens of the
candidate pool.

## Score semantics

```
score = max(name_cosine, address_cosine) + 0.05 * (S1 row shares at least one token with the candidate)
```

- **Higher score = better match.**
- Scores are computed identically for S2 and S3 candidates (same shared vectorizer over the pooled
  S2 + S3 rows), so they are **directly comparable across sources**.
- Scores are **not bounded to [0, 1]**: each cosine is in [0, 1] and the bonus adds up to 0.05, so
  the maximum is 1.05.
- The score collapses two fields into one number (it is whichever of name or address agrees
  better), so it is a ranking signal, not a probability, and it cannot tell you which field matched.
  Use the per-field features from the features stage for that.
- IDF is fit per split. Scores are comparable within a split, but a train score of 0.8 and a test
  score of 0.8 are not numerically identical measurements.
- A missing name or address is treated as empty text (cosine 0 for that field).

## Outputs

Written under `output/<source>/<split>/` (`output/` is gitignored).

| File | Format | Use |
|---|---|---|
| `candidate_pairs_scored.tsv` | Long: one row per pair, columns `source1_entity_id`, `candidate_entity_id`, `score`. Sorted by S1 row, then descending score. | Features and matcher stages. |
| `candidate_pairs.tsv` | Official submission format: one row per S1 entity, columns `source1_entity_id`, `candidate_entity_ids` (comma-separated, best first; empty string when there are no candidates). | Submission. `test/candidate_pairs.tsv` is the one to submit. |

Both files cover every S1 entity of the split, at most Top-K candidates each.

## Blocking recall

Printed for the train split only (there is no test ground truth). The headline **Blocking Recall
(validation)** counts only references listed in `src/entity_resolution/data/val_reference_ids.txt`
(`--split-dir` points at another folder holding `val_reference_ids.txt` and
`train_reference_ids.txt`). A train-set recall line is printed separately as information.
Recall = true matches found in the candidates / true matches, pooled over the evaluated references.

Measured on the mock (2,000 S1 references, 13,976 candidates): the exact brute-force blocker gave
99.53% over all references (99.73% on the 414 validation references); the ANN version gives 99.24%
over all (99.45% validation) with the default `--svd-dim 256`, `--ann-k 100`. Lower `--svd-dim`
or `--ann-k` is faster but loses recall (128 dims / ann-k 50 gave 98.0%). The shortlist is
approximate, but scores of the pairs it returns equal the brute-force scores.
