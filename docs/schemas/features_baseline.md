# Feature baseline v1 — Sujeet to Shreyashi

This is the working text-feature baseline. Retrieval scores/ranks/margins are pending
Yash's score contract. Advanced features are intentionally deferred until model error analysis.
All feature computation is offline and uses the provided files only.

## Run on a laptop

From the repository root in your IDE terminal (Python 3.12 tested):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-features.txt
.venv\Scripts\python.exe src/entity_resolution/normalization/normalize.py --source mock
.venv\Scripts\python.exe scripts/run_feature_demo.py
.venv\Scripts\python.exe -m pytest -q
```

The mock path works without real-data configuration. A warning about missing
`paths.local.yaml` is expected; `data/mock` is the fallback. For real data follow
the README path setup. The demo deliberately uses only normalized_mock train files.

The output folder is `artifacts/feature_baseline_demo/` (gitignored):

- `extractor.joblib`: training-fitted vectorizers and IDF weights.
- `train_features.parquet`, `validation_features.parquet`: IDs plus the 12 columns below.
- `train_fixture_pairs_and_labels.parquet`, `validation_fixture_pairs_and_labels.parquet`:
  IDs and labels, stored separately from features.
- `run_summary.json`: input hashes, split counts, null counts, runtime and feature schema.

For a repeat run choose a fresh `--output artifacts/feature_demo_02`. Existing output
directories are refused to avoid overwriting a run. A failed run may leave incomplete
files: only a successfully written `run_summary.json` signals completion.

## Fixed input/output contract

`join_pairs(pairs, records)` accepts one row per pair with string columns
`source1_entity_id`, `candidate_entity_id`. Normalized records must have unique
`entity_id` plus `name_norm`, `address_norm`; other columns are ignored.
No normalization or raw placeholder replacement is repeated here: the normalized
name `nan` can be a legitimate business name. Missing fields must be true nulls;
empty strings are also treated as missing defensively.

The returned joined frame can be passed to `extractor.transform`. Output preserves
pair order and contains both ID columns, followed by exactly these float32 columns:

| Column | Definition / rationale |
|---|---|
| name_jaro_winkler | Character similarity, including common-prefix evidence |
| name_token_sort | RapidFuzz token-sort ratio / 100; tolerates word reordering |
| name_tfidf_cosine | Word-unigram TF-IDF cosine; emphasizes uncommon words |
| name_idf_overlap | IDF-weighted token-set Jaccard; weighted intersection / union |
| address_jaro_winkler | Character similarity of complete normalized addresses |
| address_token_sort | Address token-sort ratio / 100 |
| address_tfidf_cosine | Separately fitted address word TF-IDF cosine |
| address_idf_overlap | Address IDF-weighted token-set Jaccard |
| source1_name_missing | 1 when the reference name is missing, otherwise 0 |
| source1_address_missing | 1 when the reference address is missing, otherwise 0 |
| candidate_name_missing | 1 when the candidate name is missing, otherwise 0 |
| candidate_address_missing | 1 when the candidate address is missing, otherwise 0 |

Similarity ranges are [0, 1]. If either field is missing, all four similarities for
that field are NaN; two missing values never create positive similarity evidence.
Flags are always 0 or 1. No ID, label, or country is fed into these numerical features.
Keep NaNs for LightGBM; any other model's imputer must be fitted on training data only.

TF-IDF uses whitespace tokenization (including one-character tokens), no additional
lowercasing, L2 normalization, sklearn's smoothed IDF, at most 100,000 words per field.
IDF-overlap uses the same fitted vocabulary/weights. Unknown or pruned tokens have
zero weight. A nonmissing pair with zero known-token denominator/cosine norms returns
0; an entirely missing training field has no vectorizer and also returns 0 for future
nonmissing comparisons. These cases remain distinct from missing input (NaN).

## Split and leakage policy

The demo intersects the committed frozen train/validation reference lists with the
mock truth. It checks disjointness and complete reference coverage. Fit includes only
mock training references and their known matched targets, excluding validation-owned
targets and unassigned distractors. Validation transformation never fits or changes
the artifact. Use the same saved artifact for subsequent inference; changing the fit
corpus requires retraining the matcher. For a US-only model, fit on that training fold.

The demo fixture includes all known positives and one deterministic negative per
reference, including true singletons. **This is not candidate generation, representative
model training data, or a valid blocking/model evaluation.** Its easy negative distribution
and ground-truth-based construction are deliberately for integration testing only.
The true validation matrix must later use Yash's independently generated candidates.

## API integration and scale

```python
from entity_resolution.features.baseline import FeatureExtractor, join_pairs, FEATURE_COLUMNS

extractor = FeatureExtractor().fit(training_only_normalized_sample)
extractor.save("artifacts/features.joblib")
features = extractor.transform(join_pairs(pair_batch, normalized_records_for_batch))
X = features[FEATURE_COLUMNS]  # IDs stay outside the model
```

Add `src` to your Python path for API usage, as the provided script does. Load only
trusted local joblib files. `fit` holds its selected corpus in memory; use a bounded,
documented training sample for full data. `transform` handles one batch at a time and
uses sparse rowwise cosine, never a pair-by-pair dense similarity matrix.

The demo loads all **mock** normalized rows in memory; it is not a full-scale source
lookup implementation. For millions of records, the integration must provide batch
record lookups from an on-disk index or equivalent. Do not point the demo at full data.
Pair/record duplicate and missing-ID checks fail loudly. Pair uniqueness is checked
within a batch; the full candidate producer must guarantee global uniqueness.

## Pending handoff from Yash

Provide actual training and validation candidates, plus per-pair blocking_score with
its meaning, direction (higher/lower is better), and whether scores are comparable
between S2 and S3. Agree on margin definition and tie handling before adding these
columns in a versioned schema update. The official list-format candidate TSV alone
cannot supply scores. Do not invent scores or silently fill them with zeros.

## Errors and checks

- Missing normalized Parquet: run the mock normalizer first.
- Missing package: install `requirements-features.txt` with the same Python used to run.
- Unknown/duplicate IDs: correct upstream pair/source data; never silently drop pairs.
- Memory pressure: reduce batch size; full-scale source indexing is still an integration task.
- NaN similarities with a missingness flag of 1: expected, do not replace with match scores.
- Commit scripts/tests/docs only; generated data, features and fitted artifacts are ignored.

The branch includes a small pandas compatibility fix to the upstream normalizer:
replace unsupported `.str.isascii()` with an ASCII regex check and use nullable
`string` dtype so nulls survive under pandas 2.2. A regression test covers both.
