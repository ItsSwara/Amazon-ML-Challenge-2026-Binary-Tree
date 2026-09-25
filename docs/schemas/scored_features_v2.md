# Scored feature handoff v2 — Sujeet to Shreyashi

This runner combines Yash's Top-50 blocking candidates with the existing text features
on the linked **training mock**. It does not generate candidates: they must be supplied
as a scored TSV produced by the blocking stage. No positive candidates are injected
from truth. It independently evaluates recall on the frozen train/validation reference
sets. It is not a full-dataset benchmark.

## Run locally

From the repository root in PowerShell, using Python 3.12 and the pinned feature environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-features.txt
.venv\Scripts\python.exe src/entity_resolution/normalization/normalize.py --source mock
.venv\Scripts\python.exe scripts/run_blocking_features.py --pairs path/to/scored_candidates.tsv --output artifacts/scored_mock_v2
.venv\Scripts\python.exe -m pytest -q
```

If your environment is outside the repository, substitute its Python executable.
Use a fresh output folder for every run; overwriting is refused. `--pairs` is required:
run the blocking stage first and pass its **pair-level scored file for this exact mock**.
The runner never imports or calls `blocking/blocker.py`.

Input TSV columns: `source1_entity_id`, `candidate_entity_id`, `score`. Higher score
is better. Scores are checked against Yash's [0, 1.05] range, with 1e-6 roundoff
tolerance. Duplicate pairs, unknown IDs, invalid prefixes, missing/nonfinite scores,
and more than 50 candidates for a reference fail loudly. No candidates are silently dropped.
References with zero candidates remain in truth and list-format exports with empty lists.

## Feature contract: 15 columns

Use `SCORED_FEATURE_COLUMNS` from `entity_resolution.features.scored`, or read the
ordered `feature_columns` array from the generated `feature_schema.json`.
The original `FEATURE_COLUMNS` constant remains the 12-column v1 contract for old users.

V2 contains the original 12 documented in [baseline features](features_baseline.md),
followed by these three float32 columns:

| Column | Meaning |
|---|---|
| `blocking_score` | `max(name char-TFIDF cosine, address char-TFIDF cosine) + 0.05` if any name/address token overlaps |
| `candidate_rank` | One-based descending score rank across combined S2/S3 candidates; tied scores use ascending lexicographic candidate ID |
| `blocking_top1_top2_gap` | Highest minus second-highest score within the S1 candidate group; repeated on every candidate row for that reference |

Rank and gap are computed on complete groups **before** feature batching or split
export. Gap is 0 for a top-score tie, NaN for only one candidate. It is a group-level
confidence signal, not each candidate's margin over a competitor. This explicit
definition replaces the ambiguous phrase "margin to runner-up". Rankings are computed
from original score precision, before float32 export. Yash's selected Top-50 membership
is unchanged, including his selection of candidates tied at the cutoff.

The character vectorizers for blocking are shared across the combined S2/S3 pool,
so this implementation's scores use the same formula for both sources. They are
retrieval similarities, not calibrated match probabilities. The address and name
feature vectorizers are separate word vectorizers fitted only on the feature-training
corpus. Yash's blocking vectorizers fit on all mock target text, including validation
targets, without truth labels; the report records this transductive retrieval policy.

## Files to give Shreyashi

Give the **entire completed output folder** together with this branch's code and
`requirements-features.txt`. Keep these generated artifacts out of Git:

| File | Use |
|---|---|
| `train_features.parquet`, `validation_features.parquet` | Two ID columns + 15 ordered float32 features |
| `train_labels.parquet`, `validation_labels.parquet` | Same pair IDs + uint8 label 0/1; only retrieved pairs |
| `train_ground_truth.tsv`, `validation_ground_truth.tsv` | Complete truth for all references, including missed targets and singletons |
| `train_reference_ids.txt`, `validation_reference_ids.txt` | Frozen split intersected with the mock |
| `extractor.joblib` | The SAME training-fitted text extractor for both splits and later transforms |
| `feature_schema.json` | Version and ordered model feature columns |
| `scored_candidates.tsv`, `retrieval_features.parquet` | Original scored candidates and derived retrieval metadata |
| `train_candidate_pairs.tsv`, `validation_candidate_pairs.tsv` | One row per reference, comma-separated candidate IDs, including empty groups |
| `run_summary.json` | Completion marker, hashes, versions, counts, split recall and runtime |

The two list-format files use the official candidate schema, but contain **mock train IDs**.
They are not competition test submissions. No `matching_results.tsv` is generated.

### Load for training

```python
import json
from pathlib import Path
import pandas as pd

folder = Path('artifacts/scored_mock_v2')
schema = json.loads((folder / 'feature_schema.json').read_text())
features = pd.read_parquet(folder / 'train_features.parquet')
labels = pd.read_parquet(folder / 'train_labels.parquet')
keys = schema['id_columns']
training = features.merge(labels, on=keys, how='left', validate='one_to_one')
assert len(training) == len(features) == len(labels)
assert training['label'].notna().all()
X = training[schema['feature_columns']]
y = training['label']
```

Keep IDs/labels out of X. Keep missing similarities and singleton gaps as NaN for
LightGBM; any alternative imputer must be fitted only on training rows. Do not fit
another extractor on validation/test. On new candidates call `add_retrieval_features`
on complete groups, load the trusted local extractor with `FeatureExtractor.load`,
and use `write_features` with the matching normalized records. The extractor contains
the 12 text features; the separate scored module supplies the other three.

For evaluation, turn predictions into sets per S1 and score against **the complete
validation_ground_truth.tsv** using Swara's scorer. Pair labels alone omit true matches
lost by blocking. Ensure every reference is evaluated, including those with no candidates.
Split calibration/threshold tuning appropriately within the available training/validation
design; this handoff does not train or calibrate a matcher.

## Interpretation and limitations

The runner reports micro blocking recall (retrieved true links / all true links) and
macro recall over nonsingleton references, with singleton counts reported separately.
Neither is the final macro F0.5 score. No model performance is claimed here.
The linked mock has selected references and an easier/smaller target pool than the full
dataset. Even high mock recall does not establish full-data recall or France generalization.

This runner holds all mock normalized rows and scored pairs in RAM. Feature calculations
and Parquet writes are batched, but the full-scale record lookup is still pending. The
upstream blocker compares every reference to every target, sorts the whole target pool,
and accumulates all pairs in memory. Do not point this runner at the multi-million-row
dataset; Yash's scalable retrieval and a disk-backed feature join are separate work.
There is no external business lookup, geocoding, API call, or data enrichment at runtime.

## Troubleshooting

- Missing normalized files: run the normalizer first.
- Wrong schema (plural `candidate_entity_ids`): use the internal scored pair file, not the official list export.
- Unknown IDs: the candidate file and normalized mock do not match; do not silently drop them.
- Invalid score: verify the file was produced by this Yash scoring implementation; another score contract needs an explicit adapter.
- Output already exists: choose a fresh directory. A failed run can leave partial files; absence of a completed manifest means it is not a handoff.
- `ModuleNotFoundError` when loading joblib: use this repository's code with `src` on the Python path and the pinned dependencies.
