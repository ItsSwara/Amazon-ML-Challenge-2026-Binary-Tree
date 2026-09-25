# Amazon ML Challenge 2026 — Binary Tree

Entity-resolution pipeline under development. Data audit, normalization, frozen splits,
scoring and a local feature-engineering baseline are available.

```text
data/
  raw/
  mock/
  processed/
src/entity_resolution/
  data/
  normalization/
  blocking/
  features/
  mining/
  models/
  evaluation/
  inference/
  submission/
configs/
experiments/
notebooks/
docs/
tests/
scripts/
artifacts/
output/
requirements.txt
README.md
```

Unimplemented stages contain `.gitkeep` so Git can track the structure.

## Feature baseline

See [feature contract and local run instructions](docs/schemas/features_baseline.md).
The mock demo generates 12 text/missingness features and separate fixture labels.
For actual Yash blocking candidates, use the [scored feature handoff v2](docs/schemas/scored_features_v2.md).
It adds score/rank/gap (15 features total), separate labels and split-specific blocking recall.
This integration is verified on the linked mock; full-scale processing remains pending.


## Setup

Each teammate does this once, individually. Nobody shares `paths.local.yaml`.

1. `pip install -r requirements.txt`
2. Copy `configs/paths.example.yaml` to `configs/paths.local.yaml`.
3. Edit `configs/paths.local.yaml` and set `dataset_dir` to your own local dataset folder (the one containing the `train/` and `test/` subfolders).
4. Verify: `python scripts/smoke_test_paths.py`

`configs/paths.local.yaml` is gitignored. In code, use `from entity_resolution.config import get_dataset_dir` instead of hardcoding paths.
