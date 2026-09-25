# Amazon ML Challenge 2026 — Binary Tree

Repository structure only. No implementation, dependencies, or datasets added.

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

Empty directories contain `.gitkeep` so Git can track the structure.


## Setup

Each teammate does this once, individually. Nobody shares `paths.local.yaml`.

1. `pip install -r requirements.txt`
2. Copy `configs/paths.example.yaml` to `configs/paths.local.yaml`.
3. Edit `configs/paths.local.yaml` and set `dataset_dir` to your own local dataset folder (the one containing `train_source1.tsv`, `test_source1.tsv`, ...).
4. Verify: `python scripts/smoke_test_paths.py`

`configs/paths.local.yaml` is gitignored. In code, use `from entity_resolution.config import get_dataset_dir` instead of hardcoding paths.
