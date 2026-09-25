"""Mock-only integration fixture, NOT retrieval or a model-quality evaluation."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from entity_resolution.features.baseline import FeatureExtractor, FEATURE_COLUMNS, ID_COLUMNS, join_pairs


def make_fixture_pairs(truth, target_ids):
    """All known positives plus one deterministic negative per reference."""
    targets = sorted(target_ids)
    rows = []
    for ref, matches in sorted(truth.items()):
        rows.extend((ref, target, 1) for target in sorted(matches))
        negative = next((target for target in targets if target not in matches), None)
        if negative is not None:
            rows.append((ref, negative, 0))
    return pd.DataFrame(rows, columns=ID_COLUMNS + ['label'])


def training_corpus(records, truth, train_ids):
    """Fit only training references and their labeled targets; exclude distractors.

    This conservative mock fit avoids including targets owned by held-out references.
    Production fit policy must be coordinated with the frozen team split.
    """
    selected = set(train_ids)
    for ref in train_ids:
        selected.update(truth[ref])
    return records[records.entity_id.isin(selected)]


def intersect_ids(path, allowed):
    with path.open(encoding='utf-8') as handle:
        return {line.strip() for line in handle if line.strip() in allowed}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(repo, out, batch_size=500):
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    if out.exists():
        raise ValueError(f'Output already exists: {out}. Choose a fresh --output directory.')
    started = time.perf_counter()
    normalized = repo / 'data/processed/normalized_mock'
    source_paths = [normalized / f'train_source{i}.parquet' for i in (1, 2, 3)]
    records = pd.concat([pd.read_parquet(path) for path in source_paths], ignore_index=True)
    truth_path = repo / 'data/mock/train_ground_truth.tsv'
    frame = pd.read_csv(truth_path, sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    truth = {r.source1_entity_id: set(r.matched_entity_ids.split(',')) if r.matched_entity_ids else set()
             for r in frame.itertuples()}
    split_dir = repo / 'src/entity_resolution/data'
    train_path, val_path = split_dir / 'train_reference_ids.txt', split_dir / 'val_reference_ids.txt'
    train_ids = intersect_ids(train_path, truth)
    val_ids = intersect_ids(val_path, truth)
    if train_ids & val_ids or train_ids | val_ids != set(truth) or not train_ids or not val_ids:
        raise ValueError('Frozen split must cover mock references exactly once, with both splits nonempty')
    fit_records = training_corpus(records, truth, train_ids)
    extractor = FeatureExtractor().fit(fit_records)
    targets = records.loc[records.entity_id.str.startswith(('S2-', 'S3-')), 'entity_id']
    pairs = make_fixture_pairs(truth, targets)
    out.mkdir(parents=True)
    extractor.save(out / 'extractor.joblib')
    summary = {'kind': 'fixture_only_not_blocking_or_model_evaluation', 'schema_version': 'baseline-v1',
               'feature_columns': FEATURE_COLUMNS, 'fit_records': len(fit_records),
               'fit_policy': 'mock training references and their true targets only',
               'batch_size': batch_size, 'splits': {},
               'input_sha256': {str(p.relative_to(repo)): file_hash(p)
                                for p in [*source_paths, truth_path, train_path, val_path]}}
    for name, ids in [('train', train_ids), ('validation', val_ids)]:
        selected = pairs[pairs.source1_entity_id.isin(ids)].reset_index(drop=True)
        selected.to_parquet(out / f'{name}_fixture_pairs_and_labels.parquet', index=False)
        schema = pa.schema([(key, pa.string()) for key in ID_COLUMNS] + [(key, pa.float32()) for key in FEATURE_COLUMNS])
        nulls = {key: 0 for key in FEATURE_COLUMNS}
        with pq.ParquetWriter(out / f'{name}_features.parquet', schema) as writer:
            for start in range(0, len(selected), batch_size):
                features = extractor.transform(join_pairs(selected.iloc[start:start + batch_size], records))
                writer.write_table(pa.Table.from_pandas(features, schema=schema, preserve_index=False))
                for key in nulls:
                    nulls[key] += int(features[key].isna().sum())
        summary['splits'][name] = {'references': len(ids), 'pairs': len(selected),
                                   'positive_pairs': int(selected.label.sum()), 'null_counts': nulls}
    summary['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=repo / 'artifacts/feature_baseline_demo')
    parser.add_argument('--batch-size', type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(repo, args.output, args.batch_size), indent=2))


if __name__ == '__main__':
    main()
