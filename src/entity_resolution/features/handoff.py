"""Turn a scored candidate TSV from the blocking stage into a MOCK feature handoff.

Candidates are supplied as a file (source1_entity_id, candidate_entity_id, score);
this runner does not call the blocker. It is a small-data integration runner, not
the full-scale production pipeline.
"""
import argparse
import csv
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution.features.baseline import FeatureExtractor, ID_COLUMNS, join_pairs
from entity_resolution.features.demo import file_hash, intersect_ids, training_corpus
from entity_resolution.features.scored import (
    SCHEMA_VERSION, SCORED_FEATURE_COLUMNS, add_retrieval_features, blocking_metrics, write_features,
)


def run(repo, out, pairs_path, batch_size=1000):
    repo, out = Path(repo), Path(out)
    if out.exists():
        raise ValueError(f'Output already exists: {out}; choose a fresh directory')
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    started = time.perf_counter()
    normalized = repo / 'data/processed/normalized_mock'
    source_paths = [normalized / f'train_source{i}.parquet' for i in (1, 2, 3)]
    sources = [pd.read_parquet(path) for path in source_paths]
    records = pd.concat(sources, ignore_index=True)
    if len(records) > 100_000:
        raise ValueError('Mock-only runner limit: 100,000 normalized records. Use a production batch lookup for full data.')
    # Validate records and truth before fitting or reading candidates.
    if not records.entity_id.is_unique or records.entity_id.isna().any():
        raise ValueError('Normalized record IDs must be unique and non-null')
    for source, prefix in zip(sources, ['S1-', 'S2-', 'S3-']):
        if not source.entity_id.str.startswith(prefix).all():
            raise ValueError(f'Wrong source ID prefix; expected {prefix}')
    truth_path = repo / 'data/mock/train_ground_truth.tsv'
    truth_frame = pd.read_csv(truth_path, sep='\t', dtype=str, keep_default_na=False,
                              encoding='utf-8-sig', quoting=csv.QUOTE_NONE)
    if not truth_frame.source1_entity_id.is_unique:
        raise ValueError('Duplicate ground-truth references')
    truth = {r.source1_entity_id: set(r.matched_entity_ids.split(',')) if r.matched_entity_ids else set()
             for r in truth_frame.itertuples()}
    if set(truth) != set(sources[0].entity_id):
        raise ValueError('Ground truth must cover every mock S1 exactly once')
    target_ids = set(sources[1].entity_id) | set(sources[2].entity_id)
    if any(not targets.issubset(target_ids) for targets in truth.values()):
        raise ValueError('Ground truth contains targets absent from the linked mock')
    split_dir = repo / 'src/entity_resolution/data'
    train_path, val_path = split_dir / 'train_reference_ids.txt', split_dir / 'val_reference_ids.txt'
    train_ids, val_ids = intersect_ids(train_path, truth), intersect_ids(val_path, truth)
    if not train_ids or not val_ids or train_ids & val_ids or train_ids | val_ids != set(truth):
        raise ValueError('Frozen split must partition all mock references')

    retrieval_start = time.perf_counter()
    pairs_path = Path(pairs_path).resolve()
    raw_pairs = pd.read_csv(pairs_path, sep='\t', dtype=str, keep_default_na=False,
                            quoting=csv.QUOTE_NONE, encoding='utf-8-sig')
    retrieval_seconds = time.perf_counter() - retrieval_start
    scored = add_retrieval_features(raw_pairs)
    if scored.groupby('source1_entity_id').size().gt(50).any():
        raise ValueError('Expected at most 50 candidates per reference')
    # Validate all IDs now, before creating a partial output directory.
    join_pairs(scored, records)
    fit_records = training_corpus(records, truth, train_ids)
    extractor = FeatureExtractor().fit(fit_records)
    out.mkdir(parents=True)
    extractor.save(out / 'extractor.joblib')
    raw_pairs[ID_COLUMNS + ['score']].to_csv(out / 'scored_candidates.tsv', sep='\t', index=False)
    scored.to_parquet(out / 'retrieval_features.parquet', index=False)
    schema = {'version': SCHEMA_VERSION, 'id_columns': ID_COLUMNS,
              'feature_columns': SCORED_FEATURE_COLUMNS, 'feature_dtype': 'float32',
              'rank': 'score descending, candidate_entity_id lexicographic ascending; one-based',
              'gap': 'highest score minus second-highest; repeated per reference; NaN if only one candidate',
              'score': 'max(name char TF-IDF cosine, address char TF-IDF cosine) + 0.05 for any token overlap'}
    (out / 'feature_schema.json').write_text(json.dumps(schema, indent=2), encoding='utf-8')
    inputs = [*source_paths, truth_path, train_path, val_path, pairs_path]
    summary = {
        'status': 'complete', 'scope': 'linked_mock_actual_blocking_not_full_dataset',
        'schema_version': SCHEMA_VERSION, 'batch_size': batch_size,
        'candidate_source': 'supplied_scored_tsv',
        'score_contract': schema['score'], 'fit_records': len(fit_records),
        'fit_policy': 'feature TF-IDF/IDF fitted on frozen training references and their true targets only',
        'blocking_fit_policy': 'Yash fits separate char vectorizers on the complete mock S2/S3 pool, including validation targets; no truth labels used',
        'retrieval_or_input_read_seconds': round(retrieval_seconds, 3),
        'input_sha256': {str(path): file_hash(path) for path in inputs},
        'versions': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'pyarrow', 'scikit-learn', 'rapidfuzz', 'joblib']},
        'python': platform.python_version(), 'splits': {},
        'source_sha256': {str(Path(path).name): file_hash(Path(path)) for path in
                          [__file__, Path(__file__).with_name('scored.py'),
                           Path(__file__).with_name('baseline.py')]},
    }
    for name, ids in [('train', train_ids), ('validation', val_ids)]:
        selected = scored[scored.source1_entity_id.isin(ids)].reset_index(drop=True)
        print(f'Writing {name}: {len(selected):,} pairs, {len(ids):,} references', flush=True)
        stats = write_features(selected, records, extractor, out / f'{name}_features.parquet', batch_size)
        labels = selected[ID_COLUMNS].copy()
        labels['label'] = np.array([int(row.candidate_entity_id in truth[row.source1_entity_id])
                                    for row in selected.itertuples()], dtype=np.uint8)
        labels.to_parquet(out / f'{name}_labels.parquet', index=False)
        # Preserve the ENTIRE truth, not merely retrieved positives, for downstream F0.5.
        truth_frame[truth_frame.source1_entity_id.isin(ids)].to_csv(
            out / f'{name}_ground_truth.tsv', sep='\t', index=False)
        (out / f'{name}_reference_ids.txt').write_text('\n'.join(sorted(ids)) + '\n', encoding='utf-8')
        ordered = selected.sort_values(['source1_entity_id', 'candidate_rank'])
        candidate_lists = ordered.groupby('source1_entity_id').candidate_entity_id.agg(','.join)
        exported = pd.DataFrame({'source1_entity_id': sorted(ids)})
        exported['candidate_entity_ids'] = exported.source1_entity_id.map(candidate_lists).fillna('')
        exported.to_csv(out / f'{name}_candidate_pairs.tsv', sep='\t', index=False)
        summary['splits'][name] = {
            'blocking': blocking_metrics(scored, truth, ids), 'features': stats,
            'positive_pairs': int(labels.label.sum()), 'negative_pairs': int((labels.label == 0).sum()),
        }
    summary['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    summary['output_sha256'] = {path.name: file_hash(path) for path in sorted(out.iterdir()) if path.is_file()}
    # Manifest is the last write. Failed runs never get a completed manifest.
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=repo / 'artifacts/scored_mock_v2')
    parser.add_argument('--pairs', type=Path, required=True,
                        help='Pre-generated scored candidate TSV from the blocking stage for this exact mock '
                             '(columns: source1_entity_id, candidate_entity_id, score)')
    parser.add_argument('--batch-size', type=int, default=1000)
    args = parser.parse_args()
    summary = run(repo, args.output, args.pairs, args.batch_size)
    print(json.dumps({key: summary[key] for key in ['scope', 'splits', 'elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
