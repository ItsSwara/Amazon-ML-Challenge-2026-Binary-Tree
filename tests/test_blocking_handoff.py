import json
import pandas as pd
import pytest

from entity_resolution.features.handoff import run


def prepare(tmp_path):
    repo = tmp_path / 'repo'
    normalized = repo / 'data/processed/normalized_mock'
    normalized.mkdir(parents=True)
    sources = [(['S1-a', 'S1-b', 'S1-c', 'S1-d'], ['alpha', 'beta', 'single', 'delta']),
               (['S2-a', 'S2-b'], ['alpha', 'beta']), (['S3-d'], ['delta'])]
    for i, (ids, names) in enumerate(sources, 1):
        pd.DataFrame({'entity_id': ids, 'name_norm': names, 'address_norm': ['road'] * len(ids),
                      'country': ['US'] * len(ids)}).to_parquet(normalized / f'train_source{i}.parquet')
    mock = repo / 'data/mock'
    mock.mkdir()
    (mock / 'train_ground_truth.tsv').write_text(
        'source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a\nS1-b\tS2-b\nS1-c\t\nS1-d\tS3-d\n')
    splits = repo / 'src/entity_resolution/data'
    splits.mkdir(parents=True)
    (splits / 'train_reference_ids.txt').write_text('S1-a\nS1-c\n')
    (splits / 'val_reference_ids.txt').write_text('S1-b\nS1-d\n')
    pairs = tmp_path / 'pairs.tsv'
    pairs.write_text('source1_entity_id\tcandidate_entity_id\tscore\n'
                     'S1-a\tS2-a\t1.05\nS1-a\tS2-b\t0.2\nS1-b\tS2-b\t0.9\n')
    return repo, pairs


def test_handoff_labels_recall_empty_refs_and_rerun_guard(tmp_path):
    repo, pairs = prepare(tmp_path)
    out = tmp_path / 'result'
    result = run(repo, out, pairs_path=pairs, batch_size=1)
    assert result['splits']['validation']['blocking']['micro_recall'] == 0.5
    assert result['fit_records'] == 3
    features = pd.read_parquet(out / 'train_features.parquet')
    labels = pd.read_parquet(out / 'train_labels.parquet')
    assert features.merge(labels, validate='one_to_one').label.tolist() == [1, 0]
    val_truth = pd.read_csv(out / 'validation_ground_truth.tsv', sep='\t', keep_default_na=False)
    assert len(val_truth) == 2  # Keep the completely missed reference for honest scoring.
    exported = pd.read_csv(out / 'validation_candidate_pairs.tsv', sep='\t', keep_default_na=False)
    assert exported.set_index('source1_entity_id').loc['S1-d', 'candidate_entity_ids'] == ''
    assert json.loads((out / 'run_summary.json').read_text())['status'] == 'complete'
    assert (out / 'feature_schema.json').is_file()
    with pytest.raises(ValueError, match='already exists'):
        run(repo, out, pairs_path=pairs)


def test_missing_target_fails_before_output(tmp_path):
    repo, pairs = prepare(tmp_path)
    pairs.write_text(pairs.read_text().replace('S2-b\t0.2', 'S2-unknown\t0.2'))
    with pytest.raises(ValueError, match='Unknown'):
        run(repo, tmp_path / 'bad', pairs_path=pairs)
    assert not (tmp_path / 'bad/run_summary.json').exists()
