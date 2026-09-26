"""Regression coverage for the feature stage's real inference call boundary."""
import pandas as pd
import pytest

from entity_resolution.inference.run_inference import RecordStore, reference_batches


def test_header_only_candidates_yield_no_batches(tmp_path):
    path = tmp_path / 'empty.tsv'
    path.write_text('source1_entity_id\tcandidate_entity_id\tscore\n')
    assert list(reference_batches(path, 1)) == []


def test_record_cache_rebuilds_for_same_count_changed_values(tmp_path):
    source = tmp_path / 'source.parquet'
    records = pd.DataFrame({'entity_id': ['S1-a'], 'name_norm': ['old name'],
                            'address_norm': [None], 'country': ['US']})
    records.to_parquet(source, index=False)
    path = tmp_path / 'records.sqlite'
    store = RecordStore.build(path, [source])
    store.close()
    records['name_norm'] = 'new name'
    records.to_parquet(source, index=False)
    store = RecordStore.build(path, [source])
    try:
        assert store.lookup(['S1-a']).name_norm.iloc[0] == 'new name'
    finally:
        store.close()


def test_path_change_with_same_count_rebuilds_record_cache(tmp_path):
    a, b = tmp_path / 'a.parquet', tmp_path / 'b.parquet'
    rec = pd.DataFrame({'entity_id': ['S1-a'], 'name_norm': ['shop'],
                        'address_norm': [None], 'country': ['US']})
    rec.to_parquet(a, index=False)
    rec.assign(country='India').to_parquet(b, index=False)
    path = tmp_path / 'records.sqlite'
    store = RecordStore.build(path, [a])
    store.close()
    store = RecordStore.build(path, [b])
    try:
        assert store.lookup(['S1-a']).country.iloc[0] == 'India'
    finally:
        store.close()


def test_bad_empty_header_still_rejected(tmp_path):
    path = tmp_path / 'bad.tsv'
    path.write_text('source1_entity_id\tcandidate_entity_ids\n')
    with pytest.raises(ValueError, match='Expected columns'):
        list(reference_batches(path, 5))


@pytest.mark.parametrize('chunk', [1, 10])
def test_oversized_group_stops_carry_growth(tmp_path, chunk):
    path = tmp_path / 'bad.tsv'
    path.write_text('source1_entity_id\tcandidate_entity_id\tscore\n' +
                    ''.join(f'S1-a\tS2-{i}\t0.5\n' for i in range(51)))
    with pytest.raises(ValueError, match='50|candidate limit'):
        list(reference_batches(path, chunk))
