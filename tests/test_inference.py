import numpy as np
import pandas as pd
import pytest

from entity_resolution.features.baseline import FeatureExtractor
from entity_resolution.features.scored import SCORED_FEATURE_COLUMNS
from entity_resolution.inference.decision import CorroborationRule
from entity_resolution.inference.run_inference import (
    CANDIDATE_HEADER, MATCHING_HEADER, RecordStore, reference_batches, run, verify_outputs,
)
from entity_resolution.models.matcher import save_matcher, train_matcher

PAIRS = [('S1-a', 'S2-x', 0.9), ('S1-a', 'S3-y', 0.5), ('S1-a', 'S2-q', 0.2),
         ('S1-b', 'S2-x', 0.8), ('S1-b', 'S2-z', 0.7), ('S1-c', 'S3-w', 0.95)]


def write_pairs(path, rows):
    pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_id', 'score']).to_csv(
        path, sep='\t', index=False)
    return path


@pytest.mark.parametrize('chunk_rows', [1, 2, 4, 100])
def test_reference_batches_keep_groups_whole_across_chunk_boundaries(tmp_path, chunk_rows):
    path = write_pairs(tmp_path / 'p.tsv', PAIRS)
    batches = list(reference_batches(path, chunk_rows))
    refs = [set(b.source1_entity_id) for b in batches]
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            assert not a & b
    assert pd.concat(batches).candidate_entity_id.tolist() == [p[1] for p in PAIRS]


def test_reference_batches_reject_ungrouped_file(tmp_path):
    path = write_pairs(tmp_path / 'p.tsv', PAIRS + [('S1-a', 'S3-late', 0.1)])
    with pytest.raises(ValueError, match='grouped|contiguous'):
        list(reference_batches(path, 2))


def write_lists(path, header, rows):
    path.write_text(header + '\n' + ''.join(f'{r}\t{",".join(ids)}\n' for r, ids in rows), encoding='utf-8')
    return path


GOOD_CAND = [('S1-a', ['S2-x', 'S3-y']), ('S1-b', ['S2-z']), ('S1-c', [])]
GOOD_MATCH = [('S1-a', ['S2-x']), ('S1-b', []), ('S1-c', [])]


def test_verify_outputs_accepts_valid_files(tmp_path):
    m = write_lists(tmp_path / 'm.tsv', MATCHING_HEADER, GOOD_MATCH)
    c = write_lists(tmp_path / 'c.tsv', CANDIDATE_HEADER, GOOD_CAND)
    assert verify_outputs(m, c, ['S1-a', 'S1-b', 'S1-c'])['matched_ids'] == 1


@pytest.mark.parametrize('match_rows, message', [
    ([('S1-a', ['S2-q'])] + GOOD_MATCH[1:], 'outside candidate'),
    (GOOD_MATCH[:2], 'missing'),
    (GOOD_MATCH + [('S1-a', [])], 'duplicate row'),
    ([('S1-a', ['S2-x', 'S2-x'])] + GOOD_MATCH[1:], 'repeated ID'),
    ([('S1-a', ['S1-b'])] + GOOD_MATCH[1:], 'S1 ID'),
    ([('S1-a', ['S2-x']), ('S1-b', ['S2-x']), ('S1-c', [])], 'outside candidate|matched to both'),
])
def test_verify_outputs_rejects_bad_matching_file(tmp_path, match_rows, message):
    m = write_lists(tmp_path / 'm.tsv', MATCHING_HEADER, match_rows)
    c = write_lists(tmp_path / 'c.tsv', CANDIDATE_HEADER, GOOD_CAND + [])
    with pytest.raises(ValueError, match=message):
        verify_outputs(m, c, ['S1-a', 'S1-b', 'S1-c'])


def test_verify_outputs_rejects_candidate_row_for_unknown_reference(tmp_path):
    m = write_lists(tmp_path / 'm.tsv', MATCHING_HEADER, GOOD_MATCH)
    c = write_lists(tmp_path / 'c.tsv', CANDIDATE_HEADER, GOOD_CAND + [('S1-zzz', [])])
    with pytest.raises(ValueError, match='unknown S1'):
        verify_outputs(m, c, ['S1-a', 'S1-b', 'S1-c'])


def make_split(tmp_path):
    """Tiny test split: S1-d has no candidates; S1-b's addresses are in France."""
    norm = tmp_path / 'normalized'
    norm.mkdir()
    recs = {
        1: [('S1-a', 'acme shop', '1 main street', 'US'), ('S1-b', 'acme shop', '1 main street', 'France'),
            ('S1-c', 'zed corp', '9 high road', 'US'), ('S1-d', 'lonely', None, 'US')],
        2: [('S2-x', 'acme shop', '1 main street', 'US'), ('S2-q', 'other', '5 far lane', 'US'),
            ('S2-z', 'acme shops', '1 main st', 'US')],
        3: [('S3-y', 'acme', '1 main street', 'US'), ('S3-w', 'zed corp', '9 high road', 'US')],
    }
    for i, rows in recs.items():
        pd.DataFrame(rows, columns=['entity_id', 'name_norm', 'address_norm', 'country']).assign(
            legal_suffix=None).to_parquet(norm / f'test_source{i}.parquet', index=False)
    extractor = FeatureExtractor().fit(pd.concat(
        [pd.read_parquet(norm / f'test_source{i}.parquet') for i in (1, 2, 3)]))
    extractor.save(tmp_path / 'extractor.joblib')
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((400, len(SCORED_FEATURE_COLUMNS))), columns=SCORED_FEATURE_COLUMNS)
    y = (X.name_token_sort > 0.6).astype(int)
    save_matcher(train_matcher(X, y, np.arange(400) // 4), tmp_path / 'model.joblib')
    return norm, write_pairs(tmp_path / 'pairs.tsv', PAIRS)


def run_tiny(tmp_path, chunk_rows, out):
    norm, pairs = make_split(tmp_path) if not (tmp_path / 'pairs.tsv').exists() else (
        tmp_path / 'normalized', tmp_path / 'pairs.tsv')
    rule = CorroborationRule(0.1, 0.45, 0.8)
    return run(pairs, norm, 'test', tmp_path / 'model.joblib', tmp_path / 'extractor.joblib', 0.5,
               tmp_path / out, tmp_path / 'store.sqlite', chunk_rows, rule)


def test_end_to_end_outputs_are_batch_size_invariant_and_complete(tmp_path):
    small = run_tiny(tmp_path, 1, 'o1')
    large = run_tiny(tmp_path, 1000, 'o2')
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        assert (tmp_path / 'o1' / name).read_text() == (tmp_path / 'o2' / name).read_text()
    assert small['verification'] == large['verification']
    match = pd.read_csv(tmp_path / 'o1/matching_results.tsv', sep='\t', dtype=str, keep_default_na=False)
    cand = pd.read_csv(tmp_path / 'o1/candidate_pairs.tsv', sep='\t', dtype=str, keep_default_na=False)
    assert sorted(match.source1_entity_id) == sorted(cand.source1_entity_id) == ['S1-a', 'S1-b', 'S1-c', 'S1-d']
    lookup = dict(zip(cand.source1_entity_id, cand.candidate_entity_ids))
    assert lookup['S1-d'] == ''                               # no candidates -> empty row
    assert lookup['S1-a'] == 'S2-x,S3-y,S2-q'                 # exactly the scored set, by rank
    matched = dict(zip(match.source1_entity_id, match.matched_entity_ids))
    assert matched['S1-b'] == ''                              # France ref, only US candidates
    assert matched['S1-d'] == ''


def test_record_store_rebuilds_when_source_changes(tmp_path):
    norm, _ = make_split(tmp_path)
    paths = [norm / f'test_source{i}.parquet' for i in (1, 2, 3)]
    store = RecordStore.build(tmp_path / 's.sqlite', paths)
    assert len(store.lookup(['S1-a', 'S2-x', 'S9-missing'])) == 2
    store.close()
    pd.read_parquet(paths[1]).iloc[:1].to_parquet(paths[1], index=False)
    store = RecordStore.build(tmp_path / 's.sqlite', paths)
    assert len(store.lookup(['S2-q'])) == 0
    store.close()
