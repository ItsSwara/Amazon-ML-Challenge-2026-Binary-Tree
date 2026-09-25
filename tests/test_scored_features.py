import numpy as np
import pandas as pd
import pytest

from entity_resolution.features.scored import (
    SCORED_FEATURE_COLUMNS, add_retrieval_features, blocking_metrics, write_features,
)
from entity_resolution.features.baseline import FeatureExtractor


def pairs():
    return pd.DataFrame({
        'source1_entity_id': ['S1-a', 'S1-b', 'S1-a', 'S1-a'],
        'candidate_entity_id': ['S3-z', 'S2-b', 'S2-a', 'S2-c'],
        'score': [0.9, 0.4, 0.9, 0.2],
    })


def test_global_ranks_tie_break_and_single_candidate_margin():
    out = add_retrieval_features(pairs())
    assert out.candidate_rank.tolist() == [2, 1, 1, 3]
    assert out.blocking_top1_top2_gap.iloc[[0, 2, 3]].eq(0).all()
    assert np.isnan(out.blocking_top1_top2_gap.iloc[1])
    pd.testing.assert_series_equal(out.candidate_entity_id, pairs().candidate_entity_id)


def test_nonzero_gap_is_shared_by_all_candidates():
    data = pairs()
    data.loc[2, 'score'] = 0.5
    out = add_retrieval_features(data)
    assert out.loc[out.source1_entity_id == 'S1-a', 'blocking_top1_top2_gap'].tolist() == pytest.approx([0.4] * 3)


def test_float_scores_preserve_ranks_after_text_roundtrip():
    data = pairs().iloc[:2].copy()
    data['source1_entity_id'] = 'S1-a'
    data['candidate_entity_id'] = ['S3-z', 'S2-a']
    data['score'] = [0.30000000000000004, 0.3]
    expected = add_retrieval_features(data)
    data['score'] = data.score.map(repr)
    pd.testing.assert_frame_equal(expected, add_retrieval_features(data))


@pytest.mark.parametrize('problem', ['duplicate', 'nan', 'infinity', 'negative', 'too_high', 'wrong_source'])
def test_invalid_candidates_rejected(problem):
    data = pairs()
    if problem == 'duplicate':
        data = pd.concat([data, data.iloc[:1]])
    elif problem == 'wrong_source':
        data.loc[0, 'candidate_entity_id'] = 'S1-a'
    else:
        data.loc[0, 'score'] = {'nan': np.nan, 'infinity': np.inf, 'negative': -1, 'too_high': 9}[problem]
    with pytest.raises(ValueError):
        add_retrieval_features(data)


def test_recall_includes_references_without_candidates():
    truth = {'S1-a': {'S2-a', 'S3-z'}, 'S1-b': set(), 'S1-c': {'S3-x'}}
    result = blocking_metrics(pairs(), truth, set(truth))
    assert result['true_links'] == 3
    assert result['retrieved_true_links'] == 2
    assert result['micro_recall'] == pytest.approx(2 / 3)
    assert result['macro_recall_non_singletons'] == pytest.approx(0.5)
    assert result['references_without_candidates'] == 1


def test_feature_batches_use_complete_group_ranks(tmp_path):
    data = add_retrieval_features(pairs())
    rec = pd.DataFrame({'entity_id': ['S1-a', 'S1-b', 'S3-z', 'S2-b', 'S2-a', 'S2-c'],
                        'name_norm': ['shop'] * 6, 'address_norm': [None] * 6})
    model = FeatureExtractor().fit(rec)
    for size in (1, 3):
        write_features(data, rec, model, tmp_path / f'b{size}.parquet', size)
    one = pd.read_parquet(tmp_path / 'b1.parquet')
    pd.testing.assert_frame_equal(one, pd.read_parquet(tmp_path / 'b3.parquet'))
    assert list(one.columns) == ['source1_entity_id', 'candidate_entity_id', *SCORED_FEATURE_COLUMNS]
    assert one.candidate_rank.tolist() == [2, 1, 1, 3]
    assert 'label' not in one


def test_empty_feature_file_has_schema(tmp_path):
    rec = pd.DataFrame({'name_norm': ['shop'], 'address_norm': ['road']})
    data = add_retrieval_features(pairs().iloc[:0])
    model = FeatureExtractor().fit(rec)
    write_features(data, rec, model, tmp_path / 'empty.parquet', 1)
    assert list(pd.read_parquet(tmp_path / 'empty.parquet').columns) == [
        'source1_entity_id', 'candidate_entity_id', *SCORED_FEATURE_COLUMNS]
