import numpy as np
import pandas as pd
import pytest

from entity_resolution.features.baseline import FeatureExtractor, FEATURE_COLUMNS, join_pairs


def records():
    return pd.DataFrame({
        'entity_id': ['S1-1', 'S2-1', 'S3-1'],
        'name_norm': ['rare shop', 'rare shop', 'common store'],
        'address_norm': ['12 main road', '12 main road', None],
    })


def pairs():
    return pd.DataFrame({'source1_entity_id': ['S1-1', 'S1-1'],
                         'candidate_entity_id': ['S2-1', 'S3-1']})


def test_identity_and_missing_address():
    model = FeatureExtractor().fit(records())
    out = model.transform(join_pairs(pairs(), records()))
    assert list(out.columns) == ['source1_entity_id', 'candidate_entity_id', *FEATURE_COLUMNS]
    assert out.loc[0, 'name_jaro_winkler'] == 1
    assert out.loc[0, 'address_tfidf_cosine'] == pytest.approx(1)
    assert out.loc[0, 'name_idf_overlap'] == pytest.approx(1)
    assert np.isnan(out.loc[1, 'address_token_sort'])
    assert out.loc[1, 'candidate_address_missing'] == 1


def test_chunking_and_persistence(tmp_path):
    model = FeatureExtractor().fit(records())
    joined = join_pairs(pairs(), records())
    expected = model.transform(joined)
    chunked = pd.concat([model.transform(joined.iloc[:1]), model.transform(joined.iloc[1:])], ignore_index=True)
    pd.testing.assert_frame_equal(expected, chunked)
    model.save(tmp_path / 'model.joblib')
    pd.testing.assert_frame_equal(expected, FeatureExtractor.load(tmp_path / 'model.joblib').transform(joined))


def test_missing_both_is_not_perfect_match_and_nan_name_is_literal():
    rec = records()
    rec['address_norm'] = pd.NA
    rec['name_norm'] = 'nan'
    out = FeatureExtractor().fit(rec).transform(join_pairs(pairs(), rec))
    assert out.address_jaro_winkler.isna().all()
    assert out.address_tfidf_cosine.isna().all()
    assert (out.name_jaro_winkler == 1).all()
    assert (out.source1_name_missing == 0).all()


def test_unseen_tokens_do_not_refit():
    model = FeatureExtractor().fit(records())
    rec = records()
    rec['name_norm'] = 'unseenxyz'
    out = model.transform(join_pairs(pairs(), rec))
    assert (out.name_tfidf_cosine == 0).all()
    assert (out.name_idf_overlap == 0).all()


@pytest.mark.parametrize('problem', ['unknown', 'duplicate_record', 'duplicate_pair', 'wrong_source'])
def test_bad_ids_fail(problem):
    rec, pair = records(), pairs()
    if problem == 'unknown':
        pair.loc[0, 'candidate_entity_id'] = 'S2-missing'
    elif problem == 'duplicate_record':
        rec = pd.concat([rec, rec.iloc[:1]])
    elif problem == 'duplicate_pair':
        pair = pd.concat([pair, pair.iloc[:1]])
    else:
        pair.loc[0, 'candidate_entity_id'] = 'S1-1'
    with pytest.raises(ValueError):
        join_pairs(pair, rec)


def test_empty_candidates_keep_schema():
    model = FeatureExtractor().fit(records())
    out = model.transform(join_pairs(pairs().iloc[:0], records()))
    assert out.empty
    assert list(out.columns) == ['source1_entity_id', 'candidate_entity_id', *FEATURE_COLUMNS]
