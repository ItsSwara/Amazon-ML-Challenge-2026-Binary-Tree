import pandas as pd
from entity_resolution.features.demo import make_fixture_pairs, training_corpus


def test_fixture_negative_is_not_a_true_match():
    truth = {'S1-1': {'S2-1'}, 'S1-2': set()}
    pairs = make_fixture_pairs(truth, ['S2-1', 'S2-2'])
    assert len(pairs) == 3
    assert not pairs.duplicated(['source1_entity_id', 'candidate_entity_id']).any()
    for row in pairs.itertuples():
        assert row.label == int(row.candidate_entity_id in truth[row.source1_entity_id])


def test_fit_corpus_excludes_validation_targets():
    records = pd.DataFrame({'entity_id': ['S1-1', 'S1-2', 'S2-1', 'S2-2', 'S3-1']})
    truth = {'S1-1': {'S2-1'}, 'S1-2': {'S2-2'}}
    fit = training_corpus(records, truth, {'S1-1'})
    assert set(fit.entity_id) == {'S1-1', 'S2-1'}
