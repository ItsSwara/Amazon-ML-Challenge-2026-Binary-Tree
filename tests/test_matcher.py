import numpy as np
import pandas as pd
import pytest

from entity_resolution.features.scored import SCORED_FEATURE_COLUMNS
from entity_resolution.models.matcher import load_matcher, predict_proba, save_matcher, train_matcher


def synthetic(n_refs=120, per_ref=10, seed=0):
    rng = np.random.default_rng(seed)
    n = n_refs * per_ref
    label = (rng.random(n) < 0.15).astype(int)
    data = pd.DataFrame(rng.random((n, len(SCORED_FEATURE_COLUMNS))).astype('float32'),
                        columns=SCORED_FEATURE_COLUMNS)
    data['name_tfidf_cosine'] = np.clip(0.3 + 0.6 * label + rng.normal(0, 0.1, n), 0, 1)
    data.loc[rng.random(n) < 0.05, 'address_tfidf_cosine'] = np.nan
    data['source1_entity_id'] = np.repeat([f'S1-{i}' for i in range(n_refs)], per_ref)
    return data, label


def test_calibrated_probabilities_are_valid_and_separate_classes(tmp_path):
    data, label = synthetic()
    model = train_matcher(data, label, data.source1_entity_id)
    prob = predict_proba(model, data)
    assert prob.shape == (len(data),) and np.all((prob >= 0) & (prob <= 1))
    assert prob[label == 1].mean() > 0.8 > 0.2 > prob[label == 0].mean()
    save_matcher(model, tmp_path / 'm.joblib')
    np.testing.assert_array_equal(prob, predict_proba(load_matcher(tmp_path / 'm.joblib'), data))


def test_feature_order_comes_from_contract_not_frame_order():
    data, label = synthetic()
    model = train_matcher(data, label, data.source1_entity_id)
    shuffled = data[list(reversed(data.columns))]
    np.testing.assert_array_equal(predict_proba(model, data), predict_proba(model, shuffled))


def test_missing_feature_column_rejected():
    data, label = synthetic()
    model = train_matcher(data, label, data.source1_entity_id)
    with pytest.raises(ValueError, match='Missing feature'):
        predict_proba(model, data.drop(columns='candidate_rank'))


def test_single_class_training_rejected():
    data, _ = synthetic()
    with pytest.raises(ValueError, match='both classes'):
        train_matcher(data, np.zeros(len(data), dtype=int), data.source1_entity_id)
