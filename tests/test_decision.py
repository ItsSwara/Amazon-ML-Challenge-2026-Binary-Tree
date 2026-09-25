import numpy as np
import pandas as pd
import pytest

from entity_resolution.inference.decision import (
    CorroborationRule, assign_candidates, best_threshold, corroboration_guard, country_veto,
    decide, singleton_valve, sweep_thresholds, to_predictions,
)

RULE = CorroborationRule(min_address_cosine=0.1, min_address_token_sort=0.45,
                         missing_address_min_probability=0.8)


def frame(rows):
    """rows: (ref, cand, probability[, overrides]) with agreeing addresses by default."""
    base = dict(address_tfidf_cosine=0.9, address_token_sort=0.9, source1_address_missing=0.0,
                candidate_address_missing=0.0, source1_country='US', candidate_country='US')
    out = []
    for row in rows:
        ref, cand, p, *extra = row
        out.append({'source1_entity_id': ref, 'candidate_entity_id': cand, 'probability': p,
                    **base, **(extra[0] if extra else {})})
    return pd.DataFrame(out)


def pairs_of(accepted):
    return set(zip(accepted.source1_entity_id, accepted.candidate_entity_id))


def test_two_references_wanting_same_candidate_only_higher_gets_it():
    data = frame([('S1-a', 'S2-x', 0.91), ('S1-b', 'S2-x', 0.97), ('S1-a', 'S3-y', 0.95)])
    out = decide(data, 0.5, veto=False)
    assert pairs_of(out) == {('S1-b', 'S2-x'), ('S1-a', 'S3-y')}
    assert out.candidate_entity_id.is_unique


def test_assignment_loser_keeps_its_other_matches_and_empty_losers_stay_empty():
    data = frame([('S1-a', 'S2-x', 0.9), ('S1-b', 'S2-x', 0.8)])
    preds = to_predictions(assign_candidates(data), ['S1-a', 'S1-b'])
    assert preds == {'S1-a': {'S2-x'}, 'S1-b': set()}


def test_assignment_tie_is_deterministic_by_reference_id():
    data = frame([('S1-z', 'S2-x', 0.9), ('S1-a', 'S2-x', 0.9)])
    for order in (data, data.iloc[::-1]):
        assert pairs_of(assign_candidates(order)) == {('S1-a', 'S2-x')}


def test_assignment_resolves_three_way_and_chained_claims():
    data = frame([('S1-a', 'S2-x', 0.7), ('S1-b', 'S2-x', 0.9), ('S1-c', 'S2-x', 0.8),
                  ('S1-b', 'S3-y', 0.6), ('S1-c', 'S3-y', 0.65)])
    assert pairs_of(assign_candidates(data)) == {('S1-b', 'S2-x'), ('S1-c', 'S3-y')}


def test_singleton_valve_empties_reference_whose_best_is_below_threshold():
    data = frame([('S1-a', 'S2-x', 0.49), ('S1-a', 'S2-y', 0.2), ('S1-b', 'S2-z', 0.8), ('S1-b', 'S2-w', 0.3)])
    preds = to_predictions(singleton_valve(data, 0.5), ['S1-a', 'S1-b', 'S1-none'])
    assert preds == {'S1-a': set(), 'S1-b': {'S2-z'}, 'S1-none': set()}


def test_threshold_is_inclusive():
    assert len(singleton_valve(frame([('S1-a', 'S2-x', 0.5)]), 0.5)) == 1


def test_corroboration_rejects_name_only_match_when_addresses_conflict():
    data = frame([('S1-a', 'S2-x', 0.99, dict(address_tfidf_cosine=0.02, address_token_sort=0.3))])
    assert corroboration_guard(data, RULE).empty


def test_corroboration_needs_both_address_signals_low_to_call_conflict():
    data = frame([('S1-a', 'S2-x', 0.99, dict(address_tfidf_cosine=0.02, address_token_sort=0.7)),
                  ('S1-a', 'S2-y', 0.99, dict(address_tfidf_cosine=0.5, address_token_sort=0.3))])
    assert len(corroboration_guard(data, RULE)) == 2


@pytest.mark.parametrize('side', ['source1_address_missing', 'candidate_address_missing'])
def test_corroboration_missing_address_keeps_high_and_drops_low_confidence(side):
    missing = {side: 1.0, 'address_tfidf_cosine': np.nan, 'address_token_sort': np.nan}
    data = frame([('S1-a', 'S2-hi', 0.95, missing), ('S1-a', 'S2-lo', 0.7, missing)])
    assert pairs_of(corroboration_guard(data, RULE)) == {('S1-a', 'S2-hi')}


def test_missing_address_is_not_treated_as_conflict():
    """NaN similarities must not compare as 'low' and trigger the conflict branch."""
    data = frame([('S1-a', 'S2-x', 0.99, dict(candidate_address_missing=1.0,
                                              address_tfidf_cosine=np.nan, address_token_sort=np.nan))])
    assert len(corroboration_guard(data, RULE)) == 1


def test_country_veto_rejects_mismatch_even_at_certainty():
    data = frame([('S1-a', 'S2-x', 1.0, dict(candidate_country='France')), ('S1-a', 'S2-y', 0.6)])
    assert pairs_of(country_veto(data)) == {('S1-a', 'S2-y')}


def test_country_veto_ignores_unknown_and_case_whitespace():
    data = frame([('S1-a', 'S2-x', 0.9, dict(candidate_country=None)),
                  ('S1-a', 'S2-y', 0.9, dict(source1_country='')),
                  ('S1-a', 'S2-z', 0.9, dict(candidate_country=' us '))])
    assert len(country_veto(data)) == 3


def test_guards_are_subtractive_on_random_pairs():
    rng = np.random.default_rng(0)
    n = 3000
    data = pd.DataFrame({
        'source1_entity_id': [f'S1-{i}' for i in rng.integers(0, 300, n)],
        'candidate_entity_id': [f'S{rng.integers(2, 4)}-{i}' for i in rng.integers(0, 400, n)],
        'probability': rng.random(n),
        'address_tfidf_cosine': rng.random(n), 'address_token_sort': rng.random(n),
        'source1_address_missing': (rng.random(n) < 0.1).astype(float),
        'candidate_address_missing': (rng.random(n) < 0.1).astype(float),
        'source1_country': rng.choice(['US', 'India', 'France'], n),
        'candidate_country': rng.choice(['US', 'India', 'France', None], n),
    }).drop_duplicates(['source1_entity_id', 'candidate_entity_id'])
    for t in (0.2, 0.5, 0.8):
        plain = pairs_of(decide(data, t, rule=None, veto=False))
        guarded = decide(data, t, rule=RULE, veto=True)
        assert pairs_of(guarded) <= plain
        assert guarded.candidate_entity_id.is_unique
        assert (guarded.probability >= t).all()


def test_sweep_uses_scorer_over_all_references():
    data = frame([('S1-a', 'S2-x', 0.9), ('S1-a', 'S2-bad', 0.4)])
    truth = {'S1-a': {'S2-x'}, 'S1-single': set()}
    sweep = sweep_thresholds(data, truth, [0.3, 0.5], lambda p, t: decide(p, t, veto=False))
    # 0.3: S1-a P=1/2 R=1 -> 5/9; singleton correct -> 1. 0.5: both perfect.
    assert sweep.f05.tolist() == pytest.approx([(5 / 9 + 1) / 2, 1.0])
    assert best_threshold(sweep) == (0.5, pytest.approx(1.0))
