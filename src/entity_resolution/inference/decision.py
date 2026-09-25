"""Stages 6-7: turn calibrated pair probabilities into per-reference match sets.

Input frames have one row per (source1_entity_id, candidate_entity_id) with a
`probability` column. Every function returns a row subset of its input, so no
stage can invent a pair the matcher did not score.

Order in `decide`: singleton valve -> global assignment -> precision guards.
Guards run last so the final output is always a subset of the unguarded output
(a guard rejecting A's claim on X never hands X to another reference).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from entity_resolution.evaluation.scorer import score

REF, CAND = 'source1_entity_id', 'candidate_entity_id'


def singleton_valve(pairs, threshold):
    """References whose best candidate is below threshold get an empty prediction.

    Otherwise keep every candidate at or above the threshold.
    """
    best = pairs.groupby(REF, sort=False).probability.transform('max')
    return pairs[(best >= threshold) & (pairs.probability >= threshold)]


def assign_candidates(pairs):
    """Greedy global assignment: each S2/S3 candidate goes to at most one reference.

    When several references claim the same candidate, the highest probability wins;
    exact ties go to the lexicographically smallest reference ID (deterministic).
    Greedy, not Hungarian: O(n log n) in the number of accepted pairs.
    """
    order = pairs.sort_values(['probability', REF], ascending=[False, True], kind='stable')
    return order.drop_duplicates(CAND, keep='first').sort_index()


@dataclass(frozen=True)
class CorroborationRule:
    """An accepted pair needs address corroboration when both addresses exist.

    conflict: both addresses present, and address TF-IDF cosine AND address
    token-sort ratio are below their floors -> the match rests on name alone -> reject.
    missing: either address genuinely missing -> keep only if probability is high.
    """
    min_address_cosine: float
    min_address_token_sort: float
    missing_address_min_probability: float


# Chosen on US out-of-fold predictions only (India untouched): the floors sit below
# the 0.1st percentile of US true matches' address token-sort (0.441); missing-address
# matches accepted at p >= 0.80 were 96% precise vs 78% for 0.68 <= p < 0.80.
DEFAULT_RULE = CorroborationRule(min_address_cosine=0.10, min_address_token_sort=0.45,
                                 missing_address_min_probability=0.80)


def address_missing(pairs):
    return (pairs.source1_address_missing.eq(1) | pairs.candidate_address_missing.eq(1)
            | pairs.address_tfidf_cosine.isna() | pairs.address_token_sort.isna())


def corroboration_guard(pairs, rule):
    missing = address_missing(pairs)
    conflict = (~missing & (pairs.address_tfidf_cosine < rule.min_address_cosine)
                & (pairs.address_token_sort < rule.min_address_token_sort))
    weak_missing = missing & (pairs.probability < rule.missing_address_min_probability)
    return pairs[~(conflict | weak_missing)]


def _country(values):
    return values.astype('string').str.strip().str.casefold()


def country_veto(pairs):
    """Reject any pair whose two known countries differ, regardless of score.

    An unknown (missing/blank) country on either side is not evidence of a
    mismatch, so it never triggers the veto.
    """
    left, right = _country(pairs.source1_country), _country(pairs.candidate_country)
    known = left.notna() & right.notna() & left.ne('') & right.ne('')
    return pairs[~(known & left.ne(right)).fillna(False).to_numpy(dtype=bool)]


def decide(pairs, threshold, rule=None, veto=True):
    accepted = assign_candidates(singleton_valve(pairs, threshold))
    if rule is not None:
        accepted = corroboration_guard(accepted, rule)
    if veto:
        accepted = country_veto(accepted)
    return accepted


def to_predictions(accepted, references):
    """dict[reference -> set]; every reference present, empty set when nothing accepted."""
    predictions = {ref: set() for ref in references}
    for ref, cand in zip(accepted[REF], accepted[CAND]):
        predictions[ref].add(cand)
    return predictions


def sweep_thresholds(pairs, truth, thresholds, decide_fn):
    """Macro F0.5 (Swara's scorer) over every reference in `truth`, per threshold."""
    rows = [(t, score(to_predictions(decide_fn(pairs, t), truth), truth)) for t in thresholds]
    return pd.DataFrame(rows, columns=['threshold', 'f05'])


def best_threshold(sweep):
    """Highest F0.5; ties resolved toward the higher (more precise) threshold."""
    top = sweep.f05.max()
    return float(sweep.loc[np.isclose(sweep.f05, top), 'threshold'].max()), float(top)
