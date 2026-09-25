"""Stage 5 leave-one-country-out validation on the scored mock handoff.

Train on country == 'US' references only; choose ONE global threshold by macro
F0.5 (Swara's scorer) on country == 'India' references only. Country is the
Source-1 reference's country. Both frozen mock splits are pooled: LOCO replaces
the random split here, so India is never seen by the booster or the calibrator.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution.evaluation.scorer import score
from entity_resolution.features.baseline import ID_COLUMNS
from entity_resolution.inference.decision import best_threshold, sweep_thresholds, to_predictions
from entity_resolution.models.matcher import predict_proba, save_matcher, train_matcher

TRAIN_COUNTRY, HOLDOUT_COUNTRY = 'US', 'India'
THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 2)


def read_truth(paths):
    frames = [pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
              for p in paths]
    frame = pd.concat(frames, ignore_index=True)
    if not frame.source1_entity_id.is_unique:
        raise ValueError('Duplicate ground-truth references')
    return {r: set(m.split(',')) if m else set()
            for r, m in zip(frame.source1_entity_id, frame.matched_entity_ids)}


def load_handoff(handoff, normalized):
    """Pairs with features, label, and both sides' country; plus complete truth."""
    parts = []
    for split in ('train', 'validation'):
        features = pd.read_parquet(handoff / f'{split}_features.parquet')
        labels = pd.read_parquet(handoff / f'{split}_labels.parquet')
        parts.append(features.merge(labels, on=ID_COLUMNS, how='left', validate='one_to_one'))
    pairs = pd.concat(parts, ignore_index=True)
    if pairs.label.isna().any():
        raise ValueError('Unlabelled pairs in handoff')
    records = pd.concat([pd.read_parquet(normalized / f'train_source{i}.parquet',
                                         columns=['entity_id', 'country']) for i in (1, 2, 3)])
    country = records.set_index('entity_id').country
    pairs['source1_country'] = pairs.source1_entity_id.map(country)
    pairs['candidate_country'] = pairs.candidate_entity_id.map(country)
    truth = read_truth([handoff / 'train_ground_truth.tsv', handoff / 'validation_ground_truth.tsv'])
    s1_country = country[country.index.str.startswith('S1-')]
    return pairs, truth, s1_country


def country_truth(truth, s1_country, name):
    return {ref: t for ref, t in truth.items() if s1_country.get(ref) == name}


def reliability(prob, label, bins=(0, .1, .3, .5, .7, .9, 1.0001)):
    frame = pd.DataFrame({'p': prob, 'y': label, 'bin': pd.cut(prob, bins, right=False)})
    return frame.groupby('bin', observed=True).agg(pairs=('y', 'size'), mean_pred=('p', 'mean'),
                                                   positive_rate=('y', 'mean'))


def run(handoff, normalized, out):
    pairs, truth, s1_country = load_handoff(handoff, normalized)
    print('references by country:', s1_country.reindex(list(truth)).value_counts().to_dict())
    train = pairs[pairs.source1_country == TRAIN_COUNTRY].reset_index(drop=True)
    hold = pairs[pairs.source1_country == HOLDOUT_COUNTRY].reset_index(drop=True)
    hold_truth = country_truth(truth, s1_country, HOLDOUT_COUNTRY)
    print(f'train ({TRAIN_COUNTRY}): {len(train):,} pairs, {train.source1_entity_id.nunique():,} refs, '
          f'{int(train.label.sum()):,} positives')
    print(f'holdout ({HOLDOUT_COUNTRY}): {len(hold):,} pairs, {len(hold_truth):,} refs '
          f'({sum(not t for t in hold_truth.values())} true singletons), {int(hold.label.sum()):,} retrieved positives')

    model = train_matcher(train, train.label, train.source1_entity_id)
    hold['probability'] = predict_proba(model, hold)

    sweep = sweep_thresholds(hold, hold_truth, THRESHOLDS, lambda p, t: p[p.probability >= t])
    threshold, f05 = best_threshold(sweep)
    ceiling = score(to_predictions(hold[hold.label == 1], hold_truth), hold_truth)
    empty = score({}, hold_truth)
    at_half = sweep.loc[np.isclose(sweep.threshold, 0.5), 'f05'].item()

    print('\nIndia reliability (isotonic fitted on US only):')
    print(reliability(hold.probability.to_numpy(), hold.label.to_numpy()).round(3).to_string())
    print('\nthreshold sweep (every 0.05):')
    print(sweep[np.isclose((sweep.threshold * 100) % 5, 0)].round(4).to_string(index=False))
    print(f'\nall-empty baseline F0.5:              {empty:.4f}')
    print(f'F0.5 at threshold 0.50:               {at_half:.4f}')
    print(f'blocking ceiling (perfect matcher):   {ceiling:.4f}')
    print(f'SELECTED threshold {threshold:.2f} -> India macro F0.5 = {f05:.4f}')

    out.mkdir(parents=True, exist_ok=True)
    save_matcher(model, out / 'matcher_us.joblib')
    hold.to_parquet(out / 'india_scored_pairs.parquet', index=False)
    sweep.to_csv(out / 'india_threshold_sweep.tsv', sep='\t', index=False)
    summary = {'train_country': TRAIN_COUNTRY, 'holdout_country': HOLDOUT_COUNTRY,
               'threshold': threshold, 'india_f05': f05, 'india_f05_at_0.5': at_half,
               'india_blocking_ceiling_f05': ceiling, 'india_all_empty_f05': empty,
               'train_pairs': len(train), 'holdout_pairs': len(hold), 'holdout_refs': len(hold_truth)}
    (out / 'loco_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--handoff', type=Path, default=repo / 'artifacts/scored_mock_v2')
    parser.add_argument('--normalized', type=Path, default=repo / 'data/processed/normalized_mock')
    parser.add_argument('--output', type=Path, default=repo / 'artifacts/matcher_loco')
    args = parser.parse_args()
    run(args.handoff, args.normalized, args.output)


if __name__ == '__main__':
    main()
