"""Stages 6-7 on the LOCO India holdout: effect of each decision step, with Swara's scorer."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution.evaluation.scorer import score
from entity_resolution.inference.decision import (
    DEFAULT_RULE, assign_candidates, best_threshold, corroboration_guard, country_veto, decide,
    singleton_valve, sweep_thresholds, to_predictions,
)
from entity_resolution.models.loco import THRESHOLDS, read_truth


def removed(before, after):
    gone = before.loc[before.index.difference(after.index)]
    return f'removed {len(gone):>3} pairs (true matches {int(gone.label.sum())}, false {int((gone.label == 0).sum())})'


def run(loco_dir, handoff):
    summary = json.loads((loco_dir / 'loco_summary.json').read_text(encoding='utf-8'))
    t = summary['threshold']
    pairs = pd.read_parquet(loco_dir / 'india_scored_pairs.parquet')
    truth = read_truth([handoff / 'train_ground_truth.tsv', handoff / 'validation_ground_truth.tsv'])
    truth = {ref: truth[ref] for ref in pairs.source1_entity_id.unique()}
    f05 = lambda accepted: score(to_predictions(accepted, truth), truth)

    plain = pairs[pairs.probability >= t]
    print(f'India holdout, threshold {t:.2f}: {len(truth)} refs, {len(pairs):,} scored pairs')
    print(f'  plain threshold                  : {len(plain):>5} pairs  F0.5 {f05(plain):.4f}')
    valve = singleton_valve(pairs, t)
    emptied = len(truth) - valve.source1_entity_id.nunique()
    print(f'  6a singleton valve               : {len(valve):>5} pairs  F0.5 {f05(valve):.4f}  '
          f'({emptied} refs predicted empty; true singletons {sum(not v for v in truth.values())})')
    contested = valve.candidate_entity_id.duplicated(keep=False)
    print(f'     candidates claimed by >1 ref  : {valve.loc[contested, "candidate_entity_id"].nunique()}')
    assigned = assign_candidates(valve)
    print(f'  6b global assignment             : {len(assigned):>5} pairs  F0.5 {f05(assigned):.4f}  {removed(valve, assigned)}')
    corroborated = corroboration_guard(assigned, DEFAULT_RULE)
    print(f'  7a corroboration {DEFAULT_RULE.min_address_cosine}/{DEFAULT_RULE.min_address_token_sort}/'
          f'{DEFAULT_RULE.missing_address_min_probability}  : {len(corroborated):>5} pairs  F0.5 {f05(corroborated):.4f}  '
          f'{removed(assigned, corroborated)}')
    vetoed = country_veto(corroborated)
    print(f'  7b country veto                  : {len(vetoed):>5} pairs  F0.5 {f05(vetoed):.4f}  {removed(corroborated, vetoed)}')
    cross = (pairs.source1_country != pairs.candidate_country)
    print(f'     cross-country among ALL scored pairs: {int(cross.sum()):,} '
          f'(true matches {int(pairs[cross].label.sum())}); max probability {pairs.loc[cross, "probability"].max():.3f}')

    full = sweep_thresholds(pairs, truth, THRESHOLDS, lambda p, x: decide(p, x, DEFAULT_RULE))
    best_t, best_f = best_threshold(full)
    print(f'\nfull-pipeline sweep: best threshold {best_t:.2f} -> F0.5 {best_f:.4f} '
          f'(Stage 5 threshold {t:.2f} with full pipeline -> {full.loc[np.isclose(full.threshold, t), "f05"].item():.4f})')


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loco', type=Path, default=repo / 'artifacts/matcher_loco')
    parser.add_argument('--handoff', type=Path, default=repo / 'artifacts/scored_mock_v2')
    args = parser.parse_args()
    run(args.loco, args.handoff)


if __name__ == '__main__':
    main()
