"""Versioned retrieval features and batch export for Yash's scored candidates."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from entity_resolution.features.baseline import FEATURE_COLUMNS, ID_COLUMNS, join_pairs

SCHEMA_VERSION = 'scored-baseline-v2'
RETRIEVAL_COLUMNS = ['blocking_score', 'candidate_rank', 'blocking_top1_top2_gap']
SCORED_FEATURE_COLUMNS = FEATURE_COLUMNS + RETRIEVAL_COLUMNS


def add_retrieval_features(pairs):
    """Require the COMPLETE candidate set per reference, before batching.

    Higher Yash score is better. Rank is one-based; ties use lexicographic ID.
    Top1-top2 gap is repeated for all candidates of a reference; undefined for
    fewer than two candidates. No fabricated score or singleton margin is used.
    """
    required = ID_COLUMNS + ['score']
    if not set(required).issubset(pairs):
        raise ValueError(f'Expected columns: {required}')
    out = pairs[required].reset_index(drop=True).copy()
    for col, prefixes in zip(ID_COLUMNS, [('S1-',), ('S2-', 'S3-')]):
        if not out[col].map(lambda x: isinstance(x, str) and x.startswith(prefixes)).all():
            raise ValueError(f'Invalid {col}')
    if out.duplicated(ID_COLUMNS).any():
        raise ValueError('Duplicate candidate pairs')
    # Python/NumPy float parsing round-trips the TSV decimal representation.
    # pd.to_numeric can round near-ties differently from the original float64.
    scores = out['score'].astype('float64')
    if not np.isfinite(scores).all() or not scores.between(0, 1.050001).all():
        raise ValueError('Yash score must be finite and in [0, 1.05] (roundoff tolerance 1e-6)')
    out['score'] = scores
    ranked = out.sort_values(['source1_entity_id', 'score', 'candidate_entity_id'],
                             ascending=[True, False, True], kind='stable').copy()
    ranked['candidate_rank'] = ranked.groupby('source1_entity_id', sort=False).cumcount() + 1
    top = ranked[ranked.candidate_rank <= 2]
    first = top[top.candidate_rank == 1].set_index('source1_entity_id')['score']
    second = top[top.candidate_rank == 2].set_index('source1_entity_id')['score']
    out['blocking_score'] = scores.astype('float32')
    out['candidate_rank'] = ranked['candidate_rank'].reindex(out.index).astype('float32')
    out['blocking_top1_top2_gap'] = out.source1_entity_id.map(first - second).astype('float32')
    return out[ID_COLUMNS + RETRIEVAL_COLUMNS]


def blocking_metrics(pairs, truth, references):
    """Link-level recall and macro recall over nonsingletons, including missed refs."""
    if not set(references).issubset(truth):
        raise ValueError('Missing ground truth for requested references')
    selected = pairs[pairs.source1_entity_id.isin(references)]
    retrieved = selected.groupby('source1_entity_id')['candidate_entity_id'].agg(set).to_dict()
    hits, total, recalls, complete = 0, 0, [], 0
    for ref in sorted(references):
        actual = truth[ref]
        found = len(actual & retrieved.get(ref, set()))
        hits += found
        total += len(actual)
        if actual:
            recalls.append(found / len(actual))
            complete += int(found == len(actual))
    counts = [len(retrieved.get(ref, set())) for ref in references]
    return {
        'references': len(references), 'pairs': len(selected),
        'true_links': total, 'retrieved_true_links': hits, 'missed_true_links': total - hits,
        'micro_recall': hits / total if total else None,
        'macro_recall_non_singletons': float(np.mean(recalls)) if recalls else None,
        'non_singleton_references': len(recalls), 'fully_retrieved_non_singletons': complete,
        'true_singletons': len(references) - len(recalls),
        'references_without_candidates': sum(count == 0 for count in counts),
        'candidates_min': min(counts, default=0), 'candidates_max': max(counts, default=0),
        'candidates_mean': float(np.mean(counts)) if counts else 0,
    }


def write_features(scored_pairs, records, extractor, path, batch_size):
    """Ranks/gaps must already be calculated on complete reference groups."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    schema = pa.schema([(key, pa.string()) for key in ID_COLUMNS] +
                       [(key, pa.float32()) for key in SCORED_FEATURE_COLUMNS])
    nulls = dict.fromkeys(SCORED_FEATURE_COLUMNS, 0)
    with pq.ParquetWriter(path, schema) as writer:
        for start in range(0, len(scored_pairs), batch_size):
            batch = scored_pairs.iloc[start:start + batch_size].reset_index(drop=True)
            features = extractor.transform(join_pairs(batch, records))
            features[RETRIEVAL_COLUMNS] = batch[RETRIEVAL_COLUMNS].astype('float32')
            writer.write_table(pa.Table.from_pandas(features, schema=schema, preserve_index=False))
            for col in nulls:
                nulls[col] += int(features[col].isna().sum())
    return {'rows': len(scored_pairs), 'null_counts': nulls}
