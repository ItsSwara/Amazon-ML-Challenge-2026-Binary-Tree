"""Stage 8: batched test inference -> output/matching_results.tsv + output/candidate_pairs.tsv.

Feature scoring is bounded by the batch (final assignment/verification is not):
  * candidate pairs (the blocker's candidate_pairs_scored.tsv) are streamed in
    chunks and cut on reference boundaries, so rank/gap features always see a
    complete group;
  * normalized records live in an on-disk SQLite store and only the IDs of the
    current batch are fetched;
  * only pairs that pass the singleton valve are kept, spilled to Parquet, and
    reloaded (three columns + guard columns) for the global assignment and guards.

candidate_pairs.tsv is exactly the set scored by the matcher, written as each
batch is scored; references the blocker returned nothing for get an empty row.
"""
import argparse
import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from entity_resolution.features.baseline import FeatureExtractor, ID_COLUMNS, join_pairs
from entity_resolution.features.scored import add_retrieval_features
from entity_resolution.inference.decision import (
    DEFAULT_RULE, REF, CAND, assign_candidates, corroboration_guard, country_veto, singleton_valve,
)
from entity_resolution.models.matcher import load_matcher, predict_proba

RECORD_COLUMNS = ['entity_id', 'name_norm', 'address_norm', 'country']
KEEP_COLUMNS = [REF, CAND, 'probability', 'address_tfidf_cosine', 'address_token_sort',
                'source1_address_missing', 'candidate_address_missing', 'source1_country', 'candidate_country']
MATCHING_HEADER = 'source1_entity_id\tmatched_entity_ids'
CANDIDATE_HEADER = 'source1_entity_id\tcandidate_entity_ids'


# --------------------------------------------------
# Disk-backed record lookup
# --------------------------------------------------

class RecordStore:
    """entity_id -> normalized name/address/country in SQLite; built once per split."""

    def __init__(self, path):
        self.conn = sqlite3.connect(path)

    @classmethod
    def build(cls, path, parquet_paths, batch_rows=200_000):
        path = Path(path)
        parquet_paths = [Path(p).resolve() for p in parquet_paths]
        manifest = cls._manifest(parquet_paths)
        expected = sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_paths)
        if path.exists():
            store = cls(path)
            try:
                saved = store.conn.execute('SELECT manifest FROM cache_metadata').fetchone()
                reusable = (saved is not None and saved[0] == manifest and
                            store.conn.execute('SELECT COUNT(*) FROM records').fetchone()[0] == expected)
            except sqlite3.DatabaseError:
                reusable = False  # Old row-count-only stores must be rebuilt once.
            if reusable:
                return store
            store.conn.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.building')
        tmp.unlink(missing_ok=True)
        conn = sqlite3.connect(tmp)
        try:
            conn.execute('PRAGMA journal_mode=OFF')
            conn.execute('PRAGMA synchronous=OFF')
            conn.execute('CREATE TABLE records (entity_id TEXT PRIMARY KEY, name_norm TEXT, '
                         'address_norm TEXT, country TEXT) WITHOUT ROWID')
            for p in parquet_paths:
                for batch in pq.ParquetFile(p).iter_batches(batch_rows, columns=RECORD_COLUMNS):
                    conn.executemany('INSERT INTO records VALUES (?, ?, ?, ?)',
                                     zip(*(batch.column(c).to_pylist() for c in RECORD_COLUMNS)))
                conn.commit()
            if cls._manifest(parquet_paths) != manifest:
                raise ValueError('Normalized files changed while building record store; retry with immutable inputs')
            conn.execute('CREATE TABLE cache_metadata (manifest TEXT NOT NULL)')
            conn.execute('INSERT INTO cache_metadata VALUES (?)', (manifest,))
            conn.commit()
        finally:
            conn.close()
        tmp.replace(path)  # Keep the previous cache until a complete rebuild exists.
        return cls(path)

    @staticmethod
    def _manifest(paths):
        sources = []
        for path in paths:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            sources.append({'path': str(path), 'sha256': digest.hexdigest()})
        return json.dumps({'version': 1, 'columns': RECORD_COLUMNS, 'sources': sources}, sort_keys=True)

    def lookup(self, ids):
        self.conn.execute('CREATE TEMP TABLE IF NOT EXISTS wanted (entity_id TEXT PRIMARY KEY)')
        self.conn.execute('DELETE FROM wanted')
        self.conn.executemany('INSERT OR IGNORE INTO wanted VALUES (?)', ((i,) for i in ids))
        rows = self.conn.execute('SELECT r.entity_id, r.name_norm, r.address_norm, r.country '
                                 'FROM wanted JOIN records r USING (entity_id)').fetchall()
        return pd.DataFrame(rows, columns=RECORD_COLUMNS)

    def close(self):
        self.conn.close()


# --------------------------------------------------
# Streaming complete reference groups
# --------------------------------------------------

def reference_batches(pairs_path, chunk_rows, max_candidates=50):
    """Yield DataFrames holding whole reference groups; a group split across
    reads is carried into the next batch. A reference that reappears after its
    group closed means the file is not grouped, and fails loudly."""
    if chunk_rows < 1 or max_candidates < 1:
        raise ValueError('chunk_rows and max_candidates must be positive')
    closed, carry = set(), None
    reader = pd.read_csv(pairs_path, sep='\t', dtype=str, keep_default_na=False,
                         quoting=csv.QUOTE_NONE, encoding='utf-8-sig', chunksize=chunk_rows)
    for chunk in reader:
        if list(chunk.columns) != ID_COLUMNS + ['score']:
            raise ValueError(f'Expected columns {ID_COLUMNS + ["score"]}, got {list(chunk.columns)}')
        if chunk.empty:
            continue
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        refs = chunk[REF]
        tail = refs.iloc[-1]
        is_tail = refs.eq(tail).to_numpy()
        if not is_tail[np.argmax(is_tail):].all():
            raise ValueError(f'Reference {tail} is not contiguous in {pairs_path}')
        carry, ready = chunk[is_tail], chunk[~is_tail]
        if len(carry) > max_candidates:
            raise ValueError(f'Reference {tail} exceeds {max_candidates} candidate limit')
        if len(ready):
            yield _close_groups(ready, closed, max_candidates)
    if carry is not None and len(carry):
        yield _close_groups(carry, closed, max_candidates)


def _close_groups(batch, closed, max_candidates=50):
    starts = batch[REF].ne(batch[REF].shift())
    group_ids = batch.loc[starts, REF]
    if not group_ids.is_unique or not closed.isdisjoint(group_ids):
        raise ValueError('Candidate file is not grouped by source1_entity_id')
    if batch.groupby(REF, sort=False).size().gt(max_candidates).any():
        raise ValueError(f'Reference exceeds {max_candidates} candidate limit')
    closed.update(group_ids)
    return batch.reset_index(drop=True)


# --------------------------------------------------
# Scoring one batch
# --------------------------------------------------

def score_batch(raw, store, extractor, model):
    scored = add_retrieval_features(raw)
    records = store.lookup(pd.unique(scored[ID_COLUMNS].to_numpy().ravel()))
    features = extractor.transform(join_pairs(scored, records))
    features[scored.columns[2:]] = scored[scored.columns[2:]].to_numpy()
    features['probability'] = predict_proba(model, features)
    country = records.set_index('entity_id').country
    features['source1_country'] = features[REF].map(country)
    features['candidate_country'] = features[CAND].map(country)
    return features


def candidate_lines(scored):
    ordered = scored.sort_values([REF, 'candidate_rank'], kind='stable')
    return [f'{ref}\t{",".join(group)}' for ref, group in
            ordered.groupby(REF, sort=False)[CAND]]


# --------------------------------------------------
# Output verification (independent of the official validator)
# --------------------------------------------------

def read_id_lists(path, header):
    with open(path, encoding='utf-8', newline='') as f:
        first = f.readline().rstrip('\r\n')
        if first != header:
            raise ValueError(f'{path}: header {first!r} != {header!r}')
        rows = {}
        for n, line in enumerate(f, start=2):
            ref, tab, rest = line.rstrip('\r\n').partition('\t')
            if not tab:
                raise ValueError(f'{path}:{n}: no tab')
            if ref in rows:
                raise ValueError(f'{path}: duplicate row for {ref}')
            ids = rest.split(',') if rest else []
            if len(ids) != len(set(ids)):
                raise ValueError(f'{path}: repeated ID in list for {ref}')
            if any(i.startswith('S1-') for i in ids):
                raise ValueError(f'{path}: S1 ID referenced as a match for {ref}')
            if not all(i.startswith(('S2-', 'S3-')) for i in ids):
                raise ValueError(f'{path}: non S2-/S3- ID for {ref}')
            rows[ref] = set(ids)
    return rows


def verify_outputs(matching_path, candidate_path, s1_ids):
    required = set(s1_ids)
    matching = read_id_lists(matching_path, MATCHING_HEADER)
    candidates = read_id_lists(candidate_path, CANDIDATE_HEADER)
    for name, rows in (('matching_results', matching), ('candidate_pairs', candidates)):
        if set(rows) != required:
            raise ValueError(f'{name}: {len(required - set(rows))} test S1 missing, '
                             f'{len(set(rows) - required)} rows for unknown S1')
    outside = [ref for ref, ids in matching.items() if not ids <= candidates[ref]]
    if outside:
        raise ValueError(f'{len(outside)} references have matches outside candidate_pairs, e.g. {outside[:3]}')
    owners = {}
    for ref, ids in matching.items():
        for i in ids:
            if owners.setdefault(i, ref) != ref:
                raise ValueError(f'{i} matched to both {owners[i]} and {ref}')
    return {'references': len(required), 'non_empty_matches': sum(bool(v) for v in matching.values()),
            'matched_ids': sum(map(len, matching.values())),
            'non_empty_candidates': sum(bool(v) for v in candidates.values()),
            'candidate_ids': sum(map(len, candidates.values()))}


# --------------------------------------------------
# Driver
# --------------------------------------------------

def run(pairs_path, normalized, split, model_path, extractor_path, threshold, out_dir, store_path,
        chunk_rows=50_000, rule=DEFAULT_RULE):
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [normalized / f'{split}_source{i}.parquet' for i in (1, 2, 3)]
    t0 = time.perf_counter()
    store = RecordStore.build(store_path, sources)
    print(f'record store ready: {store_path} ({time.perf_counter() - t0:.1f}s)', flush=True)
    s1_ids = pq.read_table(sources[0], columns=['entity_id']).column(0).to_pylist()
    if len(set(s1_ids)) != len(s1_ids):
        raise ValueError('Duplicate S1 entity IDs in test source 1')
    s1_known = set(s1_ids)
    model, extractor = load_matcher(model_path), FeatureExtractor.load(extractor_path)

    candidate_path = out_dir / 'candidate_pairs.tsv'
    matching_path = out_dir / 'matching_results.tsv'
    accepted_path = out_dir / 'accepted_pairs.parquet'
    seen, n_pairs, n_batches, writer = set(), 0, 0, None
    schema = None
    t0 = time.perf_counter()
    with open(candidate_path, 'w', encoding='utf-8', newline='\n') as cand_file:
        cand_file.write(CANDIDATE_HEADER + '\n')
        for raw in reference_batches(pairs_path, chunk_rows):
            unknown = set(raw[REF].unique()) - s1_known
            if unknown:
                raise ValueError(f'Candidate file references S1 not in test source 1: {sorted(unknown)[:3]}')
            scored = score_batch(raw, store, extractor, model)
            cand_file.write('\n'.join(candidate_lines(scored)) + '\n')
            seen.update(scored[REF].unique())
            kept = singleton_valve(scored, threshold)[KEEP_COLUMNS]
            table = pa.Table.from_pandas(kept, preserve_index=False, schema=schema)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(accepted_path, schema)
            writer.write_table(table)
            n_pairs += len(raw)
            n_batches += 1
            if n_batches % 20 == 0:
                rate = n_pairs / (time.perf_counter() - t0)
                print(f'  {len(seen):,} refs / {n_pairs:,} pairs scored ({rate:,.0f} pairs/s)', flush=True)
        for ref in s1_ids:
            if ref not in seen:
                cand_file.write(f'{ref}\t\n')
    if writer is not None:
        writer.close()
    store.close()
    score_seconds = time.perf_counter() - t0
    print(f'scored {n_pairs:,} pairs for {len(seen):,} refs in {score_seconds:.1f}s', flush=True)

    accepted = (pd.read_parquet(accepted_path) if writer is not None
                else pd.DataFrame(columns=KEEP_COLUMNS))
    after_valve = len(accepted)
    accepted = assign_candidates(accepted)
    after_assign = len(accepted)
    accepted = corroboration_guard(accepted, rule)
    after_corroboration = len(accepted)
    accepted = country_veto(accepted)
    print(f'decision: valve {after_valve:,} -> assignment {after_assign:,} -> corroboration '
          f'{after_corroboration:,} -> country veto {len(accepted):,} pairs', flush=True)

    ordered = accepted.sort_values([REF, 'probability'], ascending=[True, False], kind='stable')
    matches = ordered.groupby(REF, sort=False)[CAND].agg(','.join).to_dict()
    with open(matching_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(MATCHING_HEADER + '\n')
        for ref in s1_ids:
            f.write(f'{ref}\t{matches.get(ref, "")}\n')

    checks = verify_outputs(matching_path, candidate_path, s1_ids)
    countries = pq.read_table(sources[0], columns=['entity_id', 'country']).to_pandas()
    per_country = countries.assign(matched=countries.entity_id.isin(matches)).groupby('country').matched.agg(['size', 'mean'])
    summary = {'pairs_scored': n_pairs, 'references_with_candidates': len(seen), 'threshold': threshold,
               'after_valve': after_valve, 'after_assignment': after_assign,
               'after_corroboration': after_corroboration, 'after_country_veto': len(accepted),
               'verification': checks,
               'match_rate_by_country': {c: round(float(r['mean']), 4) for c, r in per_country.iterrows()},
               'score_seconds': round(score_seconds, 1),
               'pairs_per_second': round(n_pairs / score_seconds, 1) if score_seconds else None,
               'elapsed_seconds': round(time.perf_counter() - started, 1)}
    (out_dir / 'inference_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', choices=['mock', 'real'], default='real')
    parser.add_argument('--pairs', type=Path, help='default: output/<source>/test/candidate_pairs_scored.tsv')
    parser.add_argument('--out-dir', type=Path, help='default: output/ (real) or output/mock/submission (mock)')
    parser.add_argument('--model', type=Path, default=repo / 'artifacts/matcher_loco/matcher_us.joblib')
    parser.add_argument('--extractor', type=Path, default=repo / 'artifacts/scored_mock_v2/extractor.joblib')
    parser.add_argument('--threshold', type=float,
                        help='default: Stage 5 LOCO threshold from artifacts/matcher_loco/loco_summary.json')
    parser.add_argument('--chunk-rows', type=int, default=50_000)
    args = parser.parse_args()
    normalized = repo / 'data/processed' / ('normalized_mock' if args.source == 'mock' else 'normalized')
    pairs = args.pairs or repo / 'output' / args.source / 'test' / 'candidate_pairs_scored.tsv'
    out_dir = args.out_dir or (repo / 'output' if args.source == 'real' else repo / 'output/mock/submission')
    threshold = args.threshold
    if threshold is None:
        summary = json.loads((repo / 'artifacts/matcher_loco/loco_summary.json').read_text(encoding='utf-8'))
        threshold = summary['threshold']
    if not pairs.exists():
        raise SystemExit(f'Scored candidate file not found: {pairs}. Run the blocker for this split first.')
    store = repo / 'data/processed' / f'record_store_{args.source}_test.sqlite'
    run(pairs, normalized, 'test', args.model, args.extractor, threshold, out_dir, store, args.chunk_rows)


if __name__ == '__main__':
    main()
