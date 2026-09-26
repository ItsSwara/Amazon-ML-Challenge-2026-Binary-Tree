"""Synthetic streaming integration benchmark; NOT a model-quality benchmark.

Run from the repository root. Default: 2,000,000 pairs, 40,000 references, 50 shared
targets. Exercises SQLite lookup, complete groups, actual feature extraction,
calibrated LightGBM prediction and final exports. Shared targets keep the fixture
compact; this does not simulate 10 million distinct records/index memory.
"""
import argparse
import ctypes
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd

from entity_resolution.features.baseline import FeatureExtractor
from entity_resolution.features.scored import SCORED_FEATURE_COLUMNS
from entity_resolution.inference.run_inference import run
from entity_resolution.models.matcher import save_matcher, train_matcher


def peak_rss_bytes():
    if os.name == 'nt':
        class MemoryCounters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                    'PagefileUsage', 'PeakPagefileUsage')]
        counter = MemoryCounters()
        counter.cb = ctypes.sizeof(counter)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
        if not get_memory(get_process(), ctypes.byref(counter), counter.cb):
            raise ctypes.WinError()
        return counter.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=int, default=40_000)
    parser.add_argument('--chunk-rows', type=int, default=50_003,
                        help='Intentionally not divisible by 50, to split reference groups at reads')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.references < 2 or args.chunk_rows < 1:
        raise ValueError('Need at least 2 references and a positive chunk size')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    norm = out / 'normalized'
    norm.mkdir()
    names = ['alpha shop', 'beta engineering', 'gamma cafe', 'delta services', 'epsilon trading']

    def records(ids):
        return pd.DataFrame({'entity_id': ids,
                             'name_norm': [names[i % 5] for i in range(len(ids))],
                             'address_norm': [None if i % 7 == 0 else f'{i % 17} main road' for i in range(len(ids))],
                             'country': ['US'] * len(ids)})

    s1 = records([f'S1-{i:07}' for i in range(args.references)])
    s2 = records([f'S2-{i:03}' for i in range(25)])
    s3 = records([f'S3-{i:03}' for i in range(25)])
    for i, frame in enumerate([s1, s2, s3], 1):
        frame.to_parquet(norm / f'test_source{i}.parquet', index=False)
    # Only synthetic data: classifier learns a toy signal so this measures execution,
    # not matching quality. No real test text is fitted here.
    extractor = FeatureExtractor().fit(pd.concat([s1.head(100), s2, s3]))
    extractor.save(out / 'extractor.joblib')
    rng = np.random.default_rng(17)
    toy = pd.DataFrame(rng.random((600, len(SCORED_FEATURE_COLUMNS))).astype('float32'),
                       columns=SCORED_FEATURE_COLUMNS)
    labels = (toy.name_token_sort > 0.6).astype(int)
    model = train_matcher(toy, labels, np.arange(len(toy)) // 10)
    save_matcher(model, out / 'matcher.joblib')
    pairs = out / 'pairs.tsv'
    targets = list(s2.entity_id) + list(s3.entity_id)
    with pairs.open('w', encoding='utf-8', newline='') as handle:
        handle.write('source1_entity_id\tcandidate_entity_id\tscore\n')
        for ref in s1.entity_id:
            for j, target in enumerate(targets):
                handle.write(f'{ref}\t{target}\t{1.0 - j / 100:.2f}\n')
    print(f'Fixture ready: {args.references * 50:,} pairs; starting real inference functions', flush=True)
    result = run(pairs, norm, 'test', out / 'matcher.joblib', out / 'extractor.joblib',
                 0.5, out / 'inference', out / 'records.sqlite', args.chunk_rows)
    assert result['pairs_scored'] == args.references * 50
    assert result['verification']['references'] == args.references
    assert result['verification']['candidate_ids'] == args.references * 50
    report = {'kind': 'synthetic_streaming_stress_not_accuracy_or_full_dataset_readiness',
              'references': args.references, 'unique_targets': 50,
              'pairs': args.references * 50, 'chunk_rows': args.chunk_rows,
              'peak_process_rss_bytes': peak_rss_bytes(),
              'total_seconds': round(time.perf_counter() - started, 3),
              'platform': platform.platform(), 'python': platform.python_version(),
              'inference': result,
              'limitations': 'Only 50 shared targets; does not represent full record diversity, blocker memory or accuracy.'}
    (out / 'stress_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
