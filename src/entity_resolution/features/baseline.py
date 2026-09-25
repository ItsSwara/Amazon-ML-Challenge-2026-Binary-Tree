"""Offline baseline v1. Fit on training records only; transform never refits."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

ID_COLUMNS = ['source1_entity_id', 'candidate_entity_id']
FEATURE_COLUMNS = [
    f'{field}_{metric}' for field in ('name', 'address')
    for metric in ('jaro_winkler', 'token_sort', 'tfidf_cosine', 'idf_overlap')
] + [f'{side}_{field}_missing' for side in ('source1', 'candidate') for field in ('name', 'address')]


def join_pairs(pairs, records):
    """Validated many-to-one joins. Caller supplies normalized records for this batch.

    This API does not load the full dataset. Pair uniqueness is checked per batch;
    the producer must guarantee uniqueness across the complete candidate file.
    """
    pairs = pairs[ID_COLUMNS].copy()
    if pairs.isna().any().any() or pairs.duplicated().any():
        raise ValueError('Pair IDs must be present and unique')
    if not pairs.source1_entity_id.str.startswith('S1-').all():
        raise ValueError('Reference IDs must start S1-')
    if not pairs.candidate_entity_id.str.startswith(('S2-', 'S3-')).all():
        raise ValueError('Candidate IDs must start S2- or S3-')
    if records.entity_id.isna().any() or not records.entity_id.is_unique:
        raise ValueError('Normalized entity IDs must be present and unique')
    lookup = records.set_index('entity_id')[['name_norm', 'address_norm']]
    for side, key in zip(('source1', 'candidate'), ID_COLUMNS):
        missing = ~pairs[key].isin(lookup.index)
        if missing.any():
            raise ValueError(f'Unknown {key}: {pairs.loc[missing, key].iloc[0]}')
        selected = lookup.reindex(pairs[key]).reset_index(drop=True)
        for field in ('name', 'address'):
            pairs[f'{side}_{field}'] = selected[f'{field}_norm'].to_numpy()
    return pairs.reset_index(drop=True)


class FeatureExtractor:
    """Word TF-IDF cosine and IDF-weighted token Jaccard, plus fuzzy metrics.

    The fit corpus must be an explicitly selected training-only sample. No labels,
    test records, or validation reference records are used by the demo's fit step.
    Direct API callers own their split selection. max_features bounds vocabulary,
    not corpus size; use a documented training sample on memory-limited machines.
    """
    def __init__(self, max_features=100_000):
        self.max_features = max_features
        self.vectorizers = {}
        self.idfs = {}
        self.fitted = False

    def fit(self, records):
        if records.empty:
            raise ValueError('Training corpus is empty')
        for field in ('name', 'address'):
            texts = records[f'{field}_norm'].dropna().tolist()
            vectorizer = TfidfVectorizer(lowercase=False, tokenizer=str.split,
                                         token_pattern=None, max_features=self.max_features,
                                         dtype=np.float32)
            if any(text.strip() for text in texts):
                vectorizer.fit(texts)
                self.idfs[field] = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
                self.vectorizers[field] = vectorizer
            else:
                self.vectorizers[field] = None
                self.idfs[field] = {}
        self.fitted = True
        return self

    def transform(self, joined):
        if not self.fitted:
            raise ValueError('Fit or load the extractor before transform')
        out = joined[ID_COLUMNS].reset_index(drop=True).copy()
        for field in ('name', 'address'):
            left = joined[f'source1_{field}'].reset_index(drop=True)
            right = joined[f'candidate_{field}'].reset_index(drop=True)
            # Do not repeat raw placeholder normalization: a real name can normalize to "nan".
            lm = left.isna() | left.eq('').fillna(False)
            rm = right.isna() | right.eq('').fillna(False)
            out[f'source1_{field}_missing'] = lm.astype('float32')
            out[f'candidate_{field}_missing'] = rm.astype('float32')
            valid = ~(lm | rm)
            a, b = left[valid].tolist(), right[valid].tolist()
            for metric in ('jaro_winkler', 'token_sort', 'tfidf_cosine', 'idf_overlap'):
                out[f'{field}_{metric}'] = np.full(len(out), np.nan, dtype=np.float32)
            if not a:
                continue
            out.loc[valid, f'{field}_jaro_winkler'] = [JaroWinkler.normalized_similarity(x, y) for x, y in zip(a, b)]
            out.loc[valid, f'{field}_token_sort'] = [fuzz.token_sort_ratio(x, y) / 100 for x, y in zip(a, b)]
            vectorizer = self.vectorizers[field]
            cosine = np.zeros(len(a), dtype=np.float32)
            if vectorizer is not None:
                cosine = np.asarray(vectorizer.transform(a).multiply(vectorizer.transform(b)).sum(axis=1)).ravel()
            out.loc[valid, f'{field}_tfidf_cosine'] = np.clip(cosine, 0, 1)
            weights = self.idfs[field]
            overlap = []
            for x, y in zip(a, b):
                xs, ys = set(x.split()), set(y.split())
                denominator = sum(float(weights.get(t, 0)) for t in sorted(xs | ys))
                numerator = sum(float(weights.get(t, 0)) for t in sorted(xs & ys))
                overlap.append(numerator / denominator if denominator else 0)
            out.loc[valid, f'{field}_idf_overlap'] = overlap
        out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype('float32')
        return out[ID_COLUMNS + FEATURE_COLUMNS]

    def save(self, path):
        if not self.fitted:
            raise ValueError('Cannot save an unfitted extractor')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """Load only trusted, locally generated artifacts (joblib executes Python)."""
        model = joblib.load(path)
        if not isinstance(model, FeatureExtractor) or not model.fitted:
            raise ValueError('Not a fitted feature extractor')
        return model
