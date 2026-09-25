"""Stage 5 matcher: LightGBM on SCORED_FEATURE_COLUMNS, isotonic-calibrated.

Label 1 = candidate is in ground truth; 0 = blocking survivor not in ground truth.
Calibration folds are grouped by source1_entity_id so no reference's candidates
are split between the booster and the isotonic fit of the same fold.
"""
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

from entity_resolution.features.scored import SCORED_FEATURE_COLUMNS

LGBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    deterministic=True, force_row_wise=True, verbose=-1,
)


def feature_matrix(frame):
    """Ordered float32 feature block; NaNs stay NaN (LightGBM handles missing)."""
    missing = [c for c in SCORED_FEATURE_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f'Missing feature columns: {missing}')
    return frame[SCORED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def train_matcher(frame, labels, groups, n_splits=5, seed=0):
    X, y, groups = feature_matrix(frame), np.asarray(labels), np.asarray(groups)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Training labels must contain both classes 0 and 1')
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    base = LGBMClassifier(random_state=seed, n_jobs=4, **LGBM_PARAMS)
    return CalibratedClassifierCV(base, method='isotonic', cv=folds).fit(X, y)


def predict_proba(model, frame):
    return model.predict_proba(feature_matrix(frame))[:, 1]


def save_matcher(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_columns': SCORED_FEATURE_COLUMNS}, path)


def load_matcher(path):
    """Load only trusted, locally generated artifacts (joblib executes Python)."""
    bundle = joblib.load(path)
    if bundle.get('feature_columns') != SCORED_FEATURE_COLUMNS:
        raise ValueError('Matcher was trained on a different feature contract')
    return bundle['model']
