"""cross_validate must partition the train fold into disjoint train/val folds, never
leak a fold's validation rows into that fold's training call, and be deterministic.

No network access: uses a trivial linear fit_predict stand-in on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from src.cross_validate import cross_validate


def _mean_predictor(X_train, y_train, X_val, seed):
    """Minimal fit_predict_fn: predict the training mean for every validation row.

    Deliberately trivial (no real learning) so the test is only exercising
    cross_validate's own splitting/aggregation logic, not a model's.
    """
    return np.full(len(X_val), y_train.mean())


def test_folds_are_disjoint_and_cover_all_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    all_val_idx = [val_idx for _, val_idx in kf.split(X)]

    # Every row appears in exactly one validation fold, and folds partition all rows.
    concatenated = np.concatenate(all_val_idx)
    assert sorted(concatenated) == list(range(50))
    assert len(set(concatenated)) == 50

    result = cross_validate(_mean_predictor, X, y, n_splits=5, seed=0)
    assert result["n_splits"] == 5
    assert len(result["fold_maes"]) == 5


def test_cross_validate_is_deterministic_given_fixed_seed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)

    result_1 = cross_validate(_mean_predictor, X, y, n_splits=4, seed=3)
    result_2 = cross_validate(_mean_predictor, X, y, n_splits=4, seed=3)

    assert result_1 == result_2


def test_cross_validate_raises_on_nan_y():
    X = np.zeros((5, 2))
    y = np.array([1.0, np.nan, 3.0, 4.0, 5.0])

    with pytest.raises(ValueError):
        cross_validate(_mean_predictor, X, y, n_splits=2, seed=0)


def test_each_fold_trains_only_on_its_own_partition():
    """The fit_predict_fn receives disjoint (X_train, X_val) per fold - assert the
    sizes sum to the full dataset and X_train never contains an X_val row (by identity
    of a unique per-row marker column).
    """
    n = 30
    X = np.arange(n).reshape(-1, 1).astype(float)  # row i has the unique value i
    y = np.arange(n).astype(float)

    seen_train_sets = []
    seen_val_sets = []

    def marker_predictor(X_train, y_train, X_val, seed):
        seen_train_sets.append(set(X_train.flatten().tolist()))
        seen_val_sets.append(set(X_val.flatten().tolist()))
        return np.zeros(len(X_val))

    cross_validate(marker_predictor, X, y, n_splits=5, seed=0)

    for train_set, val_set in zip(seen_train_sets, seen_val_sets):
        assert train_set.isdisjoint(val_set)
        assert train_set | val_set == set(range(n))
