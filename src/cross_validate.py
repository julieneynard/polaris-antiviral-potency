"""K-fold cross-validation on the official TRAIN fold, for internal model diagnostics.

This is NOT the headline metric. The headline number is always MAE on the official
chronological TEST fold (see scripts/run_baseline.py); that comparison against
prospective, out-of-time data is the whole point of this benchmark. CV here answers a
different, narrower question - "how much does this model's error vary depending on
which training rows it sees?" - by resampling only within the training fold, which sits
entirely before the test fold in time. It never touches the test fold, so it cannot
leak into or substitute for the headline evaluation; that's why leakage rule #2 (never
a random split for headline results) doesn't apply to it.

Leakage-free per fold, same as the full pipeline: whichever `fit_predict_*` function is
passed in (see src/train_baseline.py) fits any data-dependent transform (e.g. Ridge's
StandardScaler) on that fold's training partition only.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import KFold

from src.evaluate import masked_mae

FitPredictFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def cross_validate(
    fit_predict_fn: FitPredictFn,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Run K-fold CV on (X, y) and return per-fold and summary MAE.

    `y` must already be free of NaNs (filter to the rows with a measured label for
    the target being evaluated before calling this, as scripts/run_baseline.py does).
    """
    if np.isnan(y).any():
        raise ValueError("y contains NaN - filter to rows with a measured label first.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_maes = []
    for train_idx, val_idx in kf.split(X):
        y_pred = fit_predict_fn(X[train_idx], y[train_idx], X[val_idx], seed)
        fold_maes.append(masked_mae(y[val_idx], y_pred))

    fold_maes = np.array(fold_maes)
    return {
        "n_splits": n_splits,
        "fold_maes": [float(m) for m in fold_maes],
        "mean_mae": float(fold_maes.mean()),
        "std_mae": float(fold_maes.std()),
    }
