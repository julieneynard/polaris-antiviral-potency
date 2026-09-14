"""Leakage-free Morgan FP + RandomForest / Ridge baseline under the official chronological split.

Leakage-free by construction:
  - Featurization (Morgan fingerprints) is a fixed deterministic hash of structure -
    no statistics are estimated from data, so computing it on train and test
    separately is safe.
  - Ridge's only "fit" step besides the regression itself is feature standardization;
    the StandardScaler is `fit` on the training fold ONLY and merely `transform`-ed
    on the test fold (see `fit_ridge_leakage_free`). RandomForest needs no scaling.
  - Each of the two pIC50 targets is modeled independently on rows where that target
    is not NaN (the dataset is sparse: not every molecule has both assay results).
    Train and test rows are never mixed.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 0


def fit_predict_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int = RANDOM_SEED
) -> np.ndarray:
    # n_jobs=1 (not -1): parallel tree aggregation can vary float summation order
    # across runs, breaking bit-for-bit reproducibility even with a fixed random_state.
    model = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_ridge(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int = RANDOM_SEED
) -> np.ndarray:
    # Scaler is fit on train only, then applied (not re-fit) to test - no leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)
