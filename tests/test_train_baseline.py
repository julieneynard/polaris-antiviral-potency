"""Leakage and determinism checks for the baseline models.

Two complementary styles of leakage test:
  1. Direct: independently reimplement "fit StandardScaler on train only" and assert
     fit_predict_ridge matches it exactly - proves the train-only-fit claim, not just
     a symptom of it.
  2. Behavioral/black-box: predictions for a given test row must be unchanged if the
     *other* rows in the test set change. If a scaler or model were (incorrectly) fit
     on train+test combined, adding/changing unrelated test rows would shift its
     statistics and change every prediction - so this catches a leakage regression
     even if the implementation changes, without assuming how it fits.

No network access: everything runs on small synthetic arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.train_baseline import fit_predict_random_forest, fit_predict_ridge


@pytest.fixture
def synthetic_regression_data():
    rng = np.random.default_rng(42)
    n_train, n_test, n_features = 60, 15, 20
    X_train = rng.normal(size=(n_train, n_features))
    true_coef = rng.normal(size=n_features)
    y_train = X_train @ true_coef + rng.normal(scale=0.1, size=n_train)
    X_test = rng.normal(size=(n_test, n_features))
    return X_train, y_train, X_test


def test_ridge_scaler_is_fit_on_train_only(synthetic_regression_data):
    X_train, y_train, X_test = synthetic_regression_data

    # Independently reimplement the leakage-free claim: scaler fit on train only.
    scaler = StandardScaler().fit(X_train)
    expected_model = Ridge(alpha=1.0, random_state=0)
    expected_model.fit(scaler.transform(X_train), y_train)
    expected_pred = expected_model.predict(scaler.transform(X_test))

    actual_pred = fit_predict_ridge(X_train, y_train, X_test, seed=0)

    np.testing.assert_allclose(actual_pred, expected_pred)


def test_ridge_scaler_statistics_are_unaffected_by_test_set_content(synthetic_regression_data):
    """If the scaler were (incorrectly) fit on train+test, its mean/std - and therefore
    every prediction - would shift when the test set's content changes. Predictions for
    a fixed set of test rows must be identical regardless of what other test rows
    accompany them.
    """
    X_train, y_train, X_test = synthetic_regression_data
    rng = np.random.default_rng(1)

    pred_small = fit_predict_ridge(X_train, y_train, X_test, seed=0)

    # Same test rows, but with extra, very differently-scaled rows appended - this
    # would shift a leaky (train+test) scaler's mean/std substantially.
    extra_rows = rng.normal(loc=50, scale=10, size=(5, X_test.shape[1]))
    X_test_extended = np.vstack([X_test, extra_rows])
    pred_extended = fit_predict_ridge(X_train, y_train, X_test_extended, seed=0)

    np.testing.assert_allclose(pred_small, pred_extended[: len(pred_small)])


def test_random_forest_predictions_are_unaffected_by_test_set_content(synthetic_regression_data):
    X_train, y_train, X_test = synthetic_regression_data
    rng = np.random.default_rng(2)

    pred_small = fit_predict_random_forest(X_train, y_train, X_test, seed=0)

    extra_rows = rng.normal(loc=50, scale=10, size=(5, X_test.shape[1]))
    X_test_extended = np.vstack([X_test, extra_rows])
    pred_extended = fit_predict_random_forest(X_train, y_train, X_test_extended, seed=0)

    np.testing.assert_array_equal(pred_small, pred_extended[: len(pred_small)])


def test_random_forest_is_bitwise_deterministic(synthetic_regression_data):
    """Regression test for the n_jobs=1 fix: parallel tree aggregation (n_jobs=-1) can
    change floating-point summation order across runs even with a fixed random_state.
    """
    X_train, y_train, X_test = synthetic_regression_data

    pred_1 = fit_predict_random_forest(X_train, y_train, X_test, seed=7)
    pred_2 = fit_predict_random_forest(X_train, y_train, X_test, seed=7)

    np.testing.assert_array_equal(pred_1, pred_2)


def test_ridge_is_bitwise_deterministic(synthetic_regression_data):
    X_train, y_train, X_test = synthetic_regression_data

    pred_1 = fit_predict_ridge(X_train, y_train, X_test, seed=7)
    pred_2 = fit_predict_ridge(X_train, y_train, X_test, seed=7)

    np.testing.assert_array_equal(pred_1, pred_2)
