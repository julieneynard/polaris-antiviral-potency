"""masked_mae must match sklearn exactly and correctly drop NaN labels.

No network access: these tests only exercise pure functions on synthetic arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from src.evaluate import masked_mae


def test_masked_mae_matches_sklearn_with_no_nans():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.1, size=50)

    assert masked_mae(y_true, y_pred) == pytest.approx(mean_absolute_error(y_true, y_pred))


def test_masked_mae_ignores_nan_rows_in_y_true():
    y_true = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    y_pred = np.array([1.5, 99.0, 2.5, -99.0, 4.5])  # predictions at NaN rows are garbage

    result = masked_mae(y_true, y_pred)

    # Only rows 0, 2, 4 have a real label; the NaN rows' huge "predictions" must not
    # contribute, or the result would be dominated by the 99.0 / -99.0 garbage values.
    expected = mean_absolute_error([1.0, 3.0, 5.0], [1.5, 2.5, 4.5])
    assert result == pytest.approx(expected)
    assert result < 1.0  # sanity bound: would be ~50 if the NaN rows leaked in


def test_masked_mae_raises_when_all_labels_are_nan():
    y_true = np.array([np.nan, np.nan])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        masked_mae(y_true, y_pred)
