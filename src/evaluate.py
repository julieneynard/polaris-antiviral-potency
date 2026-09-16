"""Evaluation harness matching the official Polaris metric logic for this benchmark.

Verified against polaris/evaluate/_metric.py (polaris-lib==0.13.0, installed in this
project's .venv):
  - The registered "mean_absolute_error" metric is exactly `sklearn.metrics.mean_absolute_error`.
  - Rows where y_true is NaN are masked out before scoring (`mask_index`).
  - Multi-task aggregation is explicitly NOT supported by polaris
    (`NotImplementedError("Multitask metrics are not yet supported...")`), so each
    target is scored independently, not averaged together. We reproduce that
    per-target, NaN-masked scoring here.

Used for both tasks (potency, ADMET). For ADMET this is a deliberate simplification:
see src/data.py for why the official log-transformed/clipped ADMET metric isn't
reproduced here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE over the subset of rows where y_true is not NaN, matching polaris' mask_index."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        raise ValueError("No non-NaN labels to score against.")
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))
