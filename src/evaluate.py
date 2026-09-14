"""Evaluation harness matching the official Polaris metric logic for this benchmark.

Verified against polaris/evaluate/_metric.py (polaris-lib==0.13.0, installed in this
project's .venv):
  - The registered "mean_absolute_error" metric is exactly `sklearn.metrics.mean_absolute_error`.
  - Rows where y_true is NaN are masked out before scoring (`mask_index`).
  - Multi-task aggregation is explicitly NOT supported by polaris
    (`NotImplementedError("Multitask metrics are not yet supported...")`), so each
    pIC50 target (SARS-CoV-2 Mpro, MERS-CoV Mpro) is scored independently, not averaged
    together. We reproduce that per-target, NaN-masked scoring here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error

TARGET_COLS = {
    "pic50_sars_cov_2_mpro": "pIC50 (SARS-CoV-2 Mpro)",
    "pic50_mers_cov_mpro": "pIC50 (MERS-CoV Mpro)",
}


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE over the subset of rows where y_true is not NaN, matching polaris' mask_index."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        raise ValueError("No non-NaN labels to score against.")
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def evaluate_predictions(y_true_by_target: dict[str, np.ndarray], y_pred_by_target: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute per-target MAE, mirroring polaris' benchmark.evaluate() semantics."""
    results = {}
    for col, y_true in y_true_by_target.items():
        y_pred = y_pred_by_target[col]
        results[col] = masked_mae(y_true, y_pred)
    return results
