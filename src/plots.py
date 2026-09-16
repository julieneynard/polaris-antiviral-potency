"""Plots for the results write-up. Consumes predictions/metrics already computed by the
run scripts (scripts/run_baseline.py, scripts/run_admet_baseline.py) - never refits a
model, so the plots can't drift from the numbers in results/*.json and the README.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, target_label: str, mae: float, out_path: Path) -> None:
    """Predicted vs. actual value on the official chronological test fold."""
    fig, ax = plt.subplots(figsize=(5, 5))

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05 or 0.3
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")

    ax.scatter(y_true, y_pred, alpha=0.6, s=25, edgecolor="none")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{target_label}\nRandomForest, official test fold (MAE = {mae:.3f})")
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cv_vs_test_grid(results: dict, out_path: Path, target_labels: dict[str, str] | None = None) -> None:
    """Small multiples: one subplot per target, each with its own MAE scale.

    Endpoints with very different natural units (e.g. ADMET's KSOL in uM vs. LogD,
    dimensionless) would be visually misleading on one shared axis, so each target gets
    its own subplot rather than being crammed into a single combined bar chart.
    """
    targets = list(results.keys())
    target_labels = target_labels or {}
    models = [("ridge", "Ridge"), ("random_forest", "RandomForest")]

    n_cols = min(3, len(targets))
    n_rows = math.ceil(len(targets) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    x = np.arange(len(models))
    width = 0.35

    for i, target in enumerate(targets):
        ax = axes[i // n_cols][i % n_cols]
        cv_means = [results[target][f"{key}_train_cv"]["mean_mae"] for key, _ in models]
        cv_stds = [results[target][f"{key}_train_cv"]["std_mae"] for key, _ in models]
        test_vals = [results[target][f"{key}_mae"] for key, _ in models]

        ax.bar(x - width / 2, cv_means, width, yerr=cv_stds, capsize=4, label="Train-fold CV MAE (mean ± std)")
        ax.bar(x + width / 2, test_vals, width, label="Official chronological test MAE")
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in models])
        ax.set_ylabel("MAE")
        ax.set_title(target_labels.get(target, target), fontsize=10)

    # Hide any unused subplot cells.
    for j in range(len(targets), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Train-fold CV vs. official chronological test MAE")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def applicability_domain_plot(
    similarities: np.ndarray,
    abs_errors: np.ndarray,
    target_label: str,
    pearson_r: float,
    out_path: Path,
) -> None:
    """Nearest-neighbor Tanimoto similarity (to the train set) vs. absolute error, with
    a linear trend line. A downward trend (negative r) means the model does worse on
    structurally novel test molecules - the applicability-domain effect.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.scatter(similarities, abs_errors, alpha=0.6, s=25, edgecolor="none")

    slope, intercept = np.polyfit(similarities, abs_errors, 1)
    x_line = np.array([similarities.min(), similarities.max()])
    ax.plot(x_line, slope * x_line + intercept, color="firebrick", linewidth=1.5, label="linear trend")

    ax.set_xlabel("Max Tanimoto similarity to nearest training molecule")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"{target_label}\nRandomForest, official test fold (Pearson r = {pearson_r:.2f})")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
