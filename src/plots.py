"""Plots for the results write-up. Consumes predictions/metrics already computed by
scripts/run_baseline.py - never refits a model, so the plots can't drift from the
numbers reported in results/baseline_metrics.json and the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, target_label: str, mae: float, out_path: Path) -> None:
    """Predicted vs. actual pIC50 on the official chronological test fold."""
    fig, ax = plt.subplots(figsize=(5, 5))

    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x")

    ax.scatter(y_true, y_pred, alpha=0.6, s=25, edgecolor="none")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title(f"{target_label}\nRandomForest, official test fold (MAE = {mae:.3f})")
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cv_vs_test_bar_chart(results: dict, out_path: Path) -> None:
    """Grouped bars: train-fold CV MAE (mean +/- std) vs. official test MAE, per model per target."""
    targets = list(results.keys())
    models = [("ridge", "Ridge"), ("random_forest", "RandomForest")]

    fig, ax = plt.subplots(figsize=(7, 5))
    n_groups = len(targets) * len(models)
    x = np.arange(n_groups)
    width = 0.35

    cv_means, cv_stds, test_vals, labels = [], [], [], []
    for target in targets:
        for key, name in models:
            cv = results[target][f"{key}_train_cv"]
            cv_means.append(cv["mean_mae"])
            cv_stds.append(cv["std_mae"])
            test_vals.append(results[target][f"{key}_mae"])
            short_target = "SARS-CoV-2" if "SARS" in target else "MERS-CoV"
            labels.append(f"{name}\n{short_target}")

    ax.bar(x - width / 2, cv_means, width, yerr=cv_stds, capsize=4, label="Train-fold CV MAE (mean ± std)")
    ax.bar(x + width / 2, test_vals, width, label="Official chronological test MAE")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAE (pIC50 units)")
    ax.set_title("Train-fold CV vs. official chronological test MAE")
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
