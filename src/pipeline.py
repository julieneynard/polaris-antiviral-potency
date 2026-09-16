"""Shared end-to-end pipeline: official chronological split -> Morgan FP features ->
5-fold CV on the train fold (diagnostic) -> fit on full train, evaluate on official
test (headline) -> save metrics JSON + plots.

Used by both scripts/run_baseline.py (potency) and scripts/run_admet_baseline.py
(ADMET) - the two tasks differ only in dataset, target columns, and (for ADMET) that
the official metric transform isn't reproducible (see src/data.py), so both are scored
with the same plain masked MAE.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.applicability_domain import applicability_domain_summary, nearest_neighbor_similarity
from src.cross_validate import cross_validate
from src.data import ChronologicalSplit, official_chronological_split
from src.evaluate import masked_mae
from src.features import featurize
from src.plots import PLOTS_DIR, applicability_domain_plot, cv_vs_test_grid, parity_plot
from src.train_baseline import RANDOM_SEED, fit_predict_random_forest, fit_predict_ridge

N_CV_SPLITS = 5


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_baseline_pipeline(
    *,
    task_name: str,
    dataset_id: str,
    load_dataframe_fn: Callable[[], pd.DataFrame],
    target_cols: list[str],
    results_path: Path,
    target_labels: dict[str, str] | None = None,
    n_cv_splits: int = N_CV_SPLITS,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run the full leakage-free baseline pipeline for one task and save results + plots."""
    set_all_seeds(seed)
    target_labels = target_labels or {col: col for col in target_cols}

    print(f"Loading {dataset_id} ...")
    df = load_dataframe_fn()
    split: ChronologicalSplit = official_chronological_split(df)
    print(f"Official chronological split: {len(split.train)} train / {len(split.test)} test")

    print("Featurizing SMILES with Morgan fingerprints (radius=2, 2048 bits) ...")
    X_train_all = featurize(split.train["cxsmiles"].tolist())
    X_test_all = featurize(split.test["cxsmiles"].tolist())

    results: dict[str, dict] = {}

    for target_col in target_cols:
        target_label = target_labels[target_col]
        y_train_full = split.train[target_col].to_numpy(dtype=float)
        y_test_full = split.test[target_col].to_numpy(dtype=float)

        # Train only on rows where this target is measured (both datasets are sparse:
        # not every molecule was assayed for every endpoint/target).
        train_mask = ~np.isnan(y_train_full)
        X_train, y_train = X_train_all[train_mask], y_train_full[train_mask]

        test_mask = ~np.isnan(y_test_full)
        X_test, y_test = X_test_all[test_mask], y_test_full[test_mask]

        print(f"\n[{target_label}] train n={train_mask.sum()}, test n={test_mask.sum()}")

        print(f"  Running {n_cv_splits}-fold CV on the train fold (diagnostic, not the headline number) ...")
        ridge_cv = cross_validate(fit_predict_ridge, X_train, y_train, n_splits=n_cv_splits, seed=seed)
        rf_cv = cross_validate(fit_predict_random_forest, X_train, y_train, n_splits=n_cv_splits, seed=seed)
        print(f"  Ridge         train-fold CV MAE: {ridge_cv['mean_mae']:.4f} +/- {ridge_cv['std_mae']:.4f}")
        print(f"  RandomForest  train-fold CV MAE: {rf_cv['mean_mae']:.4f} +/- {rf_cv['std_mae']:.4f}")

        ridge_pred = fit_predict_ridge(X_train, y_train, X_test, seed=seed)
        rf_pred = fit_predict_random_forest(X_train, y_train, X_test, seed=seed)

        ridge_mae = masked_mae(y_test, ridge_pred)
        rf_mae = masked_mae(y_test, rf_pred)

        print(f"  Ridge         official test MAE: {ridge_mae:.4f}")
        print(f"  RandomForest  official test MAE: {rf_mae:.4f}")

        results[target_col] = {
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "ridge_mae": ridge_mae,
            "random_forest_mae": rf_mae,
            "ridge_train_cv": ridge_cv,
            "random_forest_train_cv": rf_cv,
        }

        parity_path = PLOTS_DIR / f"{task_name}_parity_{_slugify(target_col)}.png"
        parity_plot(y_test, rf_pred, target_label, rf_mae, parity_path)
        print(f"  Saved parity plot to {parity_path}")

        print("  Computing applicability-domain diagnostic (nearest-neighbor Tanimoto similarity) ...")
        train_smiles = split.train["cxsmiles"].to_numpy()[train_mask].tolist()
        test_smiles = split.test["cxsmiles"].to_numpy()[test_mask].tolist()
        similarities = nearest_neighbor_similarity(train_smiles, test_smiles)
        abs_errors = np.abs(y_test - rf_pred)
        ad_summary = applicability_domain_summary(similarities, abs_errors)
        print(f"  Similarity-vs-error Pearson r: {ad_summary['similarity_error_pearson_r']:.3f}")
        results[target_col]["random_forest_applicability_domain"] = ad_summary

        ad_path = PLOTS_DIR / f"{task_name}_ad_{_slugify(target_col)}.png"
        applicability_domain_plot(
            similarities, abs_errors, target_label, ad_summary["similarity_error_pearson_r"], ad_path
        )
        print(f"  Saved applicability-domain plot to {ad_path}")

    results_path.parent.mkdir(exist_ok=True, parents=True)
    with open(results_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_id,
                "split": "official chronological (Set column: Train/Test)",
                "featurization": "Morgan fingerprint (radius=2, 2048 bits)",
                "seed": seed,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {results_path}")

    grid_path = PLOTS_DIR / f"{task_name}_cv_vs_test_mae.png"
    cv_vs_test_grid(results, grid_path, target_labels=target_labels)
    print(f"Saved CV-vs-test grid to {grid_path}")

    return results


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
