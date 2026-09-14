"""End-to-end: load Polaris ASAP Potency data -> official chronological split ->
Morgan FP features -> Ridge & RandomForest baselines -> MAE under the official metric logic.

Run with:
    uv run python scripts/run_baseline.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_potency_dataframe, official_chronological_split
from src.evaluate import masked_mae
from src.features import featurize
from src.train_baseline import RANDOM_SEED, fit_predict_random_forest, fit_predict_ridge

TARGET_COLUMNS = {
    "pIC50 (SARS-CoV-2 Mpro)": "pic50_sars_cov_2_mpro",
    "pIC50 (MERS-CoV Mpro)": "pic50_mers_cov_mpro",
}

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "baseline_metrics.json"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_all_seeds(RANDOM_SEED)

    print(f"Loading {'-'.join(['asap-discovery', 'antiviral-potency-2025-unblinded'])} ...")
    df = load_potency_dataframe()
    split = official_chronological_split(df)
    print(f"Official chronological split: {len(split.train)} train / {len(split.test)} test")

    print("Featurizing SMILES with Morgan fingerprints (radius=2, 2048 bits) ...")
    X_train_all = featurize(split.train["cxsmiles"].tolist())
    X_test_all = featurize(split.test["cxsmiles"].tolist())

    results: dict[str, dict[str, float]] = {}

    for target_label, target_col in TARGET_COLUMNS.items():
        y_train_full = split.train[target_col].to_numpy(dtype=float)
        y_test_full = split.test[target_col].to_numpy(dtype=float)

        # Train only on rows where this target is measured (dataset is sparse).
        train_mask = ~np.isnan(y_train_full)
        X_train, y_train = X_train_all[train_mask], y_train_full[train_mask]

        # Test rows without a label for this target contribute nothing to its MAE
        # (masked_mae drops NaN y_true rows), but we must predict for all rows we score against.
        test_mask = ~np.isnan(y_test_full)
        X_test, y_test = X_test_all[test_mask], y_test_full[test_mask]

        print(f"\n[{target_label}] train n={train_mask.sum()}, test n={test_mask.sum()}")

        ridge_pred = fit_predict_ridge(X_train, y_train, X_test, seed=RANDOM_SEED)
        rf_pred = fit_predict_random_forest(X_train, y_train, X_test, seed=RANDOM_SEED)

        ridge_mae = masked_mae(y_test, ridge_pred)
        rf_mae = masked_mae(y_test, rf_pred)

        print(f"  Ridge MAE:         {ridge_mae:.4f}")
        print(f"  RandomForest MAE:  {rf_mae:.4f}")

        results[target_label] = {
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "ridge_mae": ridge_mae,
            "random_forest_mae": rf_mae,
        }

    RESULTS_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "dataset": "asap-discovery/antiviral-potency-2025-unblinded",
                "split": "official chronological (Set column: Train/Test)",
                "featurization": "Morgan fingerprint (radius=2, 2048 bits)",
                "seed": RANDOM_SEED,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
