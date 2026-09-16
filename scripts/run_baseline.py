"""Potency task: load Polaris ASAP Antiviral Potency data -> official chronological
split -> Morgan FP features -> 5-fold CV on the train fold (diagnostic) -> fit on full
train, evaluate on official test (headline) -> MAE under the official metric logic.

Run with:
    uv run python scripts/run_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import POTENCY_DATASET_ID, POTENCY_TARGET_COLS, load_potency_dataframe
from src.pipeline import run_baseline_pipeline

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "potency_metrics.json"


def main() -> None:
    run_baseline_pipeline(
        task_name="potency",
        dataset_id=POTENCY_DATASET_ID,
        load_dataframe_fn=load_potency_dataframe,
        target_cols=POTENCY_TARGET_COLS,
        results_path=RESULTS_PATH,
    )


if __name__ == "__main__":
    main()
