"""ADMET task: load Polaris ASAP Antiviral ADMET data -> official chronological split ->
Morgan FP features -> 5-fold CV on the train fold (diagnostic) -> fit on full train,
evaluate on official test (headline).

Metric note: the official ADMET competition scores MLM/HLM/KSOL/MDR1-MDCKII using MAE
on a log-transformed, detection-limit-clipped endpoint (LogD gets plain MAE directly).
Polaris/ASAP have not published the exact clip thresholds, so this project scores every
ADMET endpoint with the same plain masked MAE used for the potency task instead of
guessing at unpublished thresholds. See src/data.py and the README for details.

Run with:
    uv run python scripts/run_admet_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ADMET_DATASET_ID, ADMET_TARGET_COLS, ADMET_TARGET_LABELS, load_admet_dataframe
from src.pipeline import run_baseline_pipeline

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "admet_metrics.json"


def main() -> None:
    run_baseline_pipeline(
        task_name="admet",
        dataset_id=ADMET_DATASET_ID,
        load_dataframe_fn=load_admet_dataframe,
        target_cols=ADMET_TARGET_COLS,
        target_labels=ADMET_TARGET_LABELS,
        results_path=RESULTS_PATH,
    )


if __name__ == "__main__":
    main()
