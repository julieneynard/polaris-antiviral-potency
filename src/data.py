"""Load ASAP Discovery Polaris Hub datasets and their official chronological split.

Two tasks share this loader:
  - Potency: asap-discovery/antiviral-potency-2025-unblinded
    https://polarishub.io/datasets/asap-discovery/antiviral-potency-2025-unblinded
  - ADMET:   asap-discovery/antiviral-admet-2025-unblinded
    https://polarishub.io/datasets/asap-discovery/antiviral-admet-2025-unblinded

Both are post-challenge "unblinded" releases of the 2025 ASAP Discovery x Polaris x
OpenADMET blind challenge, and both carry a `Set` column with values {"Train", "Test"}
that reproduces the exact temporal split used to score the original blind challenge -
confirmed against each dataset's competition page, which describes a temporal split
with matching train/test sizes:
  - Potency: 1,031 / 297 (https://polarishub.io/competitions/asap-discovery/antiviral-potency-2025)
  - ADMET:     434 / 126 (https://polarishub.io/competitions/asap-discovery/antiviral-admet-2025)

We deliberately load these datasets directly (not `po.load_competition`) because the
competition objects hide test-set labels; the unblinded datasets are the correct
artifact for a retrospective, fully-evaluable benchmark.

Both are `DatasetV2` (Zarr-backed), which has no in-memory DataFrame API. We pull each
into memory with `Dataset.load_to_memory()`, then read each column via
`Dataset.get_data(row, col)` into a pandas DataFrame.

Note: `Dataset.cache()` (download the Zarr archive to disk) hit a store-type mismatch
in polaris-lib==0.13.0 against these datasets' remote store (`zarr.copy_store` silently
copied 0 keys, leaving an unconsolidated/unreadable local archive). `load_to_memory()`
reads the same remote arrays in bulk and works correctly, so we use that instead - both
datasets are small (hundreds to ~1,300 rows), so this is cheap.

ADMET metric note: the official ADMET competition scores MLM/HLM/KSOL/MDR1-MDCKII using
"MAE on the log-transformed endpoint, after clipping to the strictly positive detection
limit" (LogD gets plain MAE directly, already log-scale) - but Polaris/ASAP have not
published the exact clip thresholds, and third-party reproductions report the organizers
changed the transform (log vs. log1p) mid-competition. That transform is challenge-
specific tooling, not part of polaris-lib's own `evaluate()` (confirmed from
polaris/evaluate/_metric.py - it only does plain masked MAE, no built-in transform), so
there is no way to load it from the library either. Rather than guess unpublished
thresholds, this project scores all ADMET endpoints with the same plain masked MAE used
for potency (see src/evaluate.py) and documents the divergence from the official
transform in the README instead of silently approximating it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import polaris as po
from polaris.dataset import DatasetV2

CXSMILES_COL = "CXSMILES"
MOLECULE_NAME_COL = "Molecule Name"
SPLIT_COL = "Set"

POTENCY_DATASET_ID = "asap-discovery/antiviral-potency-2025-unblinded"
POTENCY_TARGET_COLS = ["pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"]

ADMET_DATASET_ID = "asap-discovery/antiviral-admet-2025-unblinded"
ADMET_TARGET_COLS = ["MLM", "HLM", "KSOL", "LogD", "MDR1-MDCKII"]
# Display labels with units, for plot titles/axes - confirmed from the dataset page.
ADMET_TARGET_LABELS = {
    "MLM": "MLM (uL/min/mg)",
    "HLM": "HLM (uL/min/mg)",
    "KSOL": "KSOL (uM)",
    "LogD": "LogD",
    "MDR1-MDCKII": "MDR1-MDCKII (1e-6 cm/s)",
}


def load_asap_dataframe(dataset_id: str, target_cols: list[str]) -> pd.DataFrame:
    """Load an ASAP Discovery Polaris Hub dataset as a pandas DataFrame.

    Returns a DataFrame with `molecule_name`, `cxsmiles`, `split`, and one column per
    entry in `target_cols` (using the dataset's own column name, unchanged).
    """
    dataset: DatasetV2 = po.load_dataset(dataset_id)
    dataset.load_to_memory()

    records = []
    for row in dataset.rows:
        record = {
            "molecule_name": dataset.get_data(row=row, col=MOLECULE_NAME_COL),
            "cxsmiles": dataset.get_data(row=row, col=CXSMILES_COL),
            "split": dataset.get_data(row=row, col=SPLIT_COL),
        }
        for col in target_cols:
            record[col] = dataset.get_data(row=row, col=col)
        records.append(record)
    df = pd.DataFrame.from_records(records)

    valid_splits = {"Train", "Test"}
    observed_splits = set(df["split"].unique())
    if not observed_splits <= valid_splits:
        raise ValueError(f"Unexpected split values: {observed_splits - valid_splits}")

    return df


def load_potency_dataframe() -> pd.DataFrame:
    """Load the ASAP Antiviral Potency dataset as a pandas DataFrame."""
    return load_asap_dataframe(POTENCY_DATASET_ID, POTENCY_TARGET_COLS)


def load_admet_dataframe() -> pd.DataFrame:
    """Load the ASAP Antiviral ADMET dataset as a pandas DataFrame."""
    return load_asap_dataframe(ADMET_DATASET_ID, ADMET_TARGET_COLS)


@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def official_chronological_split(df: pd.DataFrame) -> ChronologicalSplit:
    """Split the dataframe using the dataset's official `Set` column.

    This is the temporal/chronological split used by the original blind challenge -
    NOT a random split. Train and test rows are disjoint by construction (each row
    has exactly one `Set` value), so there is no leakage from this step itself; any
    leakage risk comes later, from fitting transforms on train+test combined.
    """
    train = df[df["split"] == "Train"].reset_index(drop=True)
    test = df[df["split"] == "Test"].reset_index(drop=True)
    return ChronologicalSplit(train=train, test=test)
