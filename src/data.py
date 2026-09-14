"""Load the Polaris ASAP Antiviral Potency dataset and its official chronological split.

Dataset: asap-discovery/antiviral-potency-2025-unblinded (Polaris Hub, CC0-1.0)
https://polarishub.io/datasets/asap-discovery/antiviral-potency-2025-unblinded

This is the post-challenge "unblinded" release of the 2025 ASAP Discovery x Polaris x
OpenADMET blind potency challenge. It has 1,328 molecules and a `Set` column with values
{"Train", "Test"} (1,031 / 297) that reproduces the exact temporal split used to score
the original blind challenge (confirmed against the competition page, which describes a
temporal split with the same train/test sizes: https://polarishub.io/competitions/asap-discovery/antiviral-potency-2025).

We deliberately load this dataset directly (not `po.load_competition`) because the
competition object hides test-set labels; the unblinded dataset is the correct artifact
for a retrospective, fully-evaluable benchmark.

This is a `DatasetV2` (Zarr-backed), which has no in-memory DataFrame API. We pull it
into memory with `Dataset.load_to_memory()`, then read each column via
`Dataset.get_data(row, col)` into a pandas DataFrame.

Note: `Dataset.cache()` (download the Zarr archive to disk) hit a store-type mismatch
in polaris-lib==0.13.0 against this dataset's remote store (`zarr.copy_store` silently
copied 0 keys, leaving an unconsolidated/unreadable local archive). `load_to_memory()`
reads the same remote arrays in bulk and works correctly, so we use that instead - the
dataset is small (1,328 rows), so this is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import polaris as po
from polaris.dataset import DatasetV2

DATASET_ID = "asap-discovery/antiviral-potency-2025-unblinded"

CXSMILES_COL = "CXSMILES"
MOLECULE_NAME_COL = "Molecule Name"
SPLIT_COL = "Set"
TARGET_COLS = ["pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"]


def load_potency_dataframe() -> pd.DataFrame:
    """Load the full ASAP Antiviral Potency dataset as a pandas DataFrame."""
    dataset: DatasetV2 = po.load_dataset(DATASET_ID)
    dataset.load_to_memory()

    records = []
    for row in dataset.rows:
        records.append(
            {
                "molecule_name": dataset.get_data(row=row, col=MOLECULE_NAME_COL),
                "cxsmiles": dataset.get_data(row=row, col=CXSMILES_COL),
                "split": dataset.get_data(row=row, col=SPLIT_COL),
                "pic50_sars_cov_2_mpro": dataset.get_data(row=row, col=TARGET_COLS[0]),
                "pic50_mers_cov_mpro": dataset.get_data(row=row, col=TARGET_COLS[1]),
            }
        )
    df = pd.DataFrame.from_records(records)

    valid_splits = {"Train", "Test"}
    observed_splits = set(df["split"].unique())
    if not observed_splits <= valid_splits:
        raise ValueError(f"Unexpected split values: {observed_splits - valid_splits}")

    return df


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
