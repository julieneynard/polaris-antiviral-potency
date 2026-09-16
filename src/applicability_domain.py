"""Applicability-domain (AD) diagnostic: is test-set error explained by how far a
molecule sits from the training distribution?

RandomForest can only interpolate within the training data's structural space; the
results write-up already showed it visually (predictions compress toward the mean for
the most potent SARS-CoV-2 test compounds). This module quantifies that: for each test
molecule, compute its Morgan-fingerprint Tanimoto similarity to its nearest training
neighbor, then check whether prediction error correlates with that similarity (an
"in-domain" molecule close to something the model was trained on should be easier to
predict than an "out-of-domain" one that's structurally novel).

Uses the same fingerprint definition (radius, bits) as the modeling features in
src/features.py, so "similarity" here reflects the same structural representation the
models are actually trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from src.features import MORGAN_N_BITS, MORGAN_RADIUS


def _morgan_bitvects(smiles_list: list[str], radius: int, n_bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"RDKit could not parse molecule: {s!r}")
        fps.append(generator.GetFingerprint(mol))
    return fps


def nearest_neighbor_similarity(
    train_smiles: list[str],
    test_smiles: list[str],
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_N_BITS,
) -> np.ndarray:
    """For each test molecule, the max Tanimoto similarity to any training molecule."""
    train_fps = _morgan_bitvects(train_smiles, radius, n_bits)
    test_fps = _morgan_bitvects(test_smiles, radius, n_bits)

    similarities = np.empty(len(test_fps))
    for i, fp in enumerate(test_fps):
        similarities[i] = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    return similarities


def applicability_domain_summary(similarities: np.ndarray, abs_errors: np.ndarray, n_bins: int = 3) -> dict:
    """Correlate nearest-neighbor similarity with absolute error, and break error down
    by similarity tertile (or `n_bins` equal-count bins).

    A negative correlation (lower similarity -> higher error) is the expected signature
    of a real applicability-domain effect - the model does worse on structurally novel
    molecules. `n_bins` equal-*count* (not equal-width) bins keep each bin's MAE
    statistically comparable even when similarities are unevenly distributed.
    """
    similarities = np.asarray(similarities, dtype=float)
    abs_errors = np.asarray(abs_errors, dtype=float)

    pearson_r = float(np.corrcoef(similarities, abs_errors)[0, 1])

    bin_labels = pd.qcut(similarities, q=n_bins, duplicates="drop")
    bins = []
    for interval in bin_labels.categories:
        mask = bin_labels == interval
        bins.append(
            {
                "similarity_range": [float(interval.left), float(interval.right)],
                "n": int(mask.sum()),
                "mean_abs_error": float(abs_errors[mask].mean()),
            }
        )

    return {
        "similarity_error_pearson_r": pearson_r,
        "bins_low_to_high_similarity": bins,
    }
