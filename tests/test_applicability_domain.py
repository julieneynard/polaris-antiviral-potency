"""Applicability-domain diagnostic: similarity computation and the summary/binning math.

No network access: RDKit fingerprint computation is entirely local, same as the
featurization tests would be.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.applicability_domain import applicability_domain_summary, nearest_neighbor_similarity


def test_molecule_identical_to_a_training_molecule_has_similarity_one():
    train_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    test_smiles = ["CCO"]  # exact duplicate of a training molecule

    similarities = nearest_neighbor_similarity(train_smiles, test_smiles)

    assert similarities[0] == pytest.approx(1.0)


def test_structurally_novel_molecule_has_lower_similarity_than_a_near_duplicate():
    train_smiles = ["CCCCCCCC"]  # octane
    test_smiles = [
        "CCCCCCCC",  # identical
        "c1ccc2ccccc2c1",  # naphthalene - structurally unrelated to an alkane
    ]

    similarities = nearest_neighbor_similarity(train_smiles, test_smiles)

    assert similarities[0] == pytest.approx(1.0)
    assert similarities[1] < similarities[0]


def test_nearest_neighbor_uses_the_max_not_mean_similarity():
    # One close train match and several unrelated ones - similarity should reflect the
    # close match, not be dragged down by the unrelated training molecules.
    train_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccc2ccccc2c1"]
    test_smiles = ["CCO"]

    similarities = nearest_neighbor_similarity(train_smiles, test_smiles)

    assert similarities[0] == pytest.approx(1.0)


def test_summary_detects_negative_correlation_between_similarity_and_error():
    rng = np.random.default_rng(0)
    similarities = np.linspace(0.1, 0.9, 60)
    # Error decreases as similarity increases, plus a little noise.
    abs_errors = (1.0 - similarities) * 2 + rng.normal(scale=0.01, size=60)

    summary = applicability_domain_summary(similarities, abs_errors, n_bins=3)

    assert summary["similarity_error_pearson_r"] < -0.9  # strong, expected-sign correlation
    assert len(summary["bins_low_to_high_similarity"]) == 3

    bins = summary["bins_low_to_high_similarity"]
    total_n = sum(b["n"] for b in bins)
    assert total_n == 60
    # Lowest-similarity bin should have the highest mean error, given the construction.
    assert bins[0]["mean_abs_error"] > bins[-1]["mean_abs_error"]


def test_summary_bins_are_ordered_by_similarity_range():
    similarities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # Non-constant so the correlation calc doesn't divide by a zero stddev.
    abs_errors = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.0])

    summary = applicability_domain_summary(similarities, abs_errors, n_bins=3)
    ranges = [b["similarity_range"] for b in summary["bins_low_to_high_similarity"]]

    for i in range(len(ranges) - 1):
        assert ranges[i][1] <= ranges[i + 1][0]
