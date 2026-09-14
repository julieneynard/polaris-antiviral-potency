# Polaris ASAP Antiviral Potency — Retrospective Benchmark

## Scientific goal

A rigorous, reproducible evaluation of ML models for predicting antiviral potency
(pIC50) against two coronavirus main proteases — **SARS-CoV-2 Mpro** and **MERS-CoV
Mpro** — using the [ASAP Discovery x Polaris x OpenADMET "Antiviral Potency 2025"
challenge](https://polarishub.io/competitions/asap-discovery/antiviral-potency-2025)
dataset.

The challenge itself has concluded and been unblinded, so this project runs it
**retrospectively**: the goal is not to compete on the leaderboard, but to demonstrate
a methodologically sound evaluation pipeline for a real prospective drug-discovery
benchmark — the kind of rigor (correct split, no leakage, matching the official
metric) that's easy to get subtly wrong and that matters more than squeezing out extra
accuracy from a fancier model.

## Data and evaluation approach

- **Dataset**: [`asap-discovery/antiviral-potency-2025-unblinded`](https://polarishub.io/datasets/asap-discovery/antiviral-potency-2025-unblinded)
  on Polaris Hub (CC0-1.0, 1,328 molecules). This is the post-challenge release with
  test-set labels revealed — the original `competition` artifact on Polaris Hub hides
  test labels by design, so it can't be used for a self-contained retrospective
  evaluation.
- **Split — official, chronological, not random**: the dataset's `Set` column
  (`Train`/`Test`) reproduces the exact temporal split ASAP Discovery used to score the
  original blind challenge (1,031 train / 297 test — matches the sizes published on the
  competition page exactly). A random split would leak information from the future into
  training and overstate performance relative to the real prospective setting this
  challenge is meant to simulate; we never use one for headline results.
- **Metric**: mean absolute error (MAE), computed exactly the way `polaris-lib`
  computes it internally — `sklearn.metrics.mean_absolute_error`, per target, on rows
  where that target's label isn't missing (the dataset is sparse: not every molecule
  was assayed against both targets), with **no cross-target aggregation** (confirmed
  from `polaris/evaluate/_metric.py`, which explicitly raises `NotImplementedError` for
  multitask aggregation). See [`src/evaluate.py`](src/evaluate.py).
- **No leakage**: featurization (Morgan fingerprints) is a deterministic structural hash
  with nothing fit on data, so it's leakage-free by construction. Where a step *does*
  fit on data (the `StandardScaler` inside the Ridge baseline), it is fit on the
  training fold only and merely applied to the test fold. See
  [`src/train_baseline.py`](src/train_baseline.py).
- **Baseline models**: Morgan fingerprints (RDKit, radius=2, 2048 bits) + Ridge
  regression and RandomForest — deliberately simple, CPU-only, no deep learning.
- **Determinism**: all seeds fixed (`RANDOM_SEED = 0` in
  [`src/train_baseline.py`](src/train_baseline.py)); RandomForest is run with
  `n_jobs=1` because parallel tree aggregation can otherwise change floating-point
  summation order across runs, breaking bit-for-bit reproducibility even with a fixed
  `random_state`. Verified: two full runs of the pipeline produce byte-identical
  `results/baseline_metrics.json`.

## Results (official chronological split)

| Target | n train | n test | Ridge MAE | RandomForest MAE |
|---|---|---|---|---|
| pIC50 (SARS-CoV-2 Mpro) | 842 | 263 | 0.819 | 0.627 |
| pIC50 (MERS-CoV Mpro) | 901 | 297 | 0.864 | 0.488 |

(Regenerate with `uv run python scripts/run_baseline.py`; full numbers in
[`results/baseline_metrics.json`](results/baseline_metrics.json).)

## Setup

```bash
uv sync
```

Requires network access to Polaris Hub (no account/login needed — the dataset is
public/CC0). If your network does TLS interception (e.g. a campus/corporate proxy),
you may need `uv sync --system-certs`.

## Usage

```bash
uv run python scripts/run_baseline.py
```

## Project structure

```
src/
  data.py            # load dataset from Polaris Hub, build official chronological split
  features.py         # Morgan fingerprint featurization (RDKit)
  evaluate.py          # MAE harness matching polaris' own metric logic
  train_baseline.py    # leakage-free Ridge / RandomForest baselines
scripts/
  run_baseline.py      # end-to-end pipeline
results/
  baseline_metrics.json
```

## Scope

This is an initial baseline. Deliberately out of scope for now: graph neural networks
or other deep models, hyperparameter tuning, and cross-validation beyond the single
official split — see the project's hard rules for why (CPU-only laptop reproducibility,
start simple before adding complexity).
