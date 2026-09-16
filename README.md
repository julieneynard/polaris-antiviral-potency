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
- **Cross-validation (diagnostic, not headline)**: 5-fold CV on the train fold only,
  reported as an internal stability check (mean ± std MAE across folds). This is a
  random split *within* the training data — it never touches the test fold, so it
  cannot leak into or replace the headline chronological-test number; it exists purely
  to show how much a model's error varies depending on which training rows it sees. Any
  per-fold data-dependent step (Ridge's `StandardScaler`) is fit on that fold's training
  partition only, same as the full pipeline. See
  [`src/cross_validate.py`](src/cross_validate.py).
- **Determinism**: all seeds fixed (`RANDOM_SEED = 0` in
  [`src/train_baseline.py`](src/train_baseline.py)); RandomForest is run with
  `n_jobs=1` because parallel tree aggregation can otherwise change floating-point
  summation order across runs, breaking bit-for-bit reproducibility even with a fixed
  `random_state`. Verified: two full runs of the pipeline produce byte-identical
  `results/baseline_metrics.json`.

## Results

| Target | n train | n test | Ridge 5-fold train-CV MAE | Ridge **official test MAE** | RF 5-fold train-CV MAE | RF **official test MAE** |
|---|---|---|---|---|---|---|
| pIC50 (SARS-CoV-2 Mpro) | 842 | 263 | 0.710 ± 0.043 | **0.819** | 0.461 ± 0.056 | **0.627** |
| pIC50 (MERS-CoV Mpro) | 901 | 297 | 0.829 ± 0.031 | **0.864** | 0.509 ± 0.032 | **0.488** |

The bold columns are the headline numbers (official chronological test fold). Note that
for the SARS-CoV-2 target, the train-fold CV error is noticeably lower than the
chronological-test error — exactly the generalization gap a random split would have
hidden, and the reason the task's official split is temporal rather than random.

(Regenerate with `uv run python scripts/run_baseline.py`; full numbers, including
per-fold CV values, in [`results/baseline_metrics.json`](results/baseline_metrics.json).)

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
  data.py              # load dataset from Polaris Hub, build official chronological split
  features.py          # Morgan fingerprint featurization (RDKit)
  evaluate.py           # MAE harness matching polaris' own metric logic
  train_baseline.py     # leakage-free Ridge / RandomForest baselines
  cross_validate.py     # 5-fold CV on the train fold (diagnostic only)
scripts/
  run_baseline.py      # end-to-end pipeline
results/
  baseline_metrics.json
```

## Scope

This is an initial baseline. Deliberately out of scope for now: graph neural networks
or other deep models, and hyperparameter tuning — see the project's hard rules for why
(CPU-only laptop reproducibility, start simple before adding complexity).
