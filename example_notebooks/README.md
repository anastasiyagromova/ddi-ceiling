## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run from the repository root.

```python
import numpy as np
from src import ddiceil

# One row per unit (a drug pair, or a drug pair and an adverse event):
# y: 0/1 label from a curated answer key; F: number of co-reports carrying the signal;
# n: number of co-reports; logit_r: logit of the rate expected from the two drugs' single-drug data
fit = ddiceil.dce(y, F, n, logit_r, law="normal")      # or law="npmle"
fit["ceiling"]                                         # best AUROC of any statistic of this channel
fit["auroc_data"], fit["lr"]                           # AUROC and scores of the cross-fitted detector (DCE-LR)

# Forecast the AUROC with four times the reports (normal law)
ddiceil.forecast_auroc(y, 4 * n, logit_r, fit["law0"], fit["law1"])

# Audit a benchmark: pos is an (m, 2) array of positive drug pairs, degree the training degree of each drug
rng = np.random.default_rng(0)
neg, kept = ddiceil.pma_negatives(pos, ddiceil.propensity_bins(degree), is_positive, rng)
labels = np.r_[np.ones(len(kept)), np.zeros(len(neg))]
auroc_pma = ddiceil.auroc(labels, np.r_[score(pos[kept]), score(neg)])
ddiceil.shortcut_share(auroc_random, auroc_pma)        # share of the benchmark AUROC that is single-drug shortcut
```

### Dataset Download

The datasets used in the example notebook:

- **`datasets/faers_crescenddi.csv.gz`** (included in this repository): one row per CRESCENDDI control with at least
  one FAERS report naming both drugs, with its label, single-drug stratum and the signal count, report count and
  expected rate of the co-exposure and attribution channels. Built from the FAERS quarterly files 2014Q3 to 2026Q2
  (16.2 million deduplicated reports) and the CRESCENDDI reference set.
- **DrugBank DDI network** (`ChCh-Miner_durgbank-chem-chem.tsv.gz`) from Stanford BioSNAP, downloaded into
  `datasets/` by the notebook.

## Main Components

### Ceiling estimation (`src/ddiceil/dce.py`)

- `dce`: fits the latent laws, returns the channel ceiling and the cross-fitted detector (Algorithm 1)
- `forecast_auroc`: AUROC of the channel at other report counts
- `stratified_auroc`: AUROC within shortcut strata
- `drug_bootstrap`: two-way drug-cluster bootstrap intervals

### Shortcut audit (`src/ddiceil/pma.py`)

- `pma_negatives`, `propensity_bins`: propensity-matched negatives for a benchmark (Algorithm 2)
- `shortcut_share`, `leakage`: share of AUROC lost under matching; leakage of the matching
- `MatchedNegativeSampler`, `identifiability_first_negatives`: matched negatives for identifiability-first training

`src/test_properties.py` checks that DCE recovers a known ceiling and that PMA sends a propensity-only model to chance.

## Algorithm Overview

1. **Latent-excess model**: each unit's signal count is binomial around its single-drug expectation, shifted by a
   latent excess drawn from a class-conditional law (normal, or nonparametric on a grid)
2. **DCE**: fit the two laws by marginal likelihood
   - The ceiling is the AUROC of the latent excess itself, which bounds every statistic of the channel
   - Two-fold cross-fitting gives the likelihood-ratio detector, DCE-LR
   - Simulating from the fitted laws at other report counts gives the forecast
3. **PMA**: for each positive pair, keep one drug and replace the other by a drug in the same propensity bin
   - Any score built from single-drug propensity is then at chance
   - The AUROC a model keeps is pair-specific; the share it loses is single-drug shortcut

## Data Format

- **DCE**: NumPy arrays of equal length `y`, `F`, `n`, `logit_r`; optional `strata` for `stratified_auroc`
- **PMA**: `pos`, an (m, 2) integer array of positive pairs; one propensity per drug (for example its training
  degree); `is_positive(a, b)`, true if the pair is a positive
