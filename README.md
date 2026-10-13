# Data Intervention Lab

**Can physical logistics conditions predict future U.S. import-price changes—and how do dataset construction choices alter that answer?**

This project starts from a real forecasting question—whether logistics networks provide leading information about U.S. import prices—but treats **data construction itself as the research object**. Across 10.8M public trade records, BLS price series, NY Fed GSCPI, and AIS-derived PortWatch data, I test how hierarchy semantics, missingness, temporal availability, exposure construction, and representation-aware sampling alter what a downstream model learns.

## Why this project is unusual

The project preserves negative results. Four increasingly specific logistics hypotheses failed a predeclared 5% RMSE-improvement stage gate. Rather than tuning until something looked positive, the pipeline invalidated bad runs, preserved source provenance, and separated representation coverage from predictive utility.

## Key interventions

- **Hierarchy-aware deduplication.** Census files mix parent totals with child rows. Naive aggregation forces median top-port share to exactly 50%; hierarchy-aware parsing gives **27.9%**.
- **Missingness ≠ zero.** Incomplete PortWatch months are excluded from forecast origins rather than converted into synthetic disruptions.
- **Availability-aware evaluation.** Forecasts only use information available at the historical forecast date; future targets are never allowed into training.
- **Frozen exposure.** Product-to-port/country weights are fixed using pre-period Census data so rerouting does not erase the measured disruption.
- **Quantized representation.** A simple 32-code MiniBatch K-means codebook is learned on training features only. At a 25% data budget, VQ-balanced selection preserves **100%** of learned states vs **~89%** for random sampling while matching full-data future-holdout RMSE within ~0.02%. Importantly, this did **not** create predictive signal where none existed.
- **Representation-shift audit.** The 2025–2026 holdout activates only **17/32** training codes, with effective state count falling from **25.1** to **8.2** and train/holdout occupancy JSD of **0.360 bits**. Among future-active codes, lower historical support is associated with larger state-conditioned error (exploratory Spearman **ρ=-0.50**).

## Falsification ledger

| Signal | 3-month RRMSE | Result |
|---|---:|---|
| GSCPI | 1.042 | rejected |
| Census product×port×country | 1.024 | rejected |
| Census country→port routes | 1.012 | rejected |
| PortWatch AIS × fixed exposure | 1.094 | rejected |

Success was defined before evaluation as **RRMSE ≤ 0.95** with uncertainty excluding 1.0.

## Run locally

Open `index.html` directly, or:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Reproduce the hierarchy and representation experiments:

```bash
python3 analysis/census_hierarchy_audit.py /path/to/census_network_data_bundle.zip
python3 analysis/vq_sampling_experiment.py
python3 analysis/representation_shift_audit.py
```

The VQ scripts regenerate the checked-in derived tables used by the site. Earlier forecasting-stage scripts and data-acquisition code are retained under `analysis/study_archive/` and `analysis/acquisition/` for auditability.

## Repository structure

- `index.html` — self-contained interactive research page, no JS dependencies.
- `data/` — compact derived metrics and modeling table.
- `analysis/` — reproducible hierarchy/QC and VQ code, data-acquisition scripts, executed forecasting-stage scripts, and an explicitly rejected parser retained for provenance.
- `protocols/` — frozen study rules and implementation amendments.
- `RESEARCH_LEDGER.md` — what was frozen, invalidated, rejected, and why.
- `DATA_CARD.md` — provenance, availability, limitations, and hashes.
- `analysis/README.md` — code map and reproduction status for each script family.

## Scientific scope

This artifact does **not** claim logistics are causally irrelevant to prices. It shows that the public signals tested here did not provide stable incremental forecasting value under a strict rolling evaluation. The distinction between causal effect, representation coverage, and predictive usefulness is central to the project.
