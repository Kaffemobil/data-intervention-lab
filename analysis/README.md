# Analysis code map

This repository is a compact public research artifact, but the underlying project involved several sequential studies. The code here separates **reproducible public analyses**, **data acquisition**, **executed study scripts**, and an **invalidated analysis retained for provenance**.

## Public reproducibility scripts

- `census_hierarchy_audit.py` — streams the official Census bundle, enforces parent/child accounting identities, and reproduces the 50% naive top-share pathology plus the hierarchy-aware 2017–2018 exposure metrics.
- `hierarchy_invariant.py` — minimal form of the invariant that triggered invalidation.
- `vq_sampling_experiment.py` — regenerates the 200-seed VQ sampling replicates, summary table, and 32-code description used by the site.
- `representation_shift_audit.py` — applies the frozen codebook to the 2025+ holdout and regenerates the representation-shift tables.

## Data acquisition

- `acquisition/download_census_porths.py` — the standard-library downloader used to retrieve 306 official Census Port-HS slices. The Census key is read from `CENSUS_API_KEY` and is never persisted.
- `acquisition/download_portwatch_monthly.py` — the robust ArcGIS/PortWatch downloader used for origin-country and U.S.-destination-port monthly data.

## Executed study archive

`study_archive/` contains the scripts used in the earlier forecasting stages that produced the aggregate GSCPI and PortWatch/AIS results summarized in the falsification ledger. They are kept close to their executed form, including execution-environment paths, so the audit trail is visible rather than rewritten after the fact.

The public repository does **not** duplicate every raw input because the Census bundle alone is ~122 MB. Dataset hashes and provenance are recorded in `DATA_CARD.md` and `data/source_inventory.csv`.

## Invalidated code

`rejected/run_granular_network_study_INVALIDATED.py` is the early Census network script that was thrown out when the hierarchy bug was discovered. It remains in the repository specifically to document what failed and why. See `rejected/README.md`.

## Protocols

Pre-specified study rules and implementation amendments are in `../protocols/`, including the stage gate, availability rules, fixed-exposure window, and missingness policy.
