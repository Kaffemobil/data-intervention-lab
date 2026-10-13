# Data Card

## Data sources

### U.S. Census International Trade — Port-HS imports
- Role: product × origin country × U.S. port × transport-mode network.
- Download: 306 annual/partial-year slices; **10,770,421 raw rows**.
- Study universe: 18 HS2 manufactured-goods chapters, 2010–2026-06.
- 2017–2018 sampled vessel-import value: **$1.31T**.
- Integrity issue discovered: hierarchical total rows coexist with children.
- Bundle SHA256: `c8399ca6d4bc040ffeffe79c38701568972d912ffd217eb117b7641fe070e39b`.

### BLS Import/Export Price Indexes
- Role: future product-level price target.
- Use: direct 1/3/6-month rolling forecasts.
- Important property: published data can be revised, so historical evaluation distinguishes information availability from observation date.

### New York Fed GSCPI
- Role: aggregate supply-chain benchmark.
- Use: establishes whether granular network information adds anything beyond a free global factor.

### IMF PortWatch
- Role: AIS-derived port calls and estimated maritime throughput.
- Important rule: missing country-months remain missing; they are not encoded as zero activity.
- Bundle SHA256: `e7909d6ffcbf4a635980c8709ee757605eceefa8e945aa13bdb92d75c1db1f75`.

## Provenance and temporal semantics
Each feature conceptually has both an `observation_date` and an `available_date`. A historical forecast may only use values whose `available_date <= forecast_origin`.

## Known limitations
- The granular Census network test uses a residual-correction formulation relative to the already frozen baseline, not a complete joint refit of all raw features.
- PortWatch throughput is not vessel delay; stable cargo volumes can coexist with long waiting times.
- The VQ experiment is a compact classical vector-quantization baseline (MiniBatch K-means), not a claim about frontier-scale learned tokenizers.
- The future holdout is modest (165 product-month observations), so tiny RMSE differences are treated as descriptive rather than practically meaningful.
- The representation-shift audit is exploratory and state-conditioned errors can be noisy when a code has few holdout observations.
