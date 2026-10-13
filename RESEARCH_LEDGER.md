# Research Ledger

## Operating rule
The objective is not to make a model win. The objective is to learn which data interventions survive falsification.

## Stage gate
Primary horizon: 3 months. A new data signal is considered practically useful only if `RRMSE <= 0.95`, with the paired block-bootstrap 95% interval below 1.0.

## Interventions and decisions

### 1. Census hierarchy semantics — INVALIDATED, THEN FIXED
Raw Port-HS files contain `TOTAL FOR ALL PORTS` and `TOTAL FOR ALL COUNTRIES` beside their children. Treating parents as peer observations duplicates mass. The pathology was discovered because top-port share collapsed to exactly 0.50 across every product chapter. The run was discarded before outcome interpretation and the parser was rebuilt around hierarchy-aware invariants.

### 2. Aggregate supply-chain pressure — REJECTED
GSCPI did not clear the stage gate. The 3-month RRMSE was ~1.042.

### 3. Granular Census network — REJECTED
Product×port×country and route-topology variants did not clear the stage gate. Distributed lags also failed to produce stable gains.

### 4. PortWatch upstream throughput — REJECTED
Fixed pre-period Census exposure crossed with AIS-derived throughput worsened 3-month error. Ridge repeatedly selected the strongest shrinkage, indicating little stable incremental signal.

### 5. Representation experiment — COVERAGE IMPROVED; SIGNAL DID NOT
A 32-code vector quantization baseline was trained only on the historical training feature distribution. VQ-balanced selection preserved all learned states at 25% of the training-data budget and matched full-data holdout RMSE, but did not create a forecasting edge. This is a deliberate separation between *dataset coverage* and *downstream utility*.

### 6. Representation shift audit — EXPLORATORY DIAGNOSTIC
The frozen 32-code training representation was applied without refitting to the 2025–2026 holdout. Only **17/32** codes were active; effective state count fell from **25.1** in training to **8.2** in the holdout, with Jensen–Shannon divergence **0.360 bits**. Across holdout-active codes, lower historical support was associated with larger state-conditioned error (exploratory Spearman **ρ=-0.50**). This was added after the primary forecasting tests and does not alter their confirmatory status.

## Next research question
The most defensible next variable is actual vessel delay/dwell time rather than throughput. That experiment was designed but intentionally not included here until the source data are available and verified.

## Reproducibility note
The public VQ summary tables are regenerated directly by `analysis/vq_sampling_experiment.py`; `analysis/representation_shift_audit.py` then consumes those saved replicates. The invalidated pre-hierarchy Census script is retained separately under `analysis/rejected/` and is explicitly excluded from final results.
