# Rejected analysis retained for provenance

`run_granular_network_study_INVALIDATED.py` is the **actual early granular-network implementation that was discarded** after the Census hierarchy invariant failed.

It is intentionally preserved rather than silently deleted. The parser aggregated hierarchy parents (`TOTAL FOR ALL PORTS`, `TOTAL FOR ALL COUNTRIES`) together with their children, which mechanically duplicated mass. The resulting analysis was invalidated **before** its model metrics were interpreted.

Do not use this script to reproduce the final Census network results. Use `../census_hierarchy_audit.py` to reproduce the hierarchy diagnosis itself. The corrected network results summarized in the research ledger were produced only after hierarchy-aware slicing was enforced.
