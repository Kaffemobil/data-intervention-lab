# Executed forecasting-stage scripts

These files are preserved close to the form in which the sequential forecasting studies were run. They therefore retain execution-environment paths such as `/mnt/data` and should be read as an **audit archive**, not as a polished package API.

- `aggregate_gscpi/` — rolling 1/3/6-month BLS forecasting baseline, GSCPI augmentation, block-bootstrap evaluation, and the later GSCPI transformation validation.
- `portwatch/` — fixed-exposure PortWatch feature construction, rolling residual-correction model, block-bootstrap stage gate, and exploratory ablations.

The compact public `data/vq_modeling_table.csv` is a downstream derived table from this research sequence. The raw Census and PortWatch bundles are identified by hashes in `DATA_CARD.md` rather than duplicated here.
