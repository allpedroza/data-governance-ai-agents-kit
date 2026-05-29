# Telco Churn Features Pipeline (Lineage Demo)

This project simulates a telco pipeline that builds churn features from CDR + CRM tables.

## Expected lineage highlights
- `raw.cdr_csv` -> `bronze.cdr_raw`
- `raw.crm_csv` -> `bronze.crm_raw`
- `bronze.cdr_raw` -> `silver.cdr_clean`
- `bronze.crm_raw` -> `silver.crm_clean`
- `silver.cdr_clean` + `silver.crm_clean` -> `gold.churn_features`
- `gold.churn_features` -> `gold.churn_scoring_snapshot`

## Run lineage
```bash
python lineage_system.py /path/to/lineage_demo_pipelines/telco_churn --visualize dashboard --report
```
