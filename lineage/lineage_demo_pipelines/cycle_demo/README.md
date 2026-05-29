# Cycle Demo (Optional)

This folder intentionally includes a *bad practice* cycle to help you capture a screenshot of cycle detection
(if your lineage tool exposes it).

Expected cycle:
- `silver.customer_scores` depends on `gold.customer_value`
- `gold.customer_value` depends on `silver.customer_scores`

Run:
```bash
python lineage_system.py /path/to/lineage_demo_pipelines/cycle_demo --visualize dashboard --report
```
