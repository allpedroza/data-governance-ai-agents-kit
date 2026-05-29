# Ecommerce Orders Pipeline (Lineage Demo)

This mini project is intentionally multi-format (Python + SQL + Spark/Databricks-style + Terraform) to help the Data Lineage AI Agent extract a realistic lineage graph.

## What it models (high level)
- **Raw files** -> Bronze tables
- Bronze -> Silver (cleaning + standardization)
- Silver -> Gold (daily revenue + KPIs)

### Key assets you should see in lineage
- `raw.orders_csv` -> `bronze.orders_raw`
- `raw.customers_csv` -> `bronze.customers_raw`
- `bronze.orders_raw` -> `silver.orders_clean`
- `bronze.customers_raw` -> `silver.customers_clean`
- `silver.orders_clean` + `silver.customers_clean` -> `gold.orders_enriched`
- `gold.orders_enriched` -> `gold.revenue_daily`

## How to run lineage
From your cloned repo of the kit:

```bash
python lineage_system.py /path/to/lineage_demo_pipelines/ecommerce_orders --visualize dashboard --report
```

Optional (LLM-assisted context enrichment):
```bash
export OPENAI_API_KEY="..."
export DATA_LINEAGE_LLM_MODEL="..."
python lineage_system.py /path/to/lineage_demo_pipelines/ecommerce_orders --visualize dashboard --report
```

## Where the lineage signals are
- SQL transforms are in `sql/`
- Spark transforms are in `spark/`
- Python ingestion is in `python/`
- IaC references are in `terraform/`

Tip: for screenshots in your article, open the interactive dashboard outputs and capture:
1) Hierarchical view (upstream -> downstream)
2) Sankey view (flow)
3) Radial view (impact from `gold.orders_enriched`)
