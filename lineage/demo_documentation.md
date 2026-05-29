# 📊 Documentação de Linhagem de Dados
**Gerado em:** 2025-12-14 08:58:43

## 📈 Resumo
- **Total de Assets:** 59
- **Total de Transformações:** 42
- **Tipos de Assets:** file, terraform_resource, table

## 🗂️ Assets de Dados

### FILE
- **raw_sales.csv**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/extract_sales.py`
- **processed_sales.parquet**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/ml_pipeline.py`
- **customer_features.csv**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/ml_pipeline.py`
- **ml_prepared_data.parquet**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/ml_pipeline.py`
- **churn_predictions.csv**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/ml_pipeline.py`

### TERRAFORM_RESOURCE
- **aws_glue_catalog_database.analytics**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/infrastructure.tf`
- **aws_glue_catalog_table.sales_fact**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/infrastructure.tf`
- **databricks_table.ml_features**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/infrastructure.tf`

### TABLE
- **SELECT * FROM customers**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/extract_sales.py`
- **kafka:localhost:9092/user_events**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/streaming_notebook.py`
- **processed_stream**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/streaming_notebook.py`
- **events_stream.value**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/streaming_notebook.py`
- **processed_stream.data**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/streaming_notebook.py`
- **aggregated_stream**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/streaming_notebook.py`
- **delta:/delta/tables/event_aggregations**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/streaming_notebook.py`
- **metrics_history.metric_date**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **result.metric_date**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **metrics_history.total_revenue**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **result.total_revenue**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **result.prev_revenue**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **result.growth_rate**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **dim_customer.customer_id**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **customer_segments.customer_id**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **dim_customer.customer_segment**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **customer_segments.customer_segment**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **fact_sales.customer_id**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **customer_segments.total_purchases**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **fact_sales.amount**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **customer_segments.avg_purchase_value**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **customer_segments.customer_tier**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **dim_product.product_id**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **product_performance.product_id**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **dim_product.product_name**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **product_performance.product_name**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **dim_product.product_category**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **product_performance.product_category**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **fact_sales.quantity**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **product_performance.units_sold**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **product_performance.revenue**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **product_performance.unique_buyers**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **fact_sales.product_id**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/analysis_queries.sql`
- **sales_summary.total_sales**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **sales_summary.unique_customers**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **dim_date.date**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **sales_summary.date**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **sales_summary.product_category**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **sales_summary.customer_segment**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **fact_sales.date_id**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **mv_daily_kpis.date**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **mv_daily_kpis.daily_revenue**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **mv_daily_kpis.daily_customers**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **sales_summary.***
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **metrics_history.record_count**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **system.CURRENT_DATE**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **metrics_history.snapshot_date**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **sales_summary**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **metrics_history**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **fact_sales.date**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`
- **dim_customer.last_purchase_date**
  - Arquivo: `/var/folders/21/0w991r752_12lzxm2bs9jcn00000gn/T/lineage_demo_0c9yoxuu/transform_data.sql`

## 🔄 Fluxos de Dados Principais

**fact_sales.customer_id**
- Entradas: 0 | Saídas: 7
- Envia para: customer_segments.total_purchases, customer_segments.avg_purchase_value, customer_segments.customer_tier

**metrics_history.total_revenue**
- Entradas: 1 | Saídas: 3
- Recebe de: sales_summary.total_sales
- Envia para: result.total_revenue, result.prev_revenue, result.growth_rate

**metrics_history.metric_date**
- Entradas: 0 | Saídas: 3
- Envia para: result.metric_date, result.growth_rate, result.prev_revenue

**dim_customer.customer_id**
- Entradas: 0 | Saídas: 3
- Envia para: customer_segments.customer_id, customer_segments.total_purchases, customer_segments.avg_purchase_value

**fact_sales.amount**
- Entradas: 0 | Saídas: 3
- Envia para: customer_segments.avg_purchase_value, product_performance.revenue, sales_summary.total_sales

**customer_segments.avg_purchase_value**
- Entradas: 3 | Saídas: 0
- Recebe de: fact_sales.amount, fact_sales.customer_id, dim_customer.customer_id

## ⚠️ Alertas e Observações
- **Nós isolados:** raw_sales.csv, SELECT * FROM customers, processed_sales.parquet, customer_features.csv, ml_prepared_data.parquet
