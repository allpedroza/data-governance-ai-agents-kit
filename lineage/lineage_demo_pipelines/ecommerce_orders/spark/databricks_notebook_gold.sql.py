# Databricks notebook source
# MAGIC %sql
# MAGIC -- Gold revenue daily from enriched orders
# MAGIC CREATE OR REPLACE TABLE gold.revenue_daily AS
# MAGIC SELECT
# MAGIC   DATE_TRUNC('DAY', updated_at) AS day,
# MAGIC   COUNT(*) AS orders,
# MAGIC   SUM(net_total) AS net_revenue
# MAGIC FROM gold.orders_enriched
# MAGIC GROUP BY 1;
