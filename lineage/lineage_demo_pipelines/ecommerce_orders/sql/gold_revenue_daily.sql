-- Gold KPI from enriched orders
CREATE OR REPLACE TABLE gold.revenue_daily AS
SELECT
  DATE_TRUNC('DAY', updated_at) AS day,
  COUNT(*)                       AS orders,
  SUM(order_total - COALESCE(discount, 0)) AS net_revenue
FROM gold.orders_enriched
GROUP BY 1;
