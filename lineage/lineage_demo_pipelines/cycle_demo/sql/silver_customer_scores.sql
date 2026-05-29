CREATE OR REPLACE TABLE silver.customer_scores AS
SELECT
  customer_id,
  (CASE WHEN value_segment = 'HIGH' THEN 0.9 ELSE 0.3 END) AS score
FROM gold.customer_value;
