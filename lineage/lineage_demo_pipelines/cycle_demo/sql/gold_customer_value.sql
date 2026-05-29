CREATE OR REPLACE TABLE gold.customer_value AS
SELECT
  customer_id,
  CASE WHEN score >= 0.8 THEN 'HIGH' ELSE 'LOW' END AS value_segment
FROM silver.customer_scores;
