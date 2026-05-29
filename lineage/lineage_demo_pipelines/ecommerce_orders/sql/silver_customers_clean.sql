-- Bronze -> Silver (customers)
CREATE OR REPLACE TABLE silver.customers_clean AS
SELECT
  CAST(customer_id AS BIGINT)        AS customer_id,
  TRIM(customer_name)                AS customer_name,
  LOWER(email)                       AS email,
  CAST(created_at AS TIMESTAMP)      AS created_at,
  CAST(country AS STRING)            AS country
FROM bronze.customers_raw
WHERE customer_id IS NOT NULL;
