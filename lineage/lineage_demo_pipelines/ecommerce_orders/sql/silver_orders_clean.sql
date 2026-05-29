-- Bronze -> Silver (orders)
CREATE OR REPLACE TABLE silver.orders_clean AS
SELECT
  CAST(order_id AS BIGINT)         AS order_id,
  CAST(customer_id AS BIGINT)      AS customer_id,
  CAST(order_total AS DOUBLE)      AS order_total,
  CAST(discount AS DOUBLE)         AS discount,
  CAST(updated_at AS TIMESTAMP)    AS updated_at,
  UPPER(COALESCE(status, 'UNKNOWN')) AS status
FROM bronze.orders_raw
WHERE order_id IS NOT NULL;
