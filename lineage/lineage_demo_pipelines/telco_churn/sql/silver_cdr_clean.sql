CREATE OR REPLACE TABLE silver.cdr_clean AS
SELECT
  CAST(msisdn AS STRING)            AS msisdn,
  CAST(call_start AS TIMESTAMP)     AS call_start,
  CAST(call_duration_sec AS BIGINT) AS call_duration_sec,
  CAST(cell_id AS STRING)           AS cell_id,
  CAST(direction AS STRING)         AS direction
FROM bronze.cdr_raw
WHERE msisdn IS NOT NULL;
