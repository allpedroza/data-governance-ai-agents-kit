CREATE OR REPLACE TABLE silver.crm_clean AS
SELECT
  CAST(msisdn AS STRING)         AS msisdn,
  CAST(plan AS STRING)           AS plan,
  CAST(tenure_months AS BIGINT)  AS tenure_months,
  CAST(has_complaint AS BOOLEAN) AS has_complaint,
  CAST(churned AS BOOLEAN)       AS churned
FROM bronze.crm_raw
WHERE msisdn IS NOT NULL;
