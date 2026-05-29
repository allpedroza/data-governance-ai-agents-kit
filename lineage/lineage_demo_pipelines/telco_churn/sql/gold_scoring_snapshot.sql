CREATE OR REPLACE TABLE gold.churn_scoring_snapshot AS
SELECT
  msisdn,
  plan,
  tenure_months,
  has_complaint,
  calls_30d,
  call_seconds_30d,
  avg_call_sec_30d,
  churned
FROM gold.churn_features;
