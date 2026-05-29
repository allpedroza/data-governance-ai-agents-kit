from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as _sum, avg

spark = SparkSession.builder.getOrCreate()

cdr = spark.table("silver.cdr_clean")
crm = spark.table("silver.crm_clean")

agg = (
    cdr.groupBy("msisdn")
      .agg(
          count("*").alias("calls_30d"),
          _sum(col("call_duration_sec")).alias("call_seconds_30d"),
          avg(col("call_duration_sec")).alias("avg_call_sec_30d"),
      )
)

features = (
    crm.join(agg, "msisdn", "left")
       .select(
           "msisdn",
           "plan",
           "tenure_months",
           "has_complaint",
           "calls_30d",
           "call_seconds_30d",
           "avg_call_sec_30d",
           "churned"
       )
)

features.write.mode("overwrite").format("delta").saveAsTable("gold.churn_features")
