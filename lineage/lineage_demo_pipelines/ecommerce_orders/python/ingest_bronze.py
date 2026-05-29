"""Ingestion to bronze tables (CSV -> Delta tables).

The lineage agent should pick up:
- reads from file paths (raw.orders_csv / raw.customers_csv)
- writes to tables (bronze.orders_raw / bronze.customers_raw)
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

RAW_ORDERS = "raw/orders.csv"
RAW_CUSTOMERS = "raw/customers.csv"

# Orders -> bronze
orders_df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(RAW_ORDERS)
)
orders_df.write.mode("overwrite").format("delta").saveAsTable("bronze.orders_raw")

# Customers -> bronze
customers_df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(RAW_CUSTOMERS)
)
customers_df.write.mode("overwrite").format("delta").saveAsTable("bronze.customers_raw")
