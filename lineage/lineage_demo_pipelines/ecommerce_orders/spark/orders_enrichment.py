"""Silver -> Gold enrichment (join + derived fields)."""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, lit

spark = SparkSession.builder.getOrCreate()

orders = spark.table("silver.orders_clean")
customers = spark.table("silver.customers_clean")

enriched = (
    orders.alias("o")
    .join(customers.alias("c"), col("o.customer_id") == col("c.customer_id"), "left")
    .select(
        col("o.order_id"),
        col("o.customer_id"),
        col("c.customer_name"),
        col("c.country"),
        col("o.order_total"),
        col("o.discount"),
        (col("o.order_total") - coalesce(col("o.discount"), lit(0.0))).alias("net_total"),
        col("o.status"),
        col("o.updated_at"),
    )
)

enriched.write.mode("overwrite").format("delta").saveAsTable("gold.orders_enriched")
