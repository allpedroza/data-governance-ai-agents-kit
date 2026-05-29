from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

CDR = "raw/cdr.csv"
CRM = "raw/crm.csv"

cdr_df = spark.read.format("csv").option("header","true").option("inferSchema","true").load(CDR)
cdr_df.write.mode("overwrite").format("delta").saveAsTable("bronze.cdr_raw")

crm_df = spark.read.format("csv").option("header","true").option("inferSchema","true").load(CRM)
crm_df.write.mode("overwrite").format("delta").saveAsTable("bronze.crm_raw")
