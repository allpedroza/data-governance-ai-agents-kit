terraform {
  required_providers {
    databricks = {
      source = "databricks/databricks"
      version = "~> 1.0"
    }
  }
}

# Example: Databricks jobs referencing notebooks/scripts in this project.
# A lineage tool can use this to connect orchestration to code artifacts.

resource "databricks_job" "ingest_bronze" {
  name = "orders_ingest_bronze"

  task {
    task_key = "ingest_bronze"
    spark_python_task {
      python_file = "${path.module}/../python/ingest_bronze.py"
      parameters  = []
    }
  }
}

resource "databricks_job" "enrich_orders" {
  name = "orders_enrichment_gold"

  task {
    task_key = "orders_enrichment"
    spark_python_task {
      python_file = "${path.module}/../spark/orders_enrichment.py"
    }
  }
}
