terraform {
  required_providers {
    databricks = {
      source = "databricks/databricks"
      version = "~> 1.0"
    }
  }
}

resource "databricks_job" "telco_ingest_bronze" {
  name = "telco_ingest_bronze"
  task {
    task_key = "ingest"
    spark_python_task {
      python_file = "${path.module}/../python/ingest_bronze.py"
    }
  }
}

resource "databricks_job" "telco_build_features" {
  name = "telco_build_churn_features"
  task {
    task_key = "features"
    spark_python_task {
      python_file = "${path.module}/../spark/build_churn_features.py"
    }
  }
}
