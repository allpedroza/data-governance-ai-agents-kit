"""
Google BigQuery Connector

Specialized connector for Google BigQuery with:
- Service account authentication
- OAuth authentication
- Project/dataset management
- BigQuery-specific optimizations
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import (
    WarehouseConnector,
    WarehouseConnectionError,
    WarehouseQueryError,
    WarehouseInfo,
    TableInfo,
    QueryResult
)

logger = logging.getLogger(__name__)


class BigQueryConnector(WarehouseConnector):
    """
    Connector for Google BigQuery.

    Supports:
    - Service account authentication
    - Application default credentials
    - OAuth authentication
    - Cross-project queries
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset: Optional[str] = None,
        credentials_path: Optional[str] = None,
        credentials_json: Optional[Dict[str, Any]] = None,
        location: Optional[str] = None,
        connection_timeout: int = 30,
        query_timeout: int = 300,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        super().__init__(connection_timeout, query_timeout, pool_size, max_overflow)

        # Load from environment if not provided
        self.project_id = project_id or os.getenv("BIGQUERY_PROJECT_ID") or os.getenv("BIGQUERY_PROJECT") or os.getenv("GCP_PROJECT")
        self.dataset = dataset or os.getenv("BIGQUERY_DATASET")
        self.credentials_path = credentials_path or os.getenv("BIGQUERY_CREDENTIALS_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.credentials_json = credentials_json
        self.location = location or os.getenv("BIGQUERY_LOCATION", "US")

        self._client = None

    @property
    def warehouse_type(self) -> str:
        return "bigquery"

    def connect(self) -> None:
        """Establish connection to BigQuery."""
        if self._connected:
            return

        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery required: pip install google-cloud-bigquery"
            )

        if not self.project_id:
            raise WarehouseConnectionError("BigQuery project_id is required")

        try:
            if self.credentials_json:
                # Use credentials from dict
                credentials = service_account.Credentials.from_service_account_info(
                    self.credentials_json
                )
                self._client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials,
                    location=self.location
                )
            elif self.credentials_path:
                # Use credentials from file
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self._client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials,
                    location=self.location
                )
            else:
                # Use application default credentials
                self._client = bigquery.Client(
                    project=self.project_id,
                    location=self.location
                )

            self._connected = True
            logger.info(f"Connected to BigQuery project: {self.project_id}")

        except Exception as e:
            raise WarehouseConnectionError(f"Failed to connect to BigQuery: {str(e)}")

    def disconnect(self) -> None:
        """Close the BigQuery connection."""
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False
        logger.info("Disconnected from BigQuery")

    def test_connection(self) -> bool:
        """Test if the connection is valid."""
        try:
            if not self._connected:
                self.connect()
            # Try to list datasets as a connection test
            list(self._client.list_datasets(max_results=1))
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_info(self) -> WarehouseInfo:
        """Get information about the BigQuery connection."""
        if not self._connected:
            self.connect()

        return WarehouseInfo(
            warehouse_type=self.warehouse_type,
            name=self.project_id,
            host="bigquery.googleapis.com",
            database=self.project_id,
            schema=self.dataset,
            connected=True,
            version="BigQuery",
            metadata={
                "location": self.location,
                "dataset": self.dataset
            }
        )

    def list_databases(self) -> List[str]:
        """List all accessible projects (treated as databases)."""
        if not self._connected:
            self.connect()

        # In BigQuery, projects are like databases
        # We return the current project and any accessible projects
        return [self.project_id]

    def list_schemas(self, database: Optional[str] = None) -> List[str]:
        """List all datasets (treated as schemas) in a project."""
        if not self._connected:
            self.connect()

        project = database or self.project_id
        datasets = list(self._client.list_datasets(project=project))
        return [ds.dataset_id for ds in datasets]

    def list_tables(
        self,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_views: bool = True
    ) -> List[TableInfo]:
        """List all tables in a dataset."""
        if not self._connected:
            self.connect()

        dataset_id = schema or self.dataset
        project = database or self.project_id

        if not dataset_id:
            raise ValueError("Dataset is required. Set it in constructor or pass as schema parameter.")

        dataset_ref = f"{project}.{dataset_id}"
        tables = list(self._client.list_tables(dataset_ref))

        result = []
        for table in tables:
            if not include_views and table.table_type == "VIEW":
                continue

            result.append(TableInfo(
                name=table.table_id,
                schema=dataset_id,
                database=project,
                table_type=table.table_type
            ))

        return result

    def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> TableInfo:
        """Get detailed information about a table."""
        if not self._connected:
            self.connect()

        dataset_id = schema or self.dataset
        project = database or self.project_id

        table_ref = f"{project}.{dataset_id}.{table_name}"
        table = self._client.get_table(table_ref)

        columns = self.get_table_schema(table_name, schema, database)

        return TableInfo(
            name=table.table_id,
            schema=dataset_id,
            database=project,
            row_count=table.num_rows,
            size_bytes=table.num_bytes,
            created_at=table.created,
            last_modified=table.modified,
            table_type=table.table_type,
            columns=columns,
            metadata={
                "description": table.description,
                "labels": dict(table.labels) if table.labels else {},
                "partition_field": table.time_partitioning.field if table.time_partitioning else None,
                "clustering_fields": list(table.clustering_fields) if table.clustering_fields else []
            }
        )

    def get_table_schema(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get table column schema."""
        if not self._connected:
            self.connect()

        dataset_id = schema or self.dataset
        project = database or self.project_id

        table_ref = f"{project}.{dataset_id}.{table_name}"
        table = self._client.get_table(table_ref)

        columns = []
        for field in table.schema:
            columns.append({
                "name": field.name,
                "type": field.field_type,
                "nullable": field.mode != "REQUIRED",
                "default": None,
                "primary_key": False,
                "comment": field.description
            })

        return columns

    def read_sample(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        n_rows: int = 1000
    ) -> QueryResult:
        """Read a sample of rows from a table."""
        dataset_id = schema or self.dataset
        project = database or self.project_id

        # Use TABLESAMPLE for large tables if available, otherwise LIMIT
        full_name = f"`{project}.{dataset_id}.{table_name}`"
        query = f"SELECT * FROM {full_name} LIMIT {n_rows}"

        return self.execute_query(query, max_rows=n_rows)

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        max_rows: Optional[int] = None
    ) -> QueryResult:
        """Execute a SQL query and return results."""
        if not self._connected:
            self.connect()

        from google.cloud import bigquery

        start_time = time.time()

        try:
            job_config = bigquery.QueryJobConfig()

            if parameters:
                job_config.query_parameters = [
                    bigquery.ScalarQueryParameter(name, "STRING", value)
                    for name, value in parameters.items()
                ]

            query_job = self._client.query(query, job_config=job_config)
            results = query_job.result(timeout=self.query_timeout)

            execution_time = (time.time() - start_time) * 1000

            columns = [field.name for field in results.schema]
            rows = []
            truncated = False

            for i, row in enumerate(results):
                if max_rows and i >= max_rows:
                    truncated = True
                    break
                rows.append(dict(row))

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=execution_time,
                query_id=query_job.job_id,
                truncated=truncated,
                metadata={
                    "bytes_processed": query_job.total_bytes_processed,
                    "bytes_billed": query_job.total_bytes_billed,
                    "cache_hit": query_job.cache_hit
                }
            )

        except Exception as e:
            raise WarehouseQueryError(f"Query execution failed: {str(e)}")

    def get_row_count(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> int:
        """Get the row count for a table."""
        if not self._connected:
            self.connect()

        dataset_id = schema or self.dataset
        project = database or self.project_id

        table_ref = f"{project}.{dataset_id}.{table_name}"
        table = self._client.get_table(table_ref)

        return table.num_rows

    def get_table_statistics(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get table statistics."""
        if not self._connected:
            self.connect()

        dataset_id = schema or self.dataset
        project = database or self.project_id

        table_ref = f"{project}.{dataset_id}.{table_name}"
        table = self._client.get_table(table_ref)

        stats = {
            "table_name": table_ref,
            "row_count": table.num_rows,
            "size_bytes": table.num_bytes,
            "created_at": table.created.isoformat() if table.created else None,
            "modified_at": table.modified.isoformat() if table.modified else None,
            "column_stats": {}
        }

        # Get column statistics using APPROX functions for efficiency
        columns = self.get_table_schema(table_name, schema, database)
        full_name = f"`{project}.{dataset_id}.{table_name}`"

        for col in columns:
            col_name = col["name"]
            col_type = col["type"].upper()

            try:
                # Get null count and approximate distinct count
                query = f"""
                    SELECT
                        COUNTIF(`{col_name}` IS NULL) as null_count,
                        APPROX_COUNT_DISTINCT(`{col_name}`) as distinct_count
                    FROM {full_name}
                """

                result = self.execute_query(query)
                row = result.rows[0] if result.rows else {}

                col_stats = {
                    "null_count": row.get("null_count", 0),
                    "distinct_count": row.get("distinct_count", 0),
                    "data_type": col_type
                }

                # Get min/max for numeric and date types
                if col_type in ["INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC", "DATE", "DATETIME", "TIMESTAMP"]:
                    query = f'SELECT MIN(`{col_name}`), MAX(`{col_name}`) FROM {full_name}'
                    result = self.execute_query(query)
                    if result.rows:
                        minmax = result.rows[0]
                        keys = list(minmax.keys())
                        col_stats["min"] = str(minmax[keys[0]]) if minmax[keys[0]] is not None else None
                        col_stats["max"] = str(minmax[keys[1]]) if minmax[keys[1]] is not None else None

                stats["column_stats"][col_name] = col_stats

            except Exception as e:
                logger.warning(f"Failed to get stats for column {col_name}: {str(e)}")
                stats["column_stats"][col_name] = {"error": str(e)}

        return stats

    def create_dataset(
        self,
        dataset_id: str,
        location: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """Create a new dataset."""
        if not self._connected:
            self.connect()

        from google.cloud import bigquery

        dataset_ref = f"{self.project_id}.{dataset_id}"
        dataset = bigquery.Dataset(dataset_ref)

        dataset.location = location or self.location
        if description:
            dataset.description = description

        self._client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Created dataset: {dataset_ref}")

    def get_job_history(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get recent query job history."""
        if not self._connected:
            self.connect()

        jobs = list(self._client.list_jobs(
            max_results=max_results,
            all_users=False
        ))

        return [
            {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "state": job.state,
                "created": job.created.isoformat() if job.created else None,
                "started": job.started.isoformat() if job.started else None,
                "ended": job.ended.isoformat() if job.ended else None,
                "bytes_processed": getattr(job, "total_bytes_processed", None),
                "bytes_billed": getattr(job, "total_bytes_billed", None)
            }
            for job in jobs
        ]
