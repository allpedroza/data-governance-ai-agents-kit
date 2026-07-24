# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
#   "databricks-sql-connector>=2.9.0",
#   "faiss-cpu>=1.7.0",
#   "flake8>=5.0.0",
#   "google-cloud-bigquery-storage>=2.0.0",
#   "google-cloud-bigquery>=3.0.0",
#   "google-cloud-storage>=2.7.0",
#   "isort>=5.0.0",
#   "kaleido>=0.2.0",
#   "matplotlib>=3.6.0",
#   "mypy>=1.0.0",
#   "networkx>=3.0",
#   "numpy>=1.24.0",
#   "openai>=1.0.0",
#   "openpyxl>=3.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "psycopg2-binary>=2.9.0",
#   "pyarrow>=14.0.0",
#   "pyodbc>=4.0.0",
#   "pyspark>=3.3.0",
#   "pytest-cov>=4.0.0",
#   "pytest>=7.0.0",
#   "python-dotenv>=1.0.0",
#   "python-igraph>=0.10.0",
#   "pyyaml>=6.0",
#   "redshift-connector>=2.0.0",
#   "requests>=2.31.0",
#   "scikit-learn>=1.0.0",
#   "seaborn>=0.12.0",
#   "sentence-transformers>=2.2.0",
#   "snowflake-connector-python>=3.0.0",
#   "snowflake-sqlalchemy>=1.5.0",
#   "spacy>=3.5.0; extra == \"spacy\"",
#   "sphinx-rtd-theme>=1.0.0",
#   "sphinx>=5.0.0",
#   "sqlalchemy-bigquery>=1.6.0",
#   "sqlalchemy-redshift>=0.8.0",
#   "sqlalchemy>=2.0.0",
#   "sqlparse>=0.4.0",
#   "streamlit>=1.32.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Databricks Unity Catalog Connector

Specialized connector for Databricks Unity Catalog with:
- Token-based authentication
- SQL Warehouse execution
- Unity Catalog namespace support (catalog.schema.table)
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


class DatabricksConnector(WarehouseConnector):
    """
    Connector for Databricks Unity Catalog.
    """

    def __init__(
        self,
        server_hostname: Optional[str] = None,
        http_path: Optional[str] = None,
        access_token: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        connection_timeout: int = 30,
        query_timeout: int = 300,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        super().__init__(connection_timeout, query_timeout, pool_size, max_overflow)

        self.server_hostname = server_hostname or os.getenv("DATABRICKS_SERVER_HOSTNAME")
        self.http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH")
        self.access_token = access_token or os.getenv("DATABRICKS_TOKEN")
        
        # Unity catalog has a 3-level namespace: catalog.schema.table
        self.catalog = catalog or os.getenv("DATABRICKS_CATALOG") or "hive_metastore"
        self.schema_name = schema or os.getenv("DATABRICKS_SCHEMA") or "default"

    @property
    def warehouse_type(self) -> str:
        return "databricks"

    def connect(self) -> None:
        """Establish connection to Databricks."""
        if self._connected:
            return

        try:
            from databricks import sql
        except ImportError:
            raise ImportError(
                "databricks-sql-connector required: pip install databricks-sql-connector"
            )

        if not self.server_hostname or not self.http_path or not self.access_token:
            raise WarehouseConnectionError("Databricks server_hostname, http_path, and access_token are required")

        try:
            self._connection = sql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token,
                catalog=self.catalog,
                schema=self.schema_name
            )
            self._connected = True
            logger.info(f"Connected to Databricks SQL Warehouse: {self.server_hostname}")
        except Exception as e:
            raise WarehouseConnectionError(f"Failed to connect to Databricks: {str(e)}")

    def disconnect(self) -> None:
        """Close the Databricks connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
        self._connected = False
        logger.info("Disconnected from Databricks")

    def test_connection(self) -> bool:
        """Test if the connection is valid."""
        try:
            if not self._connected:
                self.connect()
            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_info(self) -> WarehouseInfo:
        """Get information about the Databricks connection."""
        if not self._connected:
            self.connect()

        return WarehouseInfo(
            warehouse_type=self.warehouse_type,
            name=self.server_hostname,
            host=self.server_hostname,
            database=self.catalog,
            schema=self.schema_name,
            connected=True,
            version="Databricks SQL",
            metadata={
                "http_path": self.http_path,
                "catalog": self.catalog
            }
        )

    def list_databases(self) -> List[str]:
        """List all accessible catalogs."""
        if not self._connected:
            self.connect()
            
        with self._connection.cursor() as cursor:
            cursor.execute("SHOW CATALOGS")
            return [row.catalog for row in cursor.fetchall()]

    def list_schemas(self, database: Optional[str] = None) -> List[str]:
        """List all schemas in a catalog."""
        if not self._connected:
            self.connect()
            
        catalog = database or self.catalog
        with self._connection.cursor() as cursor:
            cursor.execute(f"SHOW SCHEMAS IN {catalog}")
            return [row.databaseName for row in cursor.fetchall()]

    def list_tables(
        self,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_views: bool = True
    ) -> List[TableInfo]:
        """List all tables in a schema."""
        if not self._connected:
            self.connect()

        catalog = database or self.catalog
        schema_name = schema or self.schema_name

        query = f"SHOW TABLES IN {catalog}.{schema_name}"
        
        tables = []
        with self._connection.cursor() as cursor:
            cursor.execute(query)
            for row in cursor.fetchall():
                # Checking table vs view if needed; basic listing adds them all
                tables.append(TableInfo(
                    name=row.tableName,
                    schema=schema_name,
                    database=catalog
                ))

        return tables

    def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> TableInfo:
        """Get detailed information about a table."""
        if not self._connected:
            self.connect()

        catalog = database or self.catalog
        schema_name = schema or self.schema_name
        full_name = f"{catalog}.{schema_name}.{table_name}"
        
        columns = self.get_table_schema(table_name, schema_name, catalog)
        
        metadata = {}
        row_count = None
        size_bytes = None
        table_type = "TABLE"
        
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(f"DESCRIBE DETAIL {full_name}")
                details = cursor.fetchone()
                if details:
                    # Map description columns to dict
                    detail_dict = {col[0]: val for col, val in zip(cursor.description, details)}
                    table_type = detail_dict.get("type", "TABLE")
                    size_bytes = detail_dict.get("sizeInBytes")
                    metadata["format"] = detail_dict.get("format")
                    metadata["createdAt"] = detail_dict.get("createdAt")
                    metadata["lastModified"] = detail_dict.get("lastModified")
        except Exception as e:
            logger.warning(f"Could not get extended details for {full_name}: {e}")

        return TableInfo(
            name=table_name,
            schema=schema_name,
            database=catalog,
            row_count=row_count,
            size_bytes=size_bytes,
            table_type=table_type,
            columns=columns,
            metadata=metadata
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

        catalog = database or self.catalog
        schema_name = schema or self.schema_name
        full_name = f"{catalog}.{schema_name}.{table_name}"
        
        columns = []
        with self._connection.cursor() as cursor:
            cursor.execute(f"DESCRIBE {full_name}")
            for row in cursor.fetchall():
                # Ignore partition information
                if row.col_name and row.col_name.startswith("#"):
                    continue
                if not row.col_name:
                    continue
                    
                columns.append({
                    "name": row.col_name,
                    "type": row.data_type,
                    "nullable": True,
                    "default": None,
                    "primary_key": False,
                    "comment": getattr(row, "comment", None)
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
        catalog = database or self.catalog
        schema_name = schema or self.schema_name
        full_name = f"{catalog}.{schema_name}.{table_name}"
        
        query = f"SELECT * FROM {full_name} LIMIT {n_rows}"
        return self.execute_query(query)

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        max_rows: Optional[int] = None
    ) -> QueryResult:
        """Execute a SQL query and return results."""
        if not self._connected:
            self.connect()

        start_time = time.time()
        try:
            with self._connection.cursor() as cursor:
                if parameters:
                    cursor.execute(query, parameters)
                else:
                    cursor.execute(query)
                
                execution_time = (time.time() - start_time) * 1000
                
                if cursor.description is None:
                    return QueryResult(
                        columns=[],
                        rows=[],
                        row_count=0,
                        execution_time_ms=execution_time
                    )
                
                columns = [desc[0] for desc in cursor.description]
                
                if max_rows:
                    results = cursor.fetchmany(max_rows)
                    truncated = len(results) == max_rows
                else:
                    results = cursor.fetchall()
                    truncated = False
                    
                rows = [dict(zip(columns, row)) for row in results]
                
                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=execution_time,
                    truncated=truncated,
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
        catalog = database or self.catalog
        schema_name = schema or self.schema_name
        full_name = f"{catalog}.{schema_name}.{table_name}"
        
        query = f"SELECT COUNT(*) as count FROM {full_name}"
        result = self.execute_query(query)
        if result.rows:
            return result.rows[0]["count"]
        return 0

    def get_table_statistics(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get table statistics."""
        if not self._connected:
            self.connect()
            
        catalog = database or self.catalog
        schema_name = schema or self.schema_name
        full_name = f"{catalog}.{schema_name}.{table_name}"
        
        row_count = self.get_row_count(table_name, schema_name, catalog)
        
        stats = {
            "table_name": full_name,
            "row_count": row_count,
            "column_stats": {}
        }
        
        columns = self.get_table_schema(table_name, schema_name, catalog)
        for col in columns:
            col_name = col["name"]
            try:
                query = f"SELECT COUNT({col_name}) as non_nulls, COUNT(DISTINCT {col_name}) as distincts FROM {full_name}"
                res = self.execute_query(query)
                if res.rows:
                    stats["column_stats"][col_name] = {
                        "null_count": row_count - res.rows[0]["non_nulls"],
                        "distinct_count": res.rows[0]["distincts"],
                        "data_type": col["type"]
                    }
            except Exception as e:
                logger.warning(f"Failed to get stats for column {col_name}: {e}")
        
        return stats

    def get_ddl(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_dependencies: bool = False
    ) -> str:
        """Get DDL for a table."""
        if not self._connected:
            self.connect()
            
        catalog = database or self.catalog
        schema_name = schema or self.schema_name
        full_name = f"{catalog}.{schema_name}.{table_name}"
        
        query = f"SHOW CREATE TABLE {full_name}"
        try:
            result = self.execute_query(query)
            if result.rows:
                return result.rows[0]["createtab_stmt"]
        except Exception as e:
            logger.warning(f"Failed to get DDL for {full_name}: {e}")
        
        return ""

    def get_query_history(
        self,
        days: int = 7,
        limit: int = 1000,
        database_filter: Optional[str] = None,
        schema_filter: Optional[str] = None,
        table_filter: Optional[str] = None,
        user_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get query history using system tables in Unity Catalog."""
        if not self._connected:
            self.connect()

        query = f"""
            SELECT 
                request_id as query_id,
                request_params.query_text as query_text,
                user_identity.email as user_name,
                event_time as start_time,
                response.status_code as status_code,
                response.error_message as error_message
            FROM system.access.audit
            WHERE service_name = 'databrickssql' 
              AND action_name = 'runQuery'
              AND event_time >= current_timestamp() - INTERVAL {days} DAYS
        """
        
        if user_filter:
            query += f" AND user_identity.email = '{user_filter}'"
            
        query += f" ORDER BY event_time DESC LIMIT {limit}"
        
        try:
            result = self.execute_query(query)
            
            history = []
            for row in result.rows:
                status = "success" if row.get("status_code") == 200 else "failed"
                history.append({
                    "query_id": row.get("query_id"),
                    "query_text": row.get("query_text"),
                    "user_name": row.get("user_name"),
                    "start_time": row.get("start_time"),
                    "end_time": None,
                    "execution_time_ms": None,
                    "rows_produced": None,
                    "bytes_scanned": None,
                    "status": status,
                    "error_message": row.get("error_message"),
                    "tables_accessed": []
                })
            return history
            
        except Exception as e:
            logger.warning(f"Could not get query history from system.access.audit: {e}")
            return []
