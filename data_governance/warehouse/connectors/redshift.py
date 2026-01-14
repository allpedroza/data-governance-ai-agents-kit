# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
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
#   "spacy>=3.5.0; extra == "spacy"",
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
Amazon Redshift Connector

Specialized connector for Amazon Redshift with:
- PostgreSQL protocol support
- IAM authentication
- Cluster management
- Redshift-specific optimizations
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


class RedshiftConnector(WarehouseConnector):
    """
    Connector for Amazon Redshift.

    Supports:
    - Direct connection via psycopg2
    - SQLAlchemy integration
    - IAM authentication
    - Temporary credentials
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        schema: Optional[str] = None,
        cluster_identifier: Optional[str] = None,
        region: Optional[str] = None,
        iam_role: Optional[str] = None,
        use_iam: bool = False,
        ssl: bool = True,
        connection_timeout: int = 30,
        query_timeout: int = 300,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        super().__init__(connection_timeout, query_timeout, pool_size, max_overflow)

        # Load from environment if not provided
        self.host = host or os.getenv("REDSHIFT_HOST")
        self.port = port or int(os.getenv("REDSHIFT_PORT", "5439"))
        self.database = database or os.getenv("REDSHIFT_DATABASE")
        self.username = username or os.getenv("REDSHIFT_USERNAME") or os.getenv("REDSHIFT_USER")
        self.password = password or os.getenv("REDSHIFT_PASSWORD")
        self.schema = schema or os.getenv("REDSHIFT_SCHEMA", "public")
        self.cluster_identifier = cluster_identifier or os.getenv("REDSHIFT_CLUSTER_IDENTIFIER") or os.getenv("REDSHIFT_CLUSTER")
        self.region = region or os.getenv("REDSHIFT_REGION") or os.getenv("AWS_REGION", "us-east-1")
        self.iam_role = iam_role or os.getenv("REDSHIFT_IAM_ROLE")
        self.use_iam = use_iam or os.getenv("REDSHIFT_USE_IAM", "false").lower() == "true"
        self.ssl = ssl

        self._cursor = None

    @property
    def warehouse_type(self) -> str:
        return "redshift"

    def _get_iam_credentials(self) -> Dict[str, str]:
        """Get temporary credentials using IAM authentication."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required for IAM authentication: pip install boto3")

        client = boto3.client("redshift", region_name=self.region)

        response = client.get_cluster_credentials(
            DbUser=self.username,
            DbName=self.database,
            ClusterIdentifier=self.cluster_identifier,
            AutoCreate=False
        )

        return {
            "user": response["DbUser"],
            "password": response["DbPassword"]
        }

    def connect(self) -> None:
        """Establish connection to Redshift."""
        if self._connected:
            return

        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 required: pip install psycopg2-binary")

        if not self.host:
            raise WarehouseConnectionError("Redshift host is required")

        connect_params = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "connect_timeout": self.connection_timeout,
        }

        if self.use_iam:
            creds = self._get_iam_credentials()
            connect_params["user"] = creds["user"]
            connect_params["password"] = creds["password"]
        else:
            connect_params["user"] = self.username
            connect_params["password"] = self.password

        if self.ssl:
            connect_params["sslmode"] = "require"

        try:
            self._connection = psycopg2.connect(**connect_params)
            self._connection.autocommit = True
            self._cursor = self._connection.cursor()
            self._connected = True
            logger.info(f"Connected to Redshift: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            raise WarehouseConnectionError(f"Failed to connect to Redshift: {str(e)}")

    def disconnect(self) -> None:
        """Close the Redshift connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None
        self._connected = False
        logger.info("Disconnected from Redshift")

    def test_connection(self) -> bool:
        """Test if the connection is valid."""
        try:
            if not self._connected:
                self.connect()
            self._cursor.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_info(self) -> WarehouseInfo:
        """Get information about the Redshift connection."""
        if not self._connected:
            self.connect()

        self._cursor.execute("SELECT version()")
        version = self._cursor.fetchone()[0]

        self._cursor.execute("SELECT current_database(), current_schema()")
        row = self._cursor.fetchone()

        return WarehouseInfo(
            warehouse_type=self.warehouse_type,
            name=self.cluster_identifier or self.host,
            host=self.host,
            database=row[0],
            schema=row[1],
            connected=True,
            version=version,
            metadata={
                "region": self.region,
                "port": self.port
            }
        )

    def list_databases(self) -> List[str]:
        """List all accessible databases."""
        if not self._connected:
            self.connect()

        self._cursor.execute("""
            SELECT datname FROM pg_database
            WHERE datistemplate = false
            ORDER BY datname
        """)
        return [row[0] for row in self._cursor.fetchall()]

    def list_schemas(self, database: Optional[str] = None) -> List[str]:
        """List all schemas in a database."""
        if not self._connected:
            self.connect()

        self._cursor.execute("""
            SELECT schema_name FROM information_schema.schemata
            WHERE schema_name NOT LIKE 'pg_%'
            AND schema_name != 'information_schema'
            ORDER BY schema_name
        """)
        return [row[0] for row in self._cursor.fetchall()]

    def list_tables(
        self,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_views: bool = True
    ) -> List[TableInfo]:
        """List all tables in a schema."""
        if not self._connected:
            self.connect()

        sch = schema or self.schema

        table_types = "'BASE TABLE'"
        if include_views:
            table_types += ", 'VIEW'"

        self._cursor.execute(f"""
            SELECT table_name, table_schema, table_type
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_type IN ({table_types})
            ORDER BY table_name
        """, (sch,))

        tables = []
        for row in self._cursor.fetchall():
            tables.append(TableInfo(
                name=row[0],
                schema=row[1],
                database=self.database,
                table_type="VIEW" if row[2] == "VIEW" else "TABLE"
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

        sch = schema or self.schema
        full_name = f"{sch}.{self._validate_identifier(table_name)}"

        # Get row count
        self._cursor.execute(f"SELECT COUNT(*) FROM {full_name}")
        row_count = self._cursor.fetchone()[0]

        # Get table size
        self._cursor.execute("""
            SELECT size FROM svv_table_info
            WHERE schema = %s AND table = %s
        """, (sch, table_name))
        size_row = self._cursor.fetchone()
        size_bytes = size_row[0] * 1024 * 1024 if size_row else None  # Convert MB to bytes

        columns = self.get_table_schema(table_name, schema)

        return TableInfo(
            name=table_name,
            schema=sch,
            database=self.database,
            row_count=row_count,
            size_bytes=size_bytes,
            columns=columns,
            table_type="TABLE"
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

        sch = schema or self.schema

        self._cursor.execute("""
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                    ON tc.constraint_name = ku.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = %s
                AND tc.table_name = %s
            ) pk ON c.column_name = pk.column_name
            WHERE c.table_schema = %s AND c.table_name = %s
            ORDER BY c.ordinal_position
        """, (sch, table_name, sch, table_name))

        columns = []
        for row in self._cursor.fetchall():
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[2] == "YES",
                "default": row[3],
                "primary_key": row[4],
                "comment": None
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
        sch = schema or self.schema
        full_name = f"{self._validate_identifier(sch)}.{self._validate_identifier(table_name)}"

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

        start_time = time.time()

        try:
            if parameters:
                self._cursor.execute(query, parameters)
            else:
                self._cursor.execute(query)

            columns = [desc[0] for desc in self._cursor.description] if self._cursor.description else []

            if max_rows:
                rows = self._cursor.fetchmany(max_rows)
                truncated = len(rows) == max_rows
            else:
                rows = self._cursor.fetchall()
                truncated = False

            execution_time = (time.time() - start_time) * 1000

            # Convert to list of dicts
            row_dicts = [dict(zip(columns, row)) for row in rows]

            return QueryResult(
                columns=columns,
                rows=row_dicts,
                row_count=len(row_dicts),
                execution_time_ms=execution_time,
                truncated=truncated
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

        sch = schema or self.schema
        full_name = f"{self._validate_identifier(sch)}.{self._validate_identifier(table_name)}"

        self._cursor.execute(f"SELECT COUNT(*) FROM {full_name}")
        return self._cursor.fetchone()[0]

    def get_table_statistics(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get table statistics."""
        if not self._connected:
            self.connect()

        sch = schema or self.schema
        full_name = f"{self._validate_identifier(sch)}.{self._validate_identifier(table_name)}"

        # Get table-level stats from SVV_TABLE_INFO
        self._cursor.execute("""
            SELECT
                tbl_rows,
                size,
                pct_used,
                unsorted,
                stats_off
            FROM svv_table_info
            WHERE schema = %s AND table = %s
        """, (sch, table_name))

        table_stats = self._cursor.fetchone()

        stats = {
            "table_name": full_name,
            "row_count": int(table_stats[0]) if table_stats else self.get_row_count(table_name, schema),
            "size_mb": table_stats[1] if table_stats else None,
            "pct_used": table_stats[2] if table_stats else None,
            "unsorted_pct": table_stats[3] if table_stats else None,
            "stats_off_pct": table_stats[4] if table_stats else None,
            "column_stats": {}
        }

        # Get column statistics
        columns = self.get_table_schema(table_name, schema)

        for col in columns:
            col_name = col["name"]
            col_type = col["type"].upper()

            try:
                # Get null count and distinct count
                self._cursor.execute(f"""
                    SELECT
                        COUNT(*) - COUNT("{col_name}") as null_count,
                        COUNT(DISTINCT "{col_name}") as distinct_count
                    FROM {full_name}
                """)
                row = self._cursor.fetchone()

                col_stats = {
                    "null_count": row[0],
                    "distinct_count": row[1],
                    "data_type": col_type
                }

                # Get min/max for numeric and date types
                if any(t in col_type for t in ["INT", "FLOAT", "NUMERIC", "DECIMAL", "DATE", "TIME", "DOUBLE"]):
                    self._cursor.execute(f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM {full_name}')
                    minmax = self._cursor.fetchone()
                    col_stats["min"] = str(minmax[0]) if minmax[0] is not None else None
                    col_stats["max"] = str(minmax[1]) if minmax[1] is not None else None

                stats["column_stats"][col_name] = col_stats

            except Exception as e:
                logger.warning(f"Failed to get stats for column {col_name}: {str(e)}")
                stats["column_stats"][col_name] = {"error": str(e)}

        return stats

    def get_ddl(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_dependencies: bool = False
    ) -> str:
        """Get DDL for a table by constructing it from metadata."""
        if not self._connected:
            self.connect()

        sch = self._validate_identifier(schema) if schema else self.schema
        tbl = self._validate_identifier(table_name)

        # Get column definitions
        columns = self.get_table_schema(table_name, schema)

        if not columns:
            return ""

        # Build CREATE TABLE statement
        ddl_parts = [f'CREATE TABLE "{sch}"."{tbl}" (']

        col_defs = []
        for col in columns:
            col_def = f'    "{col["name"]}" {col["type"]}'
            if not col.get("nullable", True):
                col_def += " NOT NULL"
            if col.get("default"):
                col_def += f' DEFAULT {col["default"]}'
            col_defs.append(col_def)

        ddl_parts.append(",\n".join(col_defs))
        ddl_parts.append("\n);")

        ddl = "\n".join(ddl_parts)

        # Get distribution and sort keys
        try:
            self._cursor.execute("""
                SELECT
                    t.diststyle,
                    d.distkey,
                    s.sortkey1
                FROM svv_table_info t
                LEFT JOIN pg_table_def d ON t.table = d.tablename AND t.schema = d.schemaname AND d.distkey = true
                LEFT JOIN (
                    SELECT tablename, schemaname, attname as sortkey1
                    FROM pg_table_def
                    WHERE sortkey = 1
                ) s ON t.table = s.tablename AND t.schema = s.schemaname
                WHERE t.schema = %s AND t.table = %s
            """, (sch, tbl))
            dist_info = self._cursor.fetchone()

            if dist_info:
                dist_style = dist_info[0]
                dist_key = dist_info[1]
                sort_key = dist_info[2]

                if dist_style and dist_style != "AUTO(ALL)":
                    ddl += f"\n-- DISTSTYLE {dist_style}"
                if dist_key:
                    ddl += f"\n-- DISTKEY ({dist_key})"
                if sort_key:
                    ddl += f"\n-- SORTKEY ({sort_key})"
        except Exception as e:
            logger.warning(f"Could not get distribution info: {str(e)}")

        return ddl

    def get_query_history(
        self,
        days: int = 7,
        limit: int = 1000,
        database_filter: Optional[str] = None,
        schema_filter: Optional[str] = None,
        table_filter: Optional[str] = None,
        user_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get query history from STL_QUERY and STL_QUERYTEXT."""
        if not self._connected:
            self.connect()

        # Build query with filters
        query = """
            SELECT
                q.query as query_id,
                LISTAGG(qt.text, '') WITHIN GROUP (ORDER BY qt.sequence) as query_text,
                u.usename as user_name,
                q.database as database_name,
                q.starttime as start_time,
                q.endtime as end_time,
                q.elapsed / 1000 as execution_time_ms,
                q.label
            FROM stl_query q
            JOIN stl_querytext qt ON q.query = qt.query
            JOIN pg_user u ON q.userid = u.usesysid
            WHERE q.starttime >= DATEADD(day, -%s, CURRENT_DATE)
            AND q.userid > 1
        """

        params = [days]

        if database_filter:
            query += " AND q.database = %s"
            params.append(database_filter)

        if table_filter:
            query += " AND qt.text ILIKE %s"
            params.append(f"%{table_filter}%")

        if user_filter:
            query += " AND u.usename = %s"
            params.append(user_filter)

        query += """
            GROUP BY q.query, u.usename, q.database, q.starttime, q.endtime, q.elapsed, q.label
            ORDER BY q.starttime DESC
            LIMIT %s
        """
        params.append(limit)

        try:
            self._cursor.execute(query, params)
            rows = self._cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "query_id": str(row[0]),
                    "query_text": row[1],
                    "user_name": row[2],
                    "database_name": row[3],
                    "start_time": row[4],
                    "end_time": row[5],
                    "execution_time_ms": row[6],
                    "rows_produced": None,
                    "bytes_scanned": None,
                    "status": "success",
                    "tables_accessed": [],
                    "label": row[7]
                })

            return results

        except Exception as e:
            logger.warning(f"Could not get query history: {str(e)}")
            # Try simpler query
            try:
                simple_query = """
                    SELECT
                        query,
                        starttime,
                        endtime,
                        elapsed / 1000000.0 as elapsed_seconds,
                        label
                    FROM stl_query
                    WHERE userid > 1
                    AND starttime >= DATEADD(day, -%s, CURRENT_DATE)
                    ORDER BY starttime DESC
                    LIMIT %s
                """
                self._cursor.execute(simple_query, [days, limit])
                rows = self._cursor.fetchall()

                return [
                    {
                        "query_id": str(row[0]),
                        "query_text": "",
                        "user_name": None,
                        "start_time": row[1],
                        "end_time": row[2],
                        "execution_time_ms": row[3] * 1000 if row[3] else None,
                        "status": "success",
                        "tables_accessed": [],
                        "label": row[4]
                    }
                    for row in rows
                ]
            except Exception as e2:
                logger.error(f"Could not get query history: {str(e2)}")
                return []
