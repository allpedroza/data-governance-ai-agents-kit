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

    def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query history."""
        if not self._connected:
            self.connect()

        self._cursor.execute("""
            SELECT
                query,
                starttime,
                endtime,
                elapsed / 1000000.0 as elapsed_seconds,
                label
            FROM stl_query
            WHERE userid > 1
            ORDER BY starttime DESC
            LIMIT %s
        """, (limit,))

        return [
            {
                "query": row[0][:500],
                "start_time": row[1].isoformat() if row[1] else None,
                "end_time": row[2].isoformat() if row[2] else None,
                "elapsed_seconds": row[3],
                "label": row[4]
            }
            for row in self._cursor.fetchall()
        ]
